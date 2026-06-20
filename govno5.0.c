#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <assert.h>

#define EPS 1e-8
#define RAND_MAX_F ((float)RAND_MAX)
#define MAX_GRAD_NORM 1.0f
#define VERSION "5.0"

float frand() { return ((float)rand() / RAND_MAX_F) * 0.02f - 0.01f; }

// ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
void clip_gradients(float *grad, int n, float max_norm) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += grad[i] * grad[i];
    norm = sqrtf(norm);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; i++) grad[i] *= scale;
    }
}

void transpose_mat(float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

void zero_matrix(float *mat, int n) {
    memset(mat, 0, n * sizeof(float));
}

// ============ NEON-УСКОРЕННЫЕ ФУНКЦИИ ============
float dot_neon(float *a, float *b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4)
        sum = vmlaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    float res = vaddvq_f32(sum);
    for (; i < n; i++) res += a[i] * b[i];
    return res;
}

void matmul_neon(float *A, float *B, float *C, int m, int n, int p) {
    zero_matrix(C, m * p);
    
    int block_size = 32;
    for (int i0 = 0; i0 < m; i0 += block_size) {
        for (int j0 = 0; j0 < p; j0 += block_size) {
            for (int k0 = 0; k0 < n; k0 += block_size) {
                int imax = (i0 + block_size < m) ? i0 + block_size : m;
                int jmax = (j0 + block_size < p) ? j0 + block_size : p;
                int kmax = (k0 + block_size < n) ? k0 + block_size : n;
                
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        float aik = A[i * n + k];
                        if (fabsf(aik) < 1e-12f) continue;
                        int c_base = i * p;
                        int b_base = k * p;
                        
                        if (((uintptr_t)(B + b_base) & 15) == 0 && 
                            ((uintptr_t)(C + c_base) & 15) == 0) {
                            int j = j0;
                            float32x4_t a_vec = vdupq_n_f32(aik);
                            for (; j <= jmax - 4; j += 4) {
                                float32x4_t b_vec = vld1q_f32(B + b_base + j);
                                float32x4_t c_vec = vld1q_f32(C + c_base + j);
                                c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                                vst1q_f32(C + c_base + j, c_vec);
                            }
                            for (; j < jmax; j++) {
                                C[c_base + j] += aik * B[b_base + j];
                            }
                        } else {
                            for (int j = j0; j < jmax; j++) {
                                C[c_base + j] += aik * B[b_base + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (sum < EPS) sum = EPS;
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

// ============ СЛОВАРЬ ============
typedef struct {
    char **words;
    int size;
    int capacity;
} Vocab;

Vocab* vocab_create(int capacity) {
    Vocab *v = malloc(sizeof(Vocab));
    v->capacity = capacity;
    v->size = 0;
    v->words = malloc(capacity * sizeof(char*));
    v->words[0] = strdup("[UNK]");
    v->size = 1;
    return v;
}

int vocab_add(Vocab *v, char *word) {
    for (int i = 0; i < v->size; i++)
        if (strcmp(v->words[i], word) == 0) return i;
    if (v->size >= v->capacity) return 0;
    v->words[v->size] = strdup(word);
    return v->size++;
}

char* vocab_get(Vocab *v, int idx) {
    if (idx >= 0 && idx < v->size) return v->words[idx];
    return "[UNK]";
}

int vocab_size(Vocab *v) { return v->size; }

void vocab_free(Vocab *v) {
    for (int i = 0; i < v->size; i++) free(v->words[i]);
    free(v->words);
    free(v);
}

// ============ СТРУКТУРА МОДЕЛИ ============
typedef struct {
    float *embedding;
    float **wq, **wk, **wv, **wo;
    float **ffn_w1, **ffn_b1;
    float **ffn_w2, **ffn_b2;
    float *out_w, *out_b;
    
    float *m_emb, *v_emb;
    float **m_wq, **v_wq, **m_wk, **v_wk, **m_wv, **v_wv, **m_wo, **v_wo;
    float **m_ffn1, **v_ffn1, **m_ffn1_b, **v_ffn1_b;
    float **m_ffn2, **v_ffn2, **m_ffn2_b, **v_ffn2_b;
    float *m_out, *v_out, *m_out_b, *v_out_b;
    
    int t;
    int vs, dim, hidden, seq_len, num_layers;
} Model;

// ============ ПРОТОТИПЫ ============
void save_model(Model *m, Vocab *vocab, char *filename);
Model* load_model(Vocab **vocab, char *filename);
void adam_update(float *param, float *m, float *v, float *grad, int n, float lr, int t);
void forward_full(Model *m, int *tokens, int seq_len, float *logits);
void backward_full(Model *m, int *tokens, int seq_len, int target, float *logits,
                   float *d_embedding, float *d_out_w, float *d_out_b,
                   float **d_wq, float **d_wk, float **d_wv, float **d_wo,
                   float **d_ffn_w1, float **d_ffn_b1,
                   float **d_ffn_w2, float **d_ffn_b2);
void train(Model *m, Vocab *vocab, int *tokens, int data_len, int epochs, 
           char *model_file, float lr, int batch_size);

// ============ ADAM UPDATE ============
void adam_update(float *param, float *m, float *v, float *grad, int n, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f;
    float lr_t = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
    float eps = 1e-8f;
    
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / (1.0f - powf(beta1, t));
        float v_hat = v[i] / (1.0f - powf(beta2, t));
        param[i] -= lr_t * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ============ СОЗДАНИЕ МОДЕЛИ ============
Model* model_create(Vocab *vocab, int embed_dim, int hidden_dim, int seq_len, int num_layers) {
    Model *m = calloc(1, sizeof(Model));
    m->vs = vocab_size(vocab);
    m->dim = embed_dim;
    m->hidden = hidden_dim;
    m->seq_len = seq_len;
    m->num_layers = num_layers;
    m->t = 0;
    
    int emb_size = m->vs * m->dim;
    int attn_size = m->dim * m->dim;
    int ffn1_size = m->dim * m->hidden;
    int ffn2_size = m->hidden * m->dim;
    int out_size = m->dim * m->vs;
    
    m->embedding = calloc(emb_size, sizeof(float));
    m->out_w = calloc(out_size, sizeof(float));
    m->out_b = calloc(m->vs, sizeof(float));
    
    m->wq = malloc(num_layers * sizeof(float*));
    m->wk = malloc(num_layers * sizeof(float*));
    m->wv = malloc(num_layers * sizeof(float*));
    m->wo = malloc(num_layers * sizeof(float*));
    m->ffn_w1 = malloc(num_layers * sizeof(float*));
    m->ffn_b1 = malloc(num_layers * sizeof(float*));
    m->ffn_w2 = malloc(num_layers * sizeof(float*));
    m->ffn_b2 = malloc(num_layers * sizeof(float*));
    
    m->m_wq = malloc(num_layers * sizeof(float*));
    m->v_wq = malloc(num_layers * sizeof(float*));
    m->m_wk = malloc(num_layers * sizeof(float*));
    m->v_wk = malloc(num_layers * sizeof(float*));
    m->m_wv = malloc(num_layers * sizeof(float*));
    m->v_wv = malloc(num_layers * sizeof(float*));
    m->m_wo = malloc(num_layers * sizeof(float*));
    m->v_wo = malloc(num_layers * sizeof(float*));
    m->m_ffn1 = malloc(num_layers * sizeof(float*));
    m->v_ffn1 = malloc(num_layers * sizeof(float*));
    m->m_ffn1_b = malloc(num_layers * sizeof(float*));
    m->v_ffn1_b = malloc(num_layers * sizeof(float*));
    m->m_ffn2 = malloc(num_layers * sizeof(float*));
    m->v_ffn2 = malloc(num_layers * sizeof(float*));
    m->m_ffn2_b = malloc(num_layers * sizeof(float*));
    m->v_ffn2_b = malloc(num_layers * sizeof(float*));
    
    m->m_emb = calloc(emb_size, sizeof(float));
    m->v_emb = calloc(emb_size, sizeof(float));
    m->m_out = calloc(out_size, sizeof(float));
    m->v_out = calloc(out_size, sizeof(float));
    m->m_out_b = calloc(m->vs, sizeof(float));
    m->v_out_b = calloc(m->vs, sizeof(float));
    
    for (int l = 0; l < num_layers; l++) {
        m->wq[l] = calloc(attn_size, sizeof(float));
        m->wk[l] = calloc(attn_size, sizeof(float));
        m->wv[l] = calloc(attn_size, sizeof(float));
        m->wo[l] = calloc(attn_size, sizeof(float));
        m->ffn_w1[l] = calloc(ffn1_size, sizeof(float));
        m->ffn_b1[l] = calloc(m->hidden, sizeof(float));
        m->ffn_w2[l] = calloc(ffn2_size, sizeof(float));
        m->ffn_b2[l] = calloc(m->dim, sizeof(float));
        
        m->m_wq[l] = calloc(attn_size, sizeof(float));
        m->v_wq[l] = calloc(attn_size, sizeof(float));
        m->m_wk[l] = calloc(attn_size, sizeof(float));
        m->v_wk[l] = calloc(attn_size, sizeof(float));
        m->m_wv[l] = calloc(attn_size, sizeof(float));
        m->v_wv[l] = calloc(attn_size, sizeof(float));
        m->m_wo[l] = calloc(attn_size, sizeof(float));
        m->v_wo[l] = calloc(attn_size, sizeof(float));
        m->m_ffn1[l] = calloc(ffn1_size, sizeof(float));
        m->v_ffn1[l] = calloc(ffn1_size, sizeof(float));
        m->m_ffn1_b[l] = calloc(m->hidden, sizeof(float));
        m->v_ffn1_b[l] = calloc(m->hidden, sizeof(float));
        m->m_ffn2[l] = calloc(ffn2_size, sizeof(float));
        m->v_ffn2[l] = calloc(ffn2_size, sizeof(float));
        m->m_ffn2_b[l] = calloc(m->dim, sizeof(float));
        m->v_ffn2_b[l] = calloc(m->dim, sizeof(float));
        
        float scale_q = sqrtf(2.0f / m->dim);
        float scale_ffn = sqrtf(2.0f / m->dim);
        
        for (int i = 0; i < attn_size; i++) {
            m->wq[l][i] = frand() * scale_q;
            m->wk[l][i] = frand() * scale_q;
            m->wv[l][i] = frand() * scale_q;
            m->wo[l][i] = frand() * scale_q;
        }
        for (int i = 0; i < ffn1_size; i++) m->ffn_w1[l][i] = frand() * scale_ffn;
        for (int i = 0; i < ffn2_size; i++) m->ffn_w2[l][i] = frand() * scale_ffn;
    }
    
    float scale_emb = sqrtf(2.0f / m->dim);
    for (int i = 0; i < emb_size; i++) m->embedding[i] = frand() * scale_emb;
    for (int i = 0; i < out_size; i++) m->out_w[i] = frand() * 0.01f;
    
    return m;
}

// ============ ВЫВОД РАЗМЕРА МОДЕЛИ ============
void print_model_size(Model *m) {
    int emb_size = m->vs * m->dim;
    int attn_size = m->dim * m->dim;
    int ffn1_size = m->dim * m->hidden;
    int ffn2_size = m->hidden * m->dim;
    int out_size = m->dim * m->vs;
    
    int total_params = emb_size + attn_size * 4 * m->num_layers + 
                       ffn1_size * m->num_layers + m->hidden * m->num_layers + 
                       ffn2_size * m->num_layers + m->dim * m->num_layers + 
                       out_size + m->vs;
    float params_mb = total_params * sizeof(float) / (1024.0f * 1024.0f);
    
    printf("\n╔════════════════════════════════════════════════════╗\n");
    printf("║          NEURAL LANGUAGE v%s                      ║\n", VERSION);
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Размер словаря:          %-28d ║\n", m->vs);
    printf("║  Размер эмбеддингов:      %-28d ║\n", m->dim);
    printf("║  Скрытый размер (ffn):    %-28d ║\n", m->hidden);
    printf("║  Длина контекста:         %-28d ║\n", m->seq_len);
    printf("║  Кол-во слоёв:            %-28d ║\n", m->num_layers);
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Всего параметров:        %-28d ║\n", total_params);
    printf("║  Размер в памяти (fp32):  %.2f MB%-22s ║\n", params_mb, "");
    printf("╚════════════════════════════════════════════════════╝\n");
}

// ============ ПРЯМОЙ ПРОХОД ============
void forward_full(Model *m, int *tokens, int seq_len, float *logits) {
    float *x = malloc(seq_len * m->dim * sizeof(float));
    if (!x) { printf("malloc x failed\n"); exit(1); }
    
    for (int i = 0; i < seq_len; i++) {
        int t = tokens[i] % m->vs;
        memcpy(x + i * m->dim, m->embedding + t * m->dim, m->dim * sizeof(float));
    }
    
    float *cur = x;
    
    for (int l = 0; l < m->num_layers; l++) {
        float *q = malloc(seq_len * m->dim * sizeof(float));
        float *k = malloc(seq_len * m->dim * sizeof(float));
        float *v = malloc(seq_len * m->dim * sizeof(float));
        float *attn_out = malloc(seq_len * m->dim * sizeof(float));
        float *attn_proj = malloc(seq_len * m->dim * sizeof(float));
        float *ffn_hidden = malloc(seq_len * m->hidden * sizeof(float));
        float *ffn_out = malloc(seq_len * m->dim * sizeof(float));
        
        matmul_neon(cur, m->wq[l], q, seq_len, m->dim, m->dim);
        matmul_neon(cur, m->wk[l], k, seq_len, m->dim, m->dim);
        matmul_neon(cur, m->wv[l], v, seq_len, m->dim, m->dim);
        
        float scale = 1.0f / sqrtf(m->dim);
        float *scores = malloc(seq_len * seq_len * sizeof(float));
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] = dot_neon(q + i*m->dim, k + j*m->dim, m->dim) * scale;
            }
            for (int j = i+1; j < seq_len; j++) scores[i * seq_len + j] = -1e9f;
            softmax(scores + i * seq_len, seq_len);
        }
        
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < m->dim; d++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++) sum += scores[i * seq_len + j] * v[j * m->dim + d];
                attn_out[i * m->dim + d] = sum;
            }
        }
        
        matmul_neon(attn_out, m->wo[l], attn_proj, seq_len, m->dim, m->dim);
        for (int i = 0; i < seq_len * m->dim; i++) attn_proj[i] += cur[i];
        
        matmul_neon(attn_proj, m->ffn_w1[l], ffn_hidden, seq_len, m->dim, m->hidden);
        for (int i = 0; i < seq_len * m->hidden; i++) {
            int idx = i % m->hidden;
            ffn_hidden[i] += m->ffn_b1[l][idx];
            if (ffn_hidden[i] < 0) ffn_hidden[i] = 0;
        }
        
        matmul_neon(ffn_hidden, m->ffn_w2[l], ffn_out, seq_len, m->hidden, m->dim);
        for (int i = 0; i < seq_len * m->dim; i++) {
            int idx = i % m->dim;
            ffn_out[i] += attn_proj[i] + m->ffn_b2[l][idx];
        }
        
        if (l > 0) free(cur);
        free(q); free(k); free(v); free(attn_out); free(attn_proj);
        free(ffn_hidden); free(scores);
        
        cur = ffn_out;
    }
    
    for (int j = 0; j < m->vs; j++) {
        float sum = m->out_b[j];
        for (int d = 0; d < m->dim; d++) {
            sum += cur[(seq_len-1) * m->dim + d] * m->out_w[d * m->vs + j];
        }
        logits[j] = sum;
    }
    softmax(logits, m->vs);
    
    free(cur);
}

// ============ ОБРАТНЫЙ ПРОХОД (ПОЛНЫЙ) ============
void backward_full(Model *m, int *tokens, int seq_len, int target, float *logits,
                   float *d_embedding, float *d_out_w, float *d_out_b,
                   float **d_wq, float **d_wk, float **d_wv, float **d_wo,
                   float **d_ffn_w1, float **d_ffn_b1,
                   float **d_ffn_w2, float **d_ffn_b2) {
    
    // --- 1. Градиент лосса ---
    float *d_logits = malloc(m->vs * sizeof(float));
    for (int i = 0; i < m->vs; i++) d_logits[i] = logits[i];
    d_logits[target] -= 1.0f;
    
    // --- 2. Градиенты выходного слоя ---
    for (int i = 0; i < m->vs; i++) {
        d_out_b[i] += d_logits[i];
        for (int d = 0; d < m->dim; d++)
            d_out_w[d * m->vs + i] += d_logits[i];
    }
    
    // --- 3. Градиент последнего скрытого состояния ---
    float *d_cur = malloc(seq_len * m->dim * sizeof(float));
    zero_matrix(d_cur, seq_len * m->dim);
    
    for (int d = 0; d < m->dim; d++) {
        float grad = 0;
        for (int i = 0; i < m->vs; i++)
            grad += d_logits[i] * m->out_w[d * m->vs + i];
        d_cur[(seq_len-1) * m->dim + d] = grad;
    }
    free(d_logits);
    
    // --- 4. Получение входных эмбеддингов ---
    float *x = malloc(seq_len * m->dim * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        int t = tokens[i] % m->vs;
        memcpy(x + i * m->dim, m->embedding + t * m->dim, m->dim * sizeof(float));
    }
    
    // --- 5. Сохранение промежуточных значений ---
    float **layer_x = malloc((m->num_layers + 1) * sizeof(float*));
    float **layer_q = malloc(m->num_layers * sizeof(float*));
    float **layer_k = malloc(m->num_layers * sizeof(float*));
    float **layer_v = malloc(m->num_layers * sizeof(float*));
    float **layer_scores = malloc(m->num_layers * sizeof(float*));
    float **layer_attn_out = malloc(m->num_layers * sizeof(float*));
    float **layer_attn_proj = malloc(m->num_layers * sizeof(float*));
    float **layer_ffn_hidden = malloc(m->num_layers * sizeof(float*));
    float **layer_ffn_out = malloc(m->num_layers * sizeof(float*));
    
    layer_x[0] = x;
    
    for (int l = 0; l < m->num_layers; l++) {
        float *q = malloc(seq_len * m->dim * sizeof(float));
        float *k = malloc(seq_len * m->dim * sizeof(float));
        float *v = malloc(seq_len * m->dim * sizeof(float));
        float *attn_out = malloc(seq_len * m->dim * sizeof(float));
        float *attn_proj = malloc(seq_len * m->dim * sizeof(float));
        float *ffn_hidden = malloc(seq_len * m->hidden * sizeof(float));
        float *ffn_out = malloc(seq_len * m->dim * sizeof(float));
        float *scores = malloc(seq_len * seq_len * sizeof(float));
        
        matmul_neon(layer_x[l], m->wq[l], q, seq_len, m->dim, m->dim);
        matmul_neon(layer_x[l], m->wk[l], k, seq_len, m->dim, m->dim);
        matmul_neon(layer_x[l], m->wv[l], v, seq_len, m->dim, m->dim);
        
        float scale = 1.0f / sqrtf(m->dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] = dot_neon(q + i*m->dim, k + j*m->dim, m->dim) * scale;
            }
            for (int j = i+1; j < seq_len; j++) scores[i * seq_len + j] = -1e9f;
            softmax(scores + i * seq_len, seq_len);
        }
        
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < m->dim; d++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++) sum += scores[i * seq_len + j] * v[j * m->dim + d];
                attn_out[i * m->dim + d] = sum;
            }
        }
        
        matmul_neon(attn_out, m->wo[l], attn_proj, seq_len, m->dim, m->dim);
        for (int i = 0; i < seq_len * m->dim; i++) attn_proj[i] += layer_x[l][i];
        
        matmul_neon(attn_proj, m->ffn_w1[l], ffn_hidden, seq_len, m->dim, m->hidden);
        for (int i = 0; i < seq_len * m->hidden; i++) {
            int idx = i % m->hidden;
            ffn_hidden[i] += m->ffn_b1[l][idx];
            if (ffn_hidden[i] < 0) ffn_hidden[i] = 0;
        }
        
        matmul_neon(ffn_hidden, m->ffn_w2[l], ffn_out, seq_len, m->hidden, m->dim);
        for (int i = 0; i < seq_len * m->dim; i++) {
            int idx = i % m->dim;
            ffn_out[i] += attn_proj[i] + m->ffn_b2[l][idx];
        }
        
        layer_q[l] = q;
        layer_k[l] = k;
        layer_v[l] = v;
        layer_scores[l] = scores;
        layer_attn_out[l] = attn_out;
        layer_attn_proj[l] = attn_proj;
        layer_ffn_hidden[l] = ffn_hidden;
        layer_ffn_out[l] = ffn_out;
        layer_x[l+1] = ffn_out;
    }
    
    // --- 6. Обратный проход ---
    for (int l = m->num_layers - 1; l >= 0; l--) {
        float *d_attn_proj = malloc(seq_len * m->dim * sizeof(float));
        float *d_ffn_out = malloc(seq_len * m->dim * sizeof(float));
        float *d_ffn_hidden = malloc(seq_len * m->hidden * sizeof(float));
        float *d_attn_out = malloc(seq_len * m->dim * sizeof(float));
        float *d_q = malloc(seq_len * m->dim * sizeof(float));
        float *d_k = malloc(seq_len * m->dim * sizeof(float));
        float *d_v = malloc(seq_len * m->dim * sizeof(float));
        float *d_scores = malloc(seq_len * seq_len * sizeof(float));
        float *d_x_prev = malloc(seq_len * m->dim * sizeof(float));
        
        float *grad_ffn_w1 = malloc(m->dim * m->hidden * sizeof(float));
        float *grad_ffn_b1 = malloc(m->hidden * sizeof(float));
        float *grad_ffn_w2 = malloc(m->hidden * m->dim * sizeof(float));
        float *grad_ffn_b2 = malloc(m->dim * sizeof(float));
        float *grad_wq = malloc(m->dim * m->dim * sizeof(float));
        float *grad_wk = malloc(m->dim * m->dim * sizeof(float));
        float *grad_wv = malloc(m->dim * m->dim * sizeof(float));
        float *grad_wo = malloc(m->dim * m->dim * sizeof(float));
        
        zero_matrix(d_attn_proj, seq_len * m->dim);
        zero_matrix(d_ffn_out, seq_len * m->dim);
        zero_matrix(d_ffn_hidden, seq_len * m->hidden);
        zero_matrix(d_attn_out, seq_len * m->dim);
        zero_matrix(d_q, seq_len * m->dim);
        zero_matrix(d_k, seq_len * m->dim);
        zero_matrix(d_v, seq_len * m->dim);
        zero_matrix(d_scores, seq_len * seq_len);
        zero_matrix(d_x_prev, seq_len * m->dim);
        zero_matrix(grad_ffn_w1, m->dim * m->hidden);
        zero_matrix(grad_ffn_b1, m->hidden);
        zero_matrix(grad_ffn_w2, m->hidden * m->dim);
        zero_matrix(grad_ffn_b2, m->dim);
        zero_matrix(grad_wq, m->dim * m->dim);
        zero_matrix(grad_wk, m->dim * m->dim);
        zero_matrix(grad_wv, m->dim * m->dim);
        zero_matrix(grad_wo, m->dim * m->dim);
        
        if (l == m->num_layers - 1) {
            memcpy(d_ffn_out, d_cur, seq_len * m->dim * sizeof(float));
        }
        
        // FFN градиенты (второй слой)
        float *ffn_hidden_t = malloc(m->hidden * seq_len * sizeof(float));
        transpose_mat(layer_ffn_hidden[l], ffn_hidden_t, seq_len, m->hidden);
        matmul_neon(ffn_hidden_t, d_ffn_out, grad_ffn_w2, m->hidden, seq_len, m->dim);
        for (int i = 0; i < m->dim; i++) {
            float sum = 0;
            for (int j = 0; j < seq_len; j++) sum += d_ffn_out[j * m->dim + i];
            grad_ffn_b2[i] += sum;
        }
        free(ffn_hidden_t);
        
        float *d_ffn_hidden_pre = malloc(seq_len * m->hidden * sizeof(float));
        float *ffn_w2_t = malloc(m->dim * m->hidden * sizeof(float));
        transpose_mat(m->ffn_w2[l], ffn_w2_t, m->hidden, m->dim);
        matmul_neon(d_ffn_out, ffn_w2_t, d_ffn_hidden_pre, seq_len, m->dim, m->hidden);
        free(ffn_w2_t);
        
        for (int i = 0; i < seq_len * m->hidden; i++) {
            d_ffn_hidden[i] = (layer_ffn_hidden[l][i] > 0) ? d_ffn_hidden_pre[i] : 0;
        }
        free(d_ffn_hidden_pre);
        
        // FFN градиенты (первый слой)
        float *attn_proj_t = malloc(m->dim * seq_len * sizeof(float));
        transpose_mat(layer_attn_proj[l], attn_proj_t, seq_len, m->dim);
        matmul_neon(attn_proj_t, d_ffn_hidden, grad_ffn_w1, m->dim, seq_len, m->hidden);
        for (int i = 0; i < m->hidden; i++) {
            float sum = 0;
            for (int j = 0; j < seq_len; j++) sum += d_ffn_hidden[j * m->hidden + i];
            grad_ffn_b1[i] += sum;
        }
        free(attn_proj_t);
        
        // Градиент для attn_proj
        float *d_attn_proj_input = malloc(seq_len * m->dim * sizeof(float));
        float *ffn_w1_t = malloc(m->hidden * m->dim * sizeof(float));
        transpose_mat(m->ffn_w1[l], ffn_w1_t, m->dim, m->hidden);
        matmul_neon(d_ffn_hidden, ffn_w1_t, d_attn_proj_input, seq_len, m->hidden, m->dim);
        free(ffn_w1_t);
        
        for (int i = 0; i < seq_len * m->dim; i++) {
            d_attn_proj[i] = d_attn_proj_input[i];
            if (l > 0) d_x_prev[i] = d_attn_proj_input[i];
        }
        free(d_attn_proj_input);
        
        // WO градиенты
        float *attn_out_t = malloc(m->dim * seq_len * sizeof(float));
        transpose_mat(layer_attn_out[l], attn_out_t, seq_len, m->dim);
        matmul_neon(attn_out_t, d_attn_proj, grad_wo, m->dim, seq_len, m->dim);
        free(attn_out_t);
        
        // Градиент для attn_out
        float *wo_t = malloc(m->dim * m->dim * sizeof(float));
        transpose_mat(m->wo[l], wo_t, m->dim, m->dim);
        matmul_neon(d_attn_proj, wo_t, d_attn_out, seq_len, m->dim, m->dim);
        free(wo_t);
        
        // Градиенты для scores
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float sum = 0;
                for (int d = 0; d < m->dim; d++) {
                    sum += d_attn_out[i * m->dim + d] * layer_v[l][j * m->dim + d];
                }
                d_scores[i * seq_len + j] = sum;
            }
        }
        
        // Softmax градиент
        for (int i = 0; i < seq_len; i++) {
            float *s = layer_scores[l] + i * seq_len;
            float *ds = d_scores + i * seq_len;
            float sum = 0;
            for (int j = 0; j < seq_len; j++) sum += ds[j] * s[j];
            for (int j = 0; j < seq_len; j++) {
                ds[j] = s[j] * (ds[j] - sum);
            }
            for (int j = i+1; j < seq_len; j++) ds[j] = 0;
        }
        
        // Градиенты для Q, K, V
        float scale = 1.0f / sqrtf(m->dim);
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < m->dim; d++) {
                float grad_q = 0, grad_k = 0;
                for (int j = 0; j < seq_len; j++) {
                    grad_q += d_scores[i * seq_len + j] * scale * layer_k[l][j * m->dim + d];
                    grad_k += d_scores[j * seq_len + i] * scale * layer_q[l][j * m->dim + d];
                }
                d_q[i * m->dim + d] = grad_q;
                d_k[i * m->dim + d] = grad_k;
            }
        }
        
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < m->dim; d++) {
                float sum = 0;
                for (int j = 0; j < seq_len; j++) {
                    sum += d_attn_out[j * m->dim + d] * layer_scores[l][j * seq_len + i];
                }
                d_v[i * m->dim + d] = sum;
            }
        }
        
        // Градиенты для WQ, WK, WV
        float *x_prev_t = malloc(m->dim * seq_len * sizeof(float));
        transpose_mat(layer_x[l], x_prev_t, seq_len, m->dim);
        matmul_neon(x_prev_t, d_q, grad_wq, m->dim, seq_len, m->dim);
        matmul_neon(x_prev_t, d_k, grad_wk, m->dim, seq_len, m->dim);
        matmul_neon(x_prev_t, d_v, grad_wv, m->dim, seq_len, m->dim);
        free(x_prev_t);
        
        // Градиент для x_prev
        float *wq_t = malloc(m->dim * m->dim * sizeof(float));
        float *wk_t = malloc(m->dim * m->dim * sizeof(float));
        float *wv_t = malloc(m->dim * m->dim * sizeof(float));
        transpose_mat(m->wq[l], wq_t, m->dim, m->dim);
        transpose_mat(m->wk[l], wk_t, m->dim, m->dim);
        transpose_mat(m->wv[l], wv_t, m->dim, m->dim);
        
        float *d_from_q = malloc(seq_len * m->dim * sizeof(float));
        float *d_from_k = malloc(seq_len * m->dim * sizeof(float));
        float *d_from_v = malloc(seq_len * m->dim * sizeof(float));
        matmul_neon(d_q, wq_t, d_from_q, seq_len, m->dim, m->dim);
        matmul_neon(d_k, wk_t, d_from_k, seq_len, m->dim, m->dim);
        matmul_neon(d_v, wv_t, d_from_v, seq_len, m->dim, m->dim);
        free(wq_t); free(wk_t); free(wv_t);
        
        for (int i = 0; i < seq_len * m->dim; i++) {
            d_x_prev[i] += d_from_q[i] + d_from_k[i] + d_from_v[i];
        }
        free(d_from_q); free(d_from_k); free(d_from_v);
        
        // Клиппинг градиентов
        clip_gradients(grad_wq, m->dim * m->dim, MAX_GRAD_NORM);
        clip_gradients(grad_wk, m->dim * m->dim, MAX_GRAD_NORM);
        clip_gradients(grad_wv, m->dim * m->dim, MAX_GRAD_NORM);
        clip_gradients(grad_wo, m->dim * m->dim, MAX_GRAD_NORM);
        clip_gradients(grad_ffn_w1, m->dim * m->hidden, MAX_GRAD_NORM);
        clip_gradients(grad_ffn_b1, m->hidden, MAX_GRAD_NORM);
        clip_gradients(grad_ffn_w2, m->hidden * m->dim, MAX_GRAD_NORM);
        clip_gradients(grad_ffn_b2, m->dim, MAX_GRAD_NORM);
        
        // Накапливаем градиенты
        for (int i = 0; i < m->dim * m->dim; i++) {
            d_wq[l][i] += grad_wq[i];
            d_wk[l][i] += grad_wk[i];
            d_wv[l][i] += grad_wv[i];
            d_wo[l][i] += grad_wo[i];
        }
        for (int i = 0; i < m->dim * m->hidden; i++) {
            d_ffn_w1[l][i] += grad_ffn_w1[i];
        }
        for (int i = 0; i < m->hidden * m->dim; i++) {
            d_ffn_w2[l][i] += grad_ffn_w2[i];
        }
        for (int i = 0; i < m->hidden; i++) {
            d_ffn_b1[l][i] += grad_ffn_b1[i];
        }
        for (int i = 0; i < m->dim; i++) {
            d_ffn_b2[l][i] += grad_ffn_b2[i];
        }
        
        if (l > 0) {
            memcpy(d_cur, d_x_prev, seq_len * m->dim * sizeof(float));
        } else {
            for (int i = 0; i < seq_len; i++) {
                int token = tokens[i] % m->vs;
                for (int d = 0; d < m->dim; d++) {
                    d_embedding[token * m->dim + d] += d_x_prev[i * m->dim + d];
                }
            }
        }
        
        free(d_attn_proj); free(d_ffn_out); free(d_ffn_hidden);
        free(d_attn_out); free(d_q); free(d_k); free(d_v);
        free(d_scores); free(d_x_prev);
        free(grad_ffn_w1); free(grad_ffn_b1); free(grad_ffn_w2); free(grad_ffn_b2);
        free(grad_wq); free(grad_wk); free(grad_wv); free(grad_wo);
    }
    
    // --- 7. Освобождение ---
    for (int l = 0; l < m->num_layers; l++) {
        free(layer_q[l]); free(layer_k[l]); free(layer_v[l]);
        free(layer_scores[l]); free(layer_attn_out[l]);
        free(layer_attn_proj[l]); free(layer_ffn_hidden[l]);
        free(layer_ffn_out[l]);
    }
    free(layer_x[0]);
    free(layer_x); free(layer_q); free(layer_k); free(layer_v);
    free(layer_scores); free(layer_attn_out); free(layer_attn_proj);
    free(layer_ffn_hidden); free(layer_ffn_out);
    free(d_cur);
}

// ============ ОБУЧЕНИЕ ============
void train(Model *m, Vocab *vocab, int *tokens, int data_len, int epochs, 
           char *model_file, float lr, int batch_size) {
    int total_windows = data_len - m->seq_len;
    if (total_windows <= 0) return;
    printf("Окон для обучения: %d\n", total_windows);
    
    int emb_size = m->vs * m->dim;
    int out_size = m->dim * m->vs;
    int attn_size = m->dim * m->dim;
    int ffn1_size = m->dim * m->hidden;
    int ffn2_size = m->hidden * m->dim;
    
    float best_loss = 1e9f;
    int no_improve = 0;
    
    float *d_emb = calloc(emb_size, sizeof(float));
    float *d_out = calloc(out_size, sizeof(float));
    float *d_out_b = calloc(m->vs, sizeof(float));
    
    float **d_wq = malloc(m->num_layers * sizeof(float*));
    float **d_wk = malloc(m->num_layers * sizeof(float*));
    float **d_wv = malloc(m->num_layers * sizeof(float*));
    float **d_wo = malloc(m->num_layers * sizeof(float*));
    float **d_ffn_w1 = malloc(m->num_layers * sizeof(float*));
    float **d_ffn_b1 = malloc(m->num_layers * sizeof(float*));
    float **d_ffn_w2 = malloc(m->num_layers * sizeof(float*));
    float **d_ffn_b2 = malloc(m->num_layers * sizeof(float*));
    
    for (int l = 0; l < m->num_layers; l++) {
        d_wq[l] = calloc(attn_size, sizeof(float));
        d_wk[l] = calloc(attn_size, sizeof(float));
        d_wv[l] = calloc(attn_size, sizeof(float));
        d_wo[l] = calloc(attn_size, sizeof(float));
        d_ffn_w1[l] = calloc(ffn1_size, sizeof(float));
        d_ffn_b1[l] = calloc(m->hidden, sizeof(float));
        d_ffn_w2[l] = calloc(ffn2_size, sizeof(float));
        d_ffn_b2[l] = calloc(m->dim, sizeof(float));
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = 0;
        
        zero_matrix(d_emb, emb_size);
        zero_matrix(d_out, out_size);
        zero_matrix(d_out_b, m->vs);
        for (int l = 0; l < m->num_layers; l++) {
            zero_matrix(d_wq[l], attn_size);
            zero_matrix(d_wk[l], attn_size);
            zero_matrix(d_wv[l], attn_size);
            zero_matrix(d_wo[l], attn_size);
            zero_matrix(d_ffn_w1[l], ffn1_size);
            zero_matrix(d_ffn_b1[l], m->hidden);
            zero_matrix(d_ffn_w2[l], ffn2_size);
            zero_matrix(d_ffn_b2[l], m->dim);
        }
        
        for (int start = 0; start < total_windows; start += batch_size) {
            int end = (start + batch_size < total_windows) ? start + batch_size : total_windows;
            
            for (int w = start; w < end; w++) {
                float *logits = malloc(m->vs * sizeof(float));
                forward_full(m, tokens + w, m->seq_len, logits);
                
                int target = tokens[w + m->seq_len];
                float loss = -logf(logits[target] + EPS);
                total_loss += loss;
                
                backward_full(m, tokens + w, m->seq_len, target, logits,
                            d_emb, d_out, d_out_b,
                            d_wq, d_wk, d_wv, d_wo,
                            d_ffn_w1, d_ffn_b1, d_ffn_w2, d_ffn_b2);
                free(logits);
            }
            
            float lr_adj = lr / (end - start);
            
            m->t++;
            adam_update(m->embedding, m->m_emb, m->v_emb, d_emb, emb_size, lr_adj, m->t);
            adam_update(m->out_w, m->m_out, m->v_out, d_out, out_size, lr_adj, m->t);
            adam_update(m->out_b, m->m_out_b, m->v_out_b, d_out_b, m->vs, lr_adj, m->t);
            
            for (int l = 0; l < m->num_layers; l++) {
                adam_update(m->wq[l], m->m_wq[l], m->v_wq[l], d_wq[l], attn_size, lr_adj, m->t);
                adam_update(m->wk[l], m->m_wk[l], m->v_wk[l], d_wk[l], attn_size, lr_adj, m->t);
                adam_update(m->wv[l], m->m_wv[l], m->v_wv[l], d_wv[l], attn_size, lr_adj, m->t);
                adam_update(m->wo[l], m->m_wo[l], m->v_wo[l], d_wo[l], attn_size, lr_adj, m->t);
                adam_update(m->ffn_w1[l], m->m_ffn1[l], m->v_ffn1[l], d_ffn_w1[l], ffn1_size, lr_adj, m->t);
                adam_update(m->ffn_b1[l], m->m_ffn1_b[l], m->v_ffn1_b[l], d_ffn_b1[l], m->hidden, lr_adj, m->t);
                adam_update(m->ffn_w2[l], m->m_ffn2[l], m->v_ffn2[l], d_ffn_w2[l], ffn2_size, lr_adj, m->t);
                adam_update(m->ffn_b2[l], m->m_ffn2_b[l], m->v_ffn2_b[l], d_ffn_b2[l], m->dim, lr_adj, m->t);
            }
            
            zero_matrix(d_emb, emb_size);
            zero_matrix(d_out, out_size);
            zero_matrix(d_out_b, m->vs);
            for (int l = 0; l < m->num_layers; l++) {
                zero_matrix(d_wq[l], attn_size);
                zero_matrix(d_wk[l], attn_size);
                zero_matrix(d_wv[l], attn_size);
                zero_matrix(d_wo[l], attn_size);
                zero_matrix(d_ffn_w1[l], ffn1_size);
                zero_matrix(d_ffn_b1[l], m->hidden);
                zero_matrix(d_ffn_w2[l], ffn2_size);
                zero_matrix(d_ffn_b2[l], m->dim);
            }
            
            num_batches++;
        }
        
        float avg_loss = total_loss / total_windows;
        printf("Epoch %d, Loss: %.4f, Batches: %d\n", epoch, avg_loss, num_batches);
        
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            no_improve = 0;
            save_model(m, vocab, model_file);
            printf("  ✓ Модель сохранена (loss: %.4f)\n", avg_loss);
        } else {
            no_improve++;
            if (no_improve >= 5) {
                printf("Ранняя остановка (no_improve=%d)\n", no_improve);
                break;
            }
        }
    }
    
    free(d_emb); free(d_out); free(d_out_b);
    for (int l = 0; l < m->num_layers; l++) {
        free(d_wq[l]); free(d_wk[l]); free(d_wv[l]); free(d_wo[l]);
        free(d_ffn_w1[l]); free(d_ffn_b1[l]);
        free(d_ffn_w2[l]); free(d_ffn_b2[l]);
    }
    free(d_wq); free(d_wk); free(d_wv); free(d_wo);
    free(d_ffn_w1); free(d_ffn_b1); free(d_ffn_w2); free(d_ffn_b2);
}

// ============ СОХРАНЕНИЕ/ЗАГРУЗКА МОДЕЛИ ============
void save_model(Model *m, Vocab *vocab, char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(&m->vs, sizeof(int), 1, f);
    fwrite(&m->dim, sizeof(int), 1, f);
    fwrite(&m->hidden, sizeof(int), 1, f);
    fwrite(&m->seq_len, sizeof(int), 1, f);
    fwrite(&m->num_layers, sizeof(int), 1, f);
    fwrite(&m->t, sizeof(int), 1, f);
    
    fwrite(m->embedding, sizeof(float), m->vs * m->dim, f);
    for (int l = 0; l < m->num_layers; l++) {
        fwrite(m->wq[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->wk[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->wv[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->wo[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->ffn_w1[l], sizeof(float), m->dim * m->hidden, f);
        fwrite(m->ffn_b1[l], sizeof(float), m->hidden, f);
        fwrite(m->ffn_w2[l], sizeof(float), m->hidden * m->dim, f);
        fwrite(m->ffn_b2[l], sizeof(float), m->dim, f);
    }
    fwrite(m->out_w, sizeof(float), m->dim * m->vs, f);
    fwrite(m->out_b, sizeof(float), m->vs, f);
    
    fwrite(m->m_emb, sizeof(float), m->vs * m->dim, f);
    fwrite(m->v_emb, sizeof(float), m->vs * m->dim, f);
    for (int l = 0; l < m->num_layers; l++) {
        fwrite(m->m_wq[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->v_wq[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->m_wk[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->v_wk[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->m_wv[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->v_wv[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->m_wo[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->v_wo[l], sizeof(float), m->dim * m->dim, f);
        fwrite(m->m_ffn1[l], sizeof(float), m->dim * m->hidden, f);
        fwrite(m->v_ffn1[l], sizeof(float), m->dim * m->hidden, f);
        fwrite(m->m_ffn1_b[l], sizeof(float), m->hidden, f);
        fwrite(m->v_ffn1_b[l], sizeof(float), m->hidden, f);
        fwrite(m->m_ffn2[l], sizeof(float), m->hidden * m->dim, f);
        fwrite(m->v_ffn2[l], sizeof(float), m->hidden * m->dim, f);
        fwrite(m->m_ffn2_b[l], sizeof(float), m->dim, f);
        fwrite(m->v_ffn2_b[l], sizeof(float), m->dim, f);
    }
    fwrite(m->m_out, sizeof(float), m->dim * m->vs, f);
    fwrite(m->v_out, sizeof(float), m->dim * m->vs, f);
    fwrite(m->m_out_b, sizeof(float), m->vs, f);
    fwrite(m->v_out_b, sizeof(float), m->vs, f);
    
    fwrite(&vocab->size, sizeof(int), 1, f);
    for (int i = 0; i < vocab->size; i++) {
        int len = strlen(vocab->words[i]) + 1;
        fwrite(&len, sizeof(int), 1, f);
        fwrite(vocab->words[i], sizeof(char), len, f);
    }
    
    fclose(f);
    printf("Модель сохранена: %s (%d слов)\n", filename, vocab->size);
}

Model* load_model(Vocab **vocab, char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    int vs, dim, hidden, seq_len, num_layers, t;
    fread(&vs, sizeof(int), 1, f);
    fread(&dim, sizeof(int), 1, f);
    fread(&hidden, sizeof(int), 1, f);
    fread(&seq_len, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&t, sizeof(int), 1, f);
    
    printf("Загрузка: vs=%d, dim=%d, hidden=%d, seq_len=%d, layers=%d, t=%d\n",
           vs, dim, hidden, seq_len, num_layers, t);
    
    long header_pos = ftell(f);
    
    long weights_size = 0;
    weights_size += vs * dim * sizeof(float);
    for (int l = 0; l < num_layers; l++) {
        weights_size += (dim * dim + dim * dim + dim * dim + dim * dim +
                        dim * hidden + hidden + hidden * dim + dim) * sizeof(float);
    }
    weights_size += (dim * vs + vs) * sizeof(float);
    
    weights_size += vs * dim * sizeof(float) * 2;
    for (int l = 0; l < num_layers; l++) {
        weights_size += (dim * dim + dim * dim + dim * dim + dim * dim) * sizeof(float) * 2;
        weights_size += (dim * hidden + hidden + hidden * dim + dim) * sizeof(float) * 2;
    }
    weights_size += (dim * vs + vs) * sizeof(float) * 2;
    
    fseek(f, header_pos + weights_size, SEEK_SET);
    
    int saved_vocab_size;
    fread(&saved_vocab_size, sizeof(int), 1, f);
    printf("Загрузка словаря: %d слов\n", saved_vocab_size);
    
    *vocab = vocab_create(saved_vocab_size + 100);
    Vocab *v = *vocab;
    
    for (int i = 0; i < saved_vocab_size; i++) {
        int len;
        fread(&len, sizeof(int), 1, f);
        if (len > 0 && len < 1000) {
            char *word = malloc(len);
            fread(word, sizeof(char), len, f);
            if (i == 0) {
                free(v->words[0]);
                v->words[0] = strdup(word);
            } else {
                vocab_add(v, word);
            }
            free(word);
        }
    }
    
    Model *m = model_create(v, dim, hidden, seq_len, num_layers);
    m->vs = vs;
    m->t = t;
    
    fseek(f, header_pos, SEEK_SET);
    
    fread(m->embedding, sizeof(float), vs * dim, f);
    for (int l = 0; l < num_layers; l++) {
        fread(m->wq[l], sizeof(float), dim * dim, f);
        fread(m->wk[l], sizeof(float), dim * dim, f);
        fread(m->wv[l], sizeof(float), dim * dim, f);
        fread(m->wo[l], sizeof(float), dim * dim, f);
        fread(m->ffn_w1[l], sizeof(float), dim * hidden, f);
        fread(m->ffn_b1[l], sizeof(float), hidden, f);
        fread(m->ffn_w2[l], sizeof(float), hidden * dim, f);
        fread(m->ffn_b2[l], sizeof(float), dim, f);
    }
    fread(m->out_w, sizeof(float), dim * vs, f);
    fread(m->out_b, sizeof(float), vs, f);
    
    fread(m->m_emb, sizeof(float), vs * dim, f);
    fread(m->v_emb, sizeof(float), vs * dim, f);
    for (int l = 0; l < num_layers; l++) {
        fread(m->m_wq[l], sizeof(float), dim * dim, f);
        fread(m->v_wq[l], sizeof(float), dim * dim, f);
        fread(m->m_wk[l], sizeof(float), dim * dim, f);
        fread(m->v_wk[l], sizeof(float), dim * dim, f);
        fread(m->m_wv[l], sizeof(float), dim * dim, f);
        fread(m->v_wv[l], sizeof(float), dim * dim, f);
        fread(m->m_wo[l], sizeof(float), dim * dim, f);
        fread(m->v_wo[l], sizeof(float), dim * dim, f);
        fread(m->m_ffn1[l], sizeof(float), dim * hidden, f);
        fread(m->v_ffn1[l], sizeof(float), dim * hidden, f);
        fread(m->m_ffn1_b[l], sizeof(float), hidden, f);
        fread(m->v_ffn1_b[l], sizeof(float), hidden, f);
        fread(m->m_ffn2[l], sizeof(float), hidden * dim, f);
        fread(m->v_ffn2[l], sizeof(float), hidden * dim, f);
        fread(m->m_ffn2_b[l], sizeof(float), dim, f);
        fread(m->v_ffn2_b[l], sizeof(float), dim, f);
    }
    fread(m->m_out, sizeof(float), dim * vs, f);
    fread(m->v_out, sizeof(float), dim * vs, f);
    fread(m->m_out_b, sizeof(float), vs, f);
    fread(m->v_out_b, sizeof(float), vs, f);
    
    fclose(f);
    printf("✅ Модель загружена: %d слов\n", v->size);
    return m;
}

// ============ ГЕНЕРАЦИЯ ============
void generate(Model *m, Vocab *vocab, char *prompt, int max_tokens,
              float temperature, int top_k, float top_p, float repeat_penalty) {
    
    if (m->vs < 2) {
        printf("❌ ОШИБКА: Модель повреждена (размер словаря = %d)\n", m->vs);
        return;
    }
    
    int max_seq = (m->seq_len > 32) ? m->seq_len : 32;
    int *tokens = malloc((max_seq + 100) * sizeof(int));
    int len = 0;
    char prompt_copy[256];
    strcpy(prompt_copy, prompt);
    
    char *p = strtok(prompt_copy, " ");
    while (p && len < m->seq_len) {
        int id = vocab_add(vocab, p);
        tokens[len++] = id;
        p = strtok(NULL, " ");
    }
    
    int last_tokens[30] = {0};
    int last_count = 0;
    
    printf("\n🎨 Генерация (temp=%.2f, top_k=%d, top_p=%.2f, penalty=%.2f):\n%s",
           temperature, top_k, top_p, repeat_penalty, prompt);
    fflush(stdout);
    
    for (int step = 0; step < max_tokens && len < max_seq + 100; step++) {
        int seq_len = len < m->seq_len ? len : m->seq_len;
        if (seq_len < 1) seq_len = 1;
        
        float *logits = malloc(m->vs * sizeof(float));
        forward_full(m, tokens + len - seq_len, seq_len, logits);
        
        for (int i = 0; i < m->vs; i++) {
            logits[i] = expf(logf(fmaxf(logits[i], 1e-8f)) / temperature);
        }
        
        for (int j = 0; j < last_count; j++) {
            int tid = last_tokens[j];
            if (tid >= 0 && tid < m->vs) {
                logits[tid] *= repeat_penalty;
            }
        }
        
        softmax(logits, m->vs);
        
        int *indices = malloc(m->vs * sizeof(int));
        for (int i = 0; i < m->vs; i++) indices[i] = i;
        for (int i = 0; i < m->vs; i++) {
            for (int j = i + 1; j < m->vs; j++) {
                if (logits[indices[j]] > logits[indices[i]]) {
                    int tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }
        
        int effective_k = (top_k < m->vs) ? top_k : m->vs;
        float cum = 0.0f;
        int nucleus_size = 0;
        for (int i = 0; i < effective_k && cum < top_p; i++) {
            cum += logits[indices[i]];
            nucleus_size++;
        }
        if (nucleus_size == 0) nucleus_size = 1;
        
        float r = (float)rand() / RAND_MAX_F;
        cum = 0.0f;
        int next = indices[0];
        for (int i = 0; i < nucleus_size; i++) {
            cum += logits[indices[i]];
            if (r < cum) { next = indices[i]; break; }
        }
        
        tokens[len++] = next;
        
        if (last_count < 30) {
            last_tokens[last_count++] = next;
        } else {
            for (int j = 0; j < 29; j++) last_tokens[j] = last_tokens[j+1];
            last_tokens[29] = next;
        }
        
        printf("%s ", vocab_get(vocab, next));
        fflush(stdout);
        
        free(logits);
        free(indices);
    }
    printf("\n");
    free(tokens);
}

// ============ ЗАГРУЗКА ДАТАСЕТА ============
int* load_dataset(Vocab *vocab, char *filename, int *out_len) {
    FILE *f = fopen(filename, "r");
    if (!f) { *out_len = 0; return NULL; }
    
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);
    
    int *tokens = malloc(1000000 * sizeof(int));
    int len = 0;
    char *p = strtok(text, " \t\n\r,.!?;:()[]{}\"");
    while (p && len < 1000000) {
        if (strlen(p) > 0) {
            int id = vocab_add(vocab, p);
            tokens[len++] = id;
        }
        p = strtok(NULL, " \t\n\r,.!?;:()[]{}\"");
    }
    *out_len = len;
    printf("Загружено %d токенов, словарь: %d\n", len, vocab_size(vocab));
    free(text);
    return tokens;
}

// ============ ВЫВОД ПОМОЩИ ============
void print_usage(char *prog) {
    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              NEURAL LANGUAGE MODEL v%s                              ║\n", VERSION);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Usage:                                                                ║\n");
    printf("║   %s train <file.txt> <model.bin> [options]                         ║\n", prog);
    printf("║   %s generate <model.bin> \"<prompt>\" [options]                     ║\n", prog);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Train options:                                                       ║\n");
    printf("║   --vocab_size <int>   Размер словаря (default: 5000)                ║\n");
    printf("║   --embed_dim <int>    Размер эмбеддингов (default: 64)              ║\n");
    printf("║   --hidden_dim <int>   Размер FFN слоя (default: 128)                ║\n");
    printf("║   --seq_len <int>      Длина контекста (default: 8)                  ║\n");
    printf("║   --num_layers <int>   Количество слоёв (default: 1)                 ║\n");
    printf("║   --lr <float>         Learning rate (default: 0.01)                 ║\n");
    printf("║   --epochs <int>       Эпох обучения (default: 100)                  ║\n");
    printf("║   --batch_size <int>   Размер батча (default: 16)                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Generate options:                                                    ║\n");
    printf("║   --temp <float>       Temperature (0.5-1.5, default: 0.85)          ║\n");
    printf("║   --top_k <int>        Top-k sampling (1-100, default: 25)           ║\n");
    printf("║   --top_p <float>      Top-p nucleus (0.5-1.0, default: 0.92)        ║\n");
    printf("║   --repeat_penalty     Penalty за повторы (0.8-1.0, default: 0.92)   ║\n");
    printf("║   --max_tokens <int>   Max токенов для генерации (default: 30)       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
}

// ============ MAIN ============
int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    int vocab_size = 5000;
    int embed_dim = 64;
    int hidden_dim = 128;
    int seq_len = 8;
    int num_layers = 1;
    float lr = 0.01f;
    int epochs = 100;
    int batch_size = 16;
    
    float temperature = 0.85f;
    int top_k = 25;
    float top_p = 0.92f;
    float repeat_penalty = 0.92f;
    int max_tokens = 30;
    
    if (strcmp(argv[1], "train") == 0 && argc >= 3) {
        char *data_file = argv[2];
        char *model_file = (argc >= 4) ? argv[3] : "model.bin";
        
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--vocab_size") == 0 && i+1 < argc)
                vocab_size = atoi(argv[++i]);
            else if (strcmp(argv[i], "--embed_dim") == 0 && i+1 < argc)
                embed_dim = atoi(argv[++i]);
            else if (strcmp(argv[i], "--hidden_dim") == 0 && i+1 < argc)
                hidden_dim = atoi(argv[++i]);
            else if (strcmp(argv[i], "--seq_len") == 0 && i+1 < argc)
                seq_len = atoi(argv[++i]);
            else if (strcmp(argv[i], "--num_layers") == 0 && i+1 < argc)
                num_layers = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc)
                lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc)
                epochs = atoi(argv[++i]);
            else if (strcmp(argv[i], "--batch_size") == 0 && i+1 < argc)
                batch_size = atoi(argv[++i]);
        }
        
        Vocab *vocab = vocab_create(vocab_size);
        int data_len;
        int *tokens = load_dataset(vocab, data_file, &data_len);
        
        if (data_len < seq_len + 1) {
            printf("❌ Ошибка: датасет слишком мал (нужно хотя бы %d токенов)\n", seq_len+1);
            return 1;
        }
        
        Model *m = model_create(vocab, embed_dim, hidden_dim, seq_len, num_layers);
        print_model_size(m);
        
        printf("\n🚀 НАЧАЛО ОБУЧЕНИЯ\n");
        printf("  LR: %.4f, Эпох: %d, Batch: %d\n", lr, epochs, batch_size);
        train(m, vocab, tokens, data_len, epochs, model_file, lr, batch_size);
        
        generate(m, vocab, "привет", 15, 0.85f, 25, 0.92f, 0.92f);
        
        free(tokens);
        vocab_free(vocab);
    }
    else if (strcmp(argv[1], "generate") == 0 && argc >= 4) {
        char *model_file = argv[2];
        char *prompt = argv[3];
        
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--temp") == 0 && i+1 < argc)
                temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "--top_k") == 0 && i+1 < argc)
                top_k = atoi(argv[++i]);
            else if (strcmp(argv[i], "--top_p") == 0 && i+1 < argc)
                top_p = atof(argv[++i]);
            else if (strcmp(argv[i], "--repeat_penalty") == 0 && i+1 < argc)
                repeat_penalty = atof(argv[++i]);
            else if (strcmp(argv[i], "--max_tokens") == 0 && i+1 < argc)
                max_tokens = atoi(argv[++i]);
        }
        
        printf("Загрузка модели из %s...\n", model_file);
        
        Vocab *vocab = NULL;
        Model *m = load_model(&vocab, model_file);
        if (!m) {
            printf("❌ Ошибка загрузки модели\n");
            return 1;
        }
        
        print_model_size(m);
        generate(m, vocab, prompt, max_tokens, temperature, top_k, top_p, repeat_penalty);
        
        vocab_free(vocab);
    }
    else {
        print_usage(argv[0]);
    }
    
    return 0;
}