#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>

#define EPS 1e-8
#define RAND_MAX_F ((float)RAND_MAX)

float frand() { return ((float)rand() / RAND_MAX_F) * 0.02f - 0.01f; }

// ============ NEON-ускоренные функции ============
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
    for (int i = 0; i < m * p; i++) C[i] = 0.0f;
    
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float aik = A[i * n + k];
            if (aik == 0.0f) continue;
            int c_base = i * p;
            int b_base = k * p;
            
            if (((uintptr_t)(B + b_base) & 15) == 0 && ((uintptr_t)(C + c_base) & 15) == 0) {
                int j = 0;
                float32x4_t a_vec = vdupq_n_f32(aik);
                for (; j <= p - 4; j += 4) {
                    float32x4_t b_vec = vld1q_f32(B + b_base + j);
                    float32x4_t c_vec = vld1q_f32(C + c_base + j);
                    c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                    vst1q_f32(C + c_base + j, c_vec);
                }
                for (; j < p; j++) {
                    C[c_base + j] += aik * B[b_base + j];
                }
            } else {
                for (int j = 0; j < p; j++) {
                    C[c_base + j] += aik * B[b_base + j];
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

void relu_neon(float *x, int n) {
    for (int i = 0; i <= n - 4; i += 4)
        vst1q_f32(x + i, vmaxq_f32(vld1q_f32(x + i), vdupq_n_f32(0.0f)));
    for (int i = (n/4)*4; i < n; i++)
        if (x[i] < 0) x[i] = 0;
}

void add_neon(float *a, float *b, int n) {
    for (int i = 0; i <= n - 4; i += 4)
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (int i = (n/4)*4; i < n; i++)
        a[i] += b[i];
}

// ============ Словарь ============
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

// ============ Структура модели ============
typedef struct {
    float *embedding;
    float **wq, **wk, **wv, **wo;
    float **ffn_w1, **ffn_b1;
    float **ffn_w2, **ffn_b2;
    float *out_w, *out_b;
    float *m_emb, *v_emb;
    float **m_wq, **v_wq, **m_wk, **v_wk, **m_wv, **v_wv, **m_wo, **v_wo;
    float **m_ffn1, **v_ffn1, **m_ffn2, **v_ffn2;
    float *m_out, *v_out, *m_out_b, *v_out_b;
    int t;
    int vs, dim, hidden, seq_len, num_layers;
} Model;

// ============ Прототипы ============
void save_model(Model *m, Vocab *vocab, char *filename);
Model* load_model(Vocab **vocab, char *filename);
void adam_update(float *param, float *m, float *v, float *grad, int n, float lr, int t);
void print_model_size(Model *m);
void forward_full(Model *m, int *tokens, int seq_len, float *logits);
void backward_full(Model *m, int *tokens, int seq_len, int target, float *logits,
                   float *d_embedding, float *d_out_w, float *d_out_b);

// ============ Adam update ============
void adam_update(float *param, float *m, float *v, float *grad, int n, float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f;
    float lr_t = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
    for (int i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / (1.0f - powf(beta1, t));
        float v_hat = v[i] / (1.0f - powf(beta2, t));
        param[i] -= lr_t * m_hat / (sqrtf(v_hat) + EPS);
    }
}

// ============ Вывод размера модели ============
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
    printf("║              МОДЕЛЬ NEURAL LANGUAGE v4.7          ║\n");
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

// ============ Создание модели ============
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
    m->m_ffn2 = malloc(num_layers * sizeof(float*));
    m->v_ffn2 = malloc(num_layers * sizeof(float*));
    
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
        m->m_ffn2[l] = calloc(ffn2_size, sizeof(float));
        m->v_ffn2[l] = calloc(ffn2_size, sizeof(float));
        
        for (int i = 0; i < attn_size; i++) {
            m->wq[l][i] = frand() * 0.01f;
            m->wk[l][i] = frand() * 0.01f;
            m->wv[l][i] = frand() * 0.01f;
            m->wo[l][i] = frand() * 0.01f;
        }
        for (int i = 0; i < ffn1_size; i++) m->ffn_w1[l][i] = frand() * 0.01f;
        for (int i = 0; i < ffn2_size; i++) m->ffn_w2[l][i] = frand() * 0.01f;
    }
    
    for (int i = 0; i < emb_size; i++) m->embedding[i] = frand() * 0.01f;
    for (int i = 0; i < out_size; i++) m->out_w[i] = frand() * 0.01f;
    
    return m;
}

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

// ============ Backward (упрощённо) ============
void backward_full(Model *m, int *tokens, int seq_len, int target, float *logits,
                   float *d_embedding, float *d_out_w, float *d_out_b) {
    
    float *d_logits = malloc(m->vs * sizeof(float));
    for (int i = 0; i < m->vs; i++) d_logits[i] = logits[i];
    d_logits[target] -= 1.0f;
    
    for (int i = 0; i < m->vs; i++) {
        d_out_b[i] += d_logits[i];
        for (int d = 0; d < m->dim; d++)
            d_out_w[d * m->vs + i] += d_logits[i];
    }
    
    float **d_final = malloc(seq_len * sizeof(float*));
    for (int i = 0; i < seq_len; i++) {
        d_final[i] = calloc(m->dim, sizeof(float));
    }
    
    for (int d = 0; d < m->dim; d++) {
        float grad = 0;
        for (int i = 0; i < m->vs; i++)
            grad += d_logits[i] * m->out_w[d * m->vs + i];
        d_final[seq_len-1][d] = grad;
    }
    
    for (int i = 0; i < seq_len; i++) {
        int token = tokens[i] % m->vs;
        for (int d = 0; d < m->dim; d++) {
            d_embedding[token * m->dim + d] += d_final[i][d];
        }
    }
    
    for (int i = 0; i < seq_len; i++) free(d_final[i]);
    free(d_final);
    free(d_logits);
}

// ============ Датасет ============
typedef struct { int *tokens; int len; } Dataset;

Dataset* load_dataset(Vocab *vocab, char *filename) {
    Dataset *ds = malloc(sizeof(Dataset));
    FILE *f = fopen(filename, "r");
    if (!f) { ds->len = 0; return ds; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);
    
    int *tokens = malloc(100000 * sizeof(int));
    int len = 0;
    char *p = strtok(text, " \t\n\r,.!?;:()[]{}\"");
    while (p && len < 100000) {
        if (strlen(p) > 0) {
            int id = vocab_add(vocab, p);
            tokens[len++] = id;
        }
        p = strtok(NULL, " \t\n\r,.!?;:()[]{}\"");
    }
    ds->tokens = tokens;
    ds->len = len;
    printf("Загружено %d токенов, словарь: %d\n", len, vocab_size(vocab));
    free(text);
    return ds;
}

// ============ Сохранение модели ============
void save_model(Model *m, Vocab *vocab, char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    fwrite(&m->vs, sizeof(int), 1, f);
    fwrite(&m->dim, sizeof(int), 1, f);
    fwrite(&m->hidden, sizeof(int), 1, f);
    fwrite(&m->seq_len, sizeof(int), 1, f);
    fwrite(&m->num_layers, sizeof(int), 1, f);
    
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
    
    fwrite(&vocab->size, sizeof(int), 1, f);
    for (int i = 0; i < vocab->size; i++) {
        int len = strlen(vocab->words[i]) + 1;
        fwrite(&len, sizeof(int), 1, f);
        fwrite(vocab->words[i], sizeof(char), len, f);
    }
    
    fclose(f);
    printf("Модель сохранена: %d слов\n", vocab->size);
}

Model* load_model(Vocab **vocab, char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    int vs, dim, hidden, seq_len, num_layers;
    fread(&vs, sizeof(int), 1, f);
    fread(&dim, sizeof(int), 1, f);
    fread(&hidden, sizeof(int), 1, f);
    fread(&seq_len, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    
    printf("Загрузка: vs=%d, dim=%d, hidden=%d, seq_len=%d, layers=%d\n", 
           vs, dim, hidden, seq_len, num_layers);
    
    long header_pos = ftell(f);
    
    long weights_size = vs * dim * sizeof(float);
    for (int l = 0; l < num_layers; l++) {
        weights_size += (dim * dim + dim * dim + dim * dim + dim * dim + 
                         dim * hidden + hidden + hidden * dim + dim) * sizeof(float);
    }
    weights_size += (dim * vs + vs) * sizeof(float);
    
    fseek(f, weights_size, SEEK_CUR);
    
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
    
    fclose(f);
    printf("Загружено %d слов\n", v->size);
    return m;
}

// ============ Тренировка ============
void train(Model *m, Vocab *vocab, Dataset *ds, int epochs, char *model_file, float lr) {
    int total_windows = ds->len - m->seq_len;
    if (total_windows <= 0) return;
    printf("Окон: %d\n", total_windows);
    
    int emb_size = m->vs * m->dim;
    int out_size = m->dim * m->vs;
    
    float best_loss = 1e9f;
    int no_improve = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        float *d_emb = calloc(emb_size, sizeof(float));
        float *d_out = calloc(out_size, sizeof(float));
        float *d_out_b = calloc(m->vs, sizeof(float));
        
        for (int w = 0; w < total_windows; w++) {
            float *logits = malloc(m->vs * sizeof(float));
            forward_full(m, ds->tokens + w, m->seq_len, logits);
            
            int target = ds->tokens[w + m->seq_len];
            float loss = -logf(logits[target] + EPS);
            total_loss += loss;
            
            backward_full(m, ds->tokens + w, m->seq_len, target, logits,
                         d_emb, d_out, d_out_b);
            free(logits);
        }
        
        float inv_windows = 1.0f / total_windows;
        float avg_loss = total_loss / total_windows;
        
        m->t++;
        adam_update(m->embedding, m->m_emb, m->v_emb, d_emb, emb_size, lr, m->t);
        adam_update(m->out_w, m->m_out, m->v_out, d_out, out_size, lr, m->t);
        adam_update(m->out_b, m->m_out_b, m->v_out_b, d_out_b, m->vs, lr, m->t);
        
        printf("Epoch %d, Loss: %.4f\n", epoch, avg_loss);
        
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            no_improve = 0;
            save_model(m, vocab, model_file);
        } else {
            no_improve++;
            if (no_improve >= 10) {
                printf("Ранняя остановка\n");
                free(d_emb); free(d_out); free(d_out_b);
                break;
            }
        }
        
        free(d_emb); free(d_out); free(d_out_b);
    }
}

// ============ Генерация ============
void generate(Model *m, Vocab *vocab, char *prompt, int max_tokens, 
              float temperature, int top_k, float top_p, float repeat_penalty) {
    
    if (m->vs < 2) {
        printf("ОШИБКА: Модель повреждена (размер словаря = %d)\n", m->vs);
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

// ============ Вывод помощи ============
void print_usage(char *prog) {
    printf("\n╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    NEURAL LANGUAGE MODEL v4.7                         ║\n");
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
    printf("║   --lr <float>         Learning rate (default: 0.2)                  ║\n");
    printf("║   --epochs <int>       Эпох обучения (default: 500)                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Generate options:                                                    ║\n");
    printf("║   --temp <float>       Temperature (0.5-1.5, default: 0.85)          ║\n");
    printf("║   --top_k <int>        Top-k sampling (1-100, default: 25)           ║\n");
    printf("║   --top_p <float>      Top-p nucleus (0.5-1.0, default: 0.92)        ║\n");
    printf("║   --repeat_penalty     Penalty за повторы (0.8-1.0, default: 0.92)   ║\n");
    printf("║   --max_tokens <int>   Max токенов для генерации (default: 30)       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
}

// ============ Main ============
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
    float lr = 0.2f;
    int epochs = 500;
    
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
        }
        
        Vocab *vocab = vocab_create(vocab_size);
        Dataset *ds = load_dataset(vocab, data_file);
        if (ds->len < seq_len + 1) {
            printf("Ошибка: датасет слишком мал (нужно хотя бы %d токенов)\n", seq_len+1);
            return 1;
        }
        
        Model *m = model_create(vocab, embed_dim, hidden_dim, seq_len, num_layers);
        print_model_size(m);
        
        printf("\n🚀 НАЧАЛО ОБУЧЕНИЯ\n");
        printf("  LR: %.4f, Эпох: %d\n", lr, epochs);
        train(m, vocab, ds, epochs, model_file, lr);
        
        generate(m, vocab, "привет", 15, 0.85f, 25, 0.92f, 0.92f);
        
        free(ds->tokens); free(ds);
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
            printf("Ошибка загрузки модели\n");
            return 1;
        }
        
        print_model_size(m);
        generate(m, vocab, prompt, max_tokens, temperature, top_k, top_p, repeat_penalty);
    }
    else {
        print_usage(argv[0]);
    }
    
    return 0;
}