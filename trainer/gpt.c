#define _GNU_SOURCE
#include "gpt.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

const char *const PARAM_NAMES[P_COUNT] = {
    "tok_emb", "pos_emb", "ln1.weight", "ln1.bias", "attn.qkv.weight", "attn.qkv.bias",
    "attn.proj.weight", "attn.proj.bias", "ln2.weight", "ln2.bias", "ff.up.weight",
    "ff.up.bias", "ff.down.weight", "ff.down.bias", "ln_f.weight", "ln_f.bias",
    "head.weight", "head.bias",
};
const int PARAM_DECAY[P_COUNT] = {1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0};

/* ---------- rng: xorshift64* ---------- */

uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float rng_uniform(uint64_t *s) { return (float)(rng_next(s) >> 40) / 16777216.0f; }

/* ---------- two-thread pool, fixed split ----------
 * The caller runs the first half and the worker the second. Every output element is
 * computed by the same instruction sequence whatever the split, so results do not
 * depend on the thread count or on timing. */

typedef void (*range_fn)(void *ctx, int lo, int hi);

static struct {
    int n;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    unsigned gen, done_gen;
    range_fn fn;
    void *ctx;
    int lo, hi;
} pool = {1, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
          0, 0, 0, 0, 0, 0};

static void *pool_worker(void *unused) {
    (void)unused;
    unsigned seen = 0;
    pthread_mutex_lock(&pool.mu);
    for (;;) {
        while (pool.gen == seen) pthread_cond_wait(&pool.go, &pool.mu);
        seen = pool.gen;
        range_fn fn = pool.fn;
        void *ctx = pool.ctx;
        int lo = pool.lo, hi = pool.hi;
        pthread_mutex_unlock(&pool.mu);
        fn(ctx, lo, hi);
        pthread_mutex_lock(&pool.mu);
        pool.done_gen = seen;
        pthread_cond_signal(&pool.done);
    }
    return NULL;
}

void par_init(int nthreads) {
    pool.n = nthreads >= 2 ? 2 : 1;
    if (pool.n == 2) pthread_create(&pool.th, NULL, pool_worker, NULL);
}

static void par_for(int n, int align, range_fn fn, void *ctx) {
    if (pool.n == 1 || n < 2 * align) { fn(ctx, 0, n); return; }
    int mid = (n / 2 + align - 1) / align * align;
    pthread_mutex_lock(&pool.mu);
    pool.fn = fn; pool.ctx = ctx; pool.lo = mid; pool.hi = n;
    unsigned gen = ++pool.gen;
    pthread_cond_signal(&pool.go);
    pthread_mutex_unlock(&pool.mu);
    fn(ctx, 0, mid);
    pthread_mutex_lock(&pool.mu);
    while (pool.done_gen != gen) pthread_cond_wait(&pool.done, &pool.mu);
    pthread_mutex_unlock(&pool.mu);
}

/* ---------- SGEMM: C[M,N] (+)= A[M,K] . B[N,K]^T (+ bias[N]) ----------
 * Both operands are read along K. Each element uses one 4-lane accumulator over the
 * multiple-of-4 part of K, a fixed horizontal sum, then a scalar tail, in every code
 * path, which is what makes blocking and the thread split invisible in the result. */

typedef struct {
    int M, N, K, lda, ldb, ldc, acc;
    const float *A, *B, *bias;
    float *C;
} Gemm;

static inline void store(const Gemm *g, int i, int j, float v) {
    if (g->bias) v += g->bias[j];
    float *c = g->C + (size_t)i * g->ldc + j;
    *c = g->acc ? *c + v : v;
}

#ifdef HAVE_NEON
static inline float hsum(float32x4_t a) {
    float32x2_t s = vadd_f32(vget_low_f32(a), vget_high_f32(a));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}
#endif

static inline float tail(const float *a, const float *b, int k0, int K, float s) {
    for (int k = k0; k < K; k++) s += a[k] * b[k];
    return s;
}

static float dot1(const float *a, const float *b, int K) {
    int K4 = K & ~3;
#ifdef HAVE_NEON
    float32x4_t s = vdupq_n_f32(0.0f);
    for (int k = 0; k < K4; k += 4) s = vfmaq_f32(s, vld1q_f32(a + k), vld1q_f32(b + k));
    return tail(a, b, K4, K, hsum(s));
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < K4; k += 4) {
        s0 += a[k] * b[k]; s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2]; s3 += a[k + 3] * b[k + 3];
    }
    return tail(a, b, K4, K, (s0 + s2) + (s1 + s3));
#endif
}

#ifdef HAVE_NEON
static void kernel4x2(const Gemm *g, int i, int j) {
    int K = g->K, K4 = K & ~3;
    const float *a0 = g->A + (size_t)i * g->lda, *a1 = a0 + g->lda, *a2 = a1 + g->lda, *a3 = a2 + g->lda;
    const float *b0 = g->B + (size_t)j * g->ldb, *b1 = b0 + g->ldb;
    float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c10 = c00, c11 = c00;
    float32x4_t c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    for (int k = 0; k < K4; k += 4) {
        float32x4_t x0 = vld1q_f32(b0 + k), x1 = vld1q_f32(b1 + k);
        float32x4_t y = vld1q_f32(a0 + k);
        c00 = vfmaq_f32(c00, y, x0); c01 = vfmaq_f32(c01, y, x1);
        y = vld1q_f32(a1 + k);
        c10 = vfmaq_f32(c10, y, x0); c11 = vfmaq_f32(c11, y, x1);
        y = vld1q_f32(a2 + k);
        c20 = vfmaq_f32(c20, y, x0); c21 = vfmaq_f32(c21, y, x1);
        y = vld1q_f32(a3 + k);
        c30 = vfmaq_f32(c30, y, x0); c31 = vfmaq_f32(c31, y, x1);
    }
    store(g, i, j, tail(a0, b0, K4, K, hsum(c00))); store(g, i, j + 1, tail(a0, b1, K4, K, hsum(c01)));
    store(g, i + 1, j, tail(a1, b0, K4, K, hsum(c10))); store(g, i + 1, j + 1, tail(a1, b1, K4, K, hsum(c11)));
    store(g, i + 2, j, tail(a2, b0, K4, K, hsum(c20))); store(g, i + 2, j + 1, tail(a2, b1, K4, K, hsum(c21)));
    store(g, i + 3, j, tail(a3, b0, K4, K, hsum(c30))); store(g, i + 3, j + 1, tail(a3, b1, K4, K, hsum(c31)));
}

static void kernel1x4(const Gemm *g, int i, int j) {
    int K = g->K, K4 = K & ~3;
    const float *a = g->A + (size_t)i * g->lda;
    const float *b0 = g->B + (size_t)j * g->ldb, *b1 = b0 + g->ldb, *b2 = b1 + g->ldb, *b3 = b2 + g->ldb;
    float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0;
    for (int k = 0; k < K4; k += 4) {
        float32x4_t y = vld1q_f32(a + k);
        c0 = vfmaq_f32(c0, y, vld1q_f32(b0 + k)); c1 = vfmaq_f32(c1, y, vld1q_f32(b1 + k));
        c2 = vfmaq_f32(c2, y, vld1q_f32(b2 + k)); c3 = vfmaq_f32(c3, y, vld1q_f32(b3 + k));
    }
    store(g, i, j, tail(a, b0, K4, K, hsum(c0))); store(g, i, j + 1, tail(a, b1, K4, K, hsum(c1)));
    store(g, i, j + 2, tail(a, b2, K4, K, hsum(c2))); store(g, i, j + 3, tail(a, b3, K4, K, hsum(c3)));
}
#endif

static void gemm_block(const Gemm *g, int m0, int m1, int n0, int n1) {
    int i = m0;
#ifdef HAVE_NEON
    for (; i + 4 <= m1; i += 4) {
        int j = n0;
        for (; j + 2 <= n1; j += 2) kernel4x2(g, i, j);
        for (; j < n1; j++)
            for (int r = 0; r < 4; r++)
                store(g, i + r, j, dot1(g->A + (size_t)(i + r) * g->lda, g->B + (size_t)j * g->ldb, g->K));
    }
#endif
    for (; i < m1; i++) {
        int j = n0;
#ifdef HAVE_NEON
        for (; j + 4 <= n1; j += 4) kernel1x4(g, i, j);
#endif
        for (; j < n1; j++)
            store(g, i, j, dot1(g->A + (size_t)i * g->lda, g->B + (size_t)j * g->ldb, g->K));
    }
}

/* B is walked in chunks of rows that fit a core's share of the 512 KB L2. */
static void gemm_rows(void *ctx, int m0, int m1) {
    const Gemm *g = ctx;
    int nc = (int)((128 * 1024) / ((size_t)g->K * sizeof(float))) & ~3;
    if (nc < 4) nc = 4;
    for (int n0 = 0; n0 < g->N; n0 += nc)
        gemm_block(g, m0, m1, n0, n0 + nc < g->N ? n0 + nc : g->N);
}

static void gemm_cols(void *ctx, int n0, int n1) {
    const Gemm *g = ctx;
    gemm_block(g, 0, g->M, n0, n1);
}

void sgemm_nt(int M, int N, int K, const float *A, int lda, const float *B, int ldb,
              float *C, int ldc, int acc, const float *bias) {
    Gemm g = {M, N, K, lda, ldb, ldc, acc, A, B, bias, C};
    if (M >= 8) par_for(M, 4, gemm_rows, &g);
    else par_for(N, 4, gemm_cols, &g);
}

static void transpose(float *dst, const float *src, int rows, int cols) {
    for (int r0 = 0; r0 < rows; r0 += 32)
        for (int c0 = 0; c0 < cols; c0 += 32)
            for (int r = r0; r < r0 + 32 && r < rows; r++)
                for (int c = c0; c < c0 + 32 && c < cols; c++)
                    dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
}

/* ---------- layers (llm.c train_gpt2.c, fp32) ---------- */

static void encoder_forward(float *out, const int *inp, const float *wte, const float *wpe, int BT, int T, int C) {
    for (int bt = 0; bt < BT; bt++) {
        const float *e = wte + (size_t)inp[bt] * C, *p = wpe + (size_t)(bt % T) * C;
        for (int i = 0; i < C; i++) out[(size_t)bt * C + i] = e[i] + p[i];
    }
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *inp, int BT, int T, int C) {
    for (int bt = 0; bt < BT; bt++) {
        float *de = dwte + (size_t)inp[bt] * C, *dp = dwpe + (size_t)(bt % T) * C;
        for (int i = 0; i < C; i++) { de[i] += dout[(size_t)bt * C + i]; dp[i] += dout[(size_t)bt * C + i]; }
    }
}

static void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                              const float *w, const float *b, int BT, int C) {
    for (int bt = 0; bt < BT; bt++) {
        const float *x = inp + (size_t)bt * C;
        float m = 0.0f, v = 0.0f;
        for (int i = 0; i < C; i++) m += x[i];
        m /= C;
        for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
        v /= C;
        float s = 1.0f / sqrtf(v + 1e-5f);
        float *o = out + (size_t)bt * C;
        for (int i = 0; i < C; i++) o[i] = (x[i] - m) * s * w[i] + b[i];
        mean[bt] = m; rstd[bt] = s;
    }
}

static void layernorm_backward(float *dinp, float *dw, float *db, const float *dout, const float *inp,
                               const float *w, const float *mean, const float *rstd, int BT, int C) {
    for (int bt = 0; bt < BT; bt++) {
        const float *dy = dout + (size_t)bt * C, *x = inp + (size_t)bt * C;
        float *dx = dinp + (size_t)bt * C;
        float m = mean[bt], s = rstd[bt], dmean = 0.0f, dnorm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float n = (x[i] - m) * s, dn = w[i] * dy[i];
            dmean += dn; dnorm_mean += dn * n;
        }
        dmean /= C; dnorm_mean /= C;
        for (int i = 0; i < C; i++) {
            float n = (x[i] - m) * s, dn = w[i] * dy[i];
            db[i] += dy[i];
            dw[i] += n * dy[i];
            dx[i] += (dn - dmean - n * dnorm_mean) * s;
        }
    }
}

static void matmul_forward(float *out, const float *inp, const float *w, const float *b, int BT, int C, int OC) {
    sgemm_nt(BT, OC, C, inp, C, w, C, out, OC, 0, b);
}

static void matmul_backward(GPT *g, float *dinp, float *dw, float *db, const float *dout,
                            const float *inp, const float *w, int BT, int C, int OC) {
    float *wT = g->scratch_w, *doutT = g->scratch_a, *inpT = g->scratch_b;
    transpose(wT, w, OC, C);
    sgemm_nt(BT, C, OC, dout, OC, wT, OC, dinp, C, 1, NULL);
    transpose(doutT, dout, BT, OC);
    transpose(inpT, inp, BT, C);
    sgemm_nt(OC, C, BT, doutT, BT, inpT, BT, dw, C, 1, NULL);
    for (int o = 0; o < OC; o++) {
        float s = 0.0f;
        for (int bt = 0; bt < BT; bt++) s += doutT[(size_t)o * BT + bt];
        db[o] += s;
    }
}

typedef struct {
    float *out, *preatt, *att, *dinp, *dpreatt, *datt;
    const float *inp, *dout;
    int T, C, NH;
} AttnJob;

static void attention_forward_range(void *ctx, int lo, int hi) {
    const AttnJob *j = ctx;
    int T = j->T, C = j->C, NH = j->NH, C3 = 3 * C, hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    for (int bh = lo; bh < hi; bh++) {
        int b = bh / NH, h = bh % NH;
        for (int t = 0; t < T; t++) {
            const float *q = j->inp + ((size_t)b * T + t) * C3 + h * hs;
            float *pre = j->preatt + ((size_t)bh * T + t) * T, *att = j->att + ((size_t)bh * T + t) * T;
            float mx = -10000.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                const float *k = j->inp + ((size_t)b * T + t2) * C3 + h * hs + C;
                float v = 0.0f;
                for (int i = 0; i < hs; i++) v += q[i] * k[i];
                v *= scale;
                pre[t2] = v;
                if (v > mx) mx = v;
            }
            float sum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) { float e = expf(pre[t2] - mx); att[t2] = e; sum += e; }
            float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
            for (int t2 = 0; t2 < T; t2++) att[t2] = t2 <= t ? att[t2] * inv : 0.0f;
            float *o = j->out + ((size_t)b * T + t) * C + h * hs;
            for (int i = 0; i < hs; i++) o[i] = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                const float *v = j->inp + ((size_t)b * T + t2) * C3 + h * hs + 2 * C;
                for (int i = 0; i < hs; i++) o[i] += att[t2] * v[i];
            }
        }
    }
}

/* Softmax backward uses the O(T) form dpre = att * (datt - sum(att * datt)). */
static void attention_backward_range(void *ctx, int lo, int hi) {
    const AttnJob *j = ctx;
    int T = j->T, C = j->C, NH = j->NH, C3 = 3 * C, hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    for (int bh = lo; bh < hi; bh++) {
        int b = bh / NH, h = bh % NH;
        for (int t = 0; t < T; t++) {
            const float *att = j->att + ((size_t)bh * T + t) * T;
            float *datt = j->datt + ((size_t)bh * T + t) * T, *dpre = j->dpreatt + ((size_t)bh * T + t) * T;
            const float *dy = j->dout + ((size_t)b * T + t) * C + h * hs;
            const float *q = j->inp + ((size_t)b * T + t) * C3 + h * hs;
            float *dq = j->dinp + ((size_t)b * T + t) * C3 + h * hs;
            for (int t2 = 0; t2 <= t; t2++) {
                const float *v = j->inp + ((size_t)b * T + t2) * C3 + h * hs + 2 * C;
                float *dv = j->dinp + ((size_t)b * T + t2) * C3 + h * hs + 2 * C;
                for (int i = 0; i < hs; i++) { datt[t2] += v[i] * dy[i]; dv[i] += att[t2] * dy[i]; }
            }
            float s = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) s += att[t2] * datt[t2];
            for (int t2 = 0; t2 <= t; t2++) dpre[t2] += att[t2] * (datt[t2] - s);
            for (int t2 = 0; t2 <= t; t2++) {
                const float *k = j->inp + ((size_t)b * T + t2) * C3 + h * hs + C;
                float *dk = j->dinp + ((size_t)b * T + t2) * C3 + h * hs + C;
                float d = dpre[t2] * scale;
                for (int i = 0; i < hs; i++) { dq[i] += k[i] * d; dk[i] += q[i] * d; }
            }
        }
    }
}

#define INV_SQRT2 0.70710678118654752f
#define INV_SQRT_2PI 0.39894228040143268f

static void gelu_forward(float *out, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * INV_SQRT2));
}

static void gelu_backward(float *dinp, const float *x, const float *dout, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float cdf = 0.5f * (1.0f + erff(x[i] * INV_SQRT2));
        float pdf = INV_SQRT_2PI * expf(-0.5f * x[i] * x[i]);
        dinp[i] += (cdf + x[i] * pdf) * dout[i];
    }
}

static void residual_forward(float *out, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void residual_backward(float *da, float *db, const float *dout, size_t n) {
    for (size_t i = 0; i < n; i++) { da[i] += dout[i]; db[i] += dout[i]; }
}

static void softmax_row(float *p, const float *logits, int V) {
    float mx = logits[0], sum = 0.0f;
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < V; i++) { p[i] = expf(logits[i] - mx); sum += p[i]; }
    for (int i = 0; i < V; i++) p[i] /= sum;
}

/* ---------- model ---------- */

static void tensor_sizes(Shape s, size_t *sz) {
    size_t V = s.V, T = s.T, L = s.L, C = s.C, F = s.F;
    size_t v[P_COUNT] = {V * C, T * C, L * C, L * C, L * 3 * C * C, L * 3 * C, L * C * C, L * C,
                         L * C, L * C, L * F * C, L * F, L * C * F, L * C, C, C, V * C, V};
    memcpy(sz, v, sizeof v);
}

int gpt_alloc(GPT *g, Shape s, int B) {
    memset(g, 0, sizeof *g);
    g->s = s; g->B = B;
    tensor_sizes(s, g->size);
    for (int i = 0; i < P_COUNT; i++) g->n += g->size[i];
    g->params = calloc(4 * g->n, sizeof(float));
    if (!g->params) return -1;
    g->grads = g->params + g->n; g->m = g->grads + g->n; g->v = g->m + g->n;
    size_t off = 0;
    for (int i = 0; i < P_COUNT; i++) {
        g->p[i] = g->params + off; g->g[i] = g->grads + off; off += g->size[i];
    }
    size_t V = s.V, T = s.T, L = s.L, NH = s.NH, C = s.C, F = s.F, BT = (size_t)B * T;
    size_t total = 0;
#define X(name, n) total += (n);
    ACT_LIST(X)
#undef X
    g->act_floats = total;
    g->act_mem = malloc(total * sizeof(float));
    g->dact_mem = malloc(total * sizeof(float));
    size_t maxd = 3 * C > F ? 3 * C : F;
    if (V > maxd) maxd = V;
    g->scratch_w = malloc(maxd * C * sizeof(float));
    g->scratch_a = malloc(BT * maxd * sizeof(float));
    g->scratch_b = malloc(BT * maxd * sizeof(float));
    if (!g->act_mem || !g->dact_mem || !g->scratch_w || !g->scratch_a || !g->scratch_b) return -1;
    float *pa = g->act_mem, *pd = g->dact_mem;
#define X(name, n) g->a.name = pa; g->da.name = pd; pa += (n); pd += (n);
    ACT_LIST(X)
#undef X
    return 0;
}

void gpt_free(GPT *g) {
    free(g->params); free(g->act_mem); free(g->dact_mem);
    free(g->scratch_w); free(g->scratch_a); free(g->scratch_b);
}

/* model.py init: Linear/Embedding weights N(0, 0.02), biases 0, LayerNorm 1/0. */
void gpt_random_init(GPT *g, uint64_t seed) {
    uint64_t r = seed ? seed : 1;
    memset(g->params, 0, 4 * g->n * sizeof(float));
    for (int t = 0; t < P_COUNT; t++) {
        if (t == P_LN1_W || t == P_LN2_W || t == P_LNF_W) {
            for (size_t i = 0; i < g->size[t]; i++) g->p[t][i] = 1.0f;
        } else if (PARAM_DECAY[t]) {
            for (size_t i = 0; i < g->size[t]; i += 2) {
                float u1 = rng_uniform(&r), u2 = rng_uniform(&r);
                float rad = sqrtf(-2.0f * logf(u1 + 1e-12f)), th = 6.2831853f * u2;
                g->p[t][i] = 0.02f * rad * cosf(th);
                if (i + 1 < g->size[t]) g->p[t][i + 1] = 0.02f * rad * sinf(th);
            }
        }
    }
}

float gpt_forward(GPT *g, const int *inp, const int *tgt, int B, int T) {
    const Shape s = g->s;
    int L = s.L, C = s.C, F = s.F, V = s.V, NH = s.NH, BT = B * T;
    Acts *a = &g->a;
    encoder_forward(a->encoded, inp, g->p[P_TOK_EMB], g->p[P_POS_EMB], BT, T, C);
    for (int l = 0; l < L; l++) {
        float *res = l == 0 ? a->encoded : a->res3 + (size_t)(l - 1) * BT * C;
        float *ln1 = a->ln1 + (size_t)l * BT * C, *qkv = a->qkv + (size_t)l * BT * 3 * C;
        float *atty = a->atty + (size_t)l * BT * C, *attproj = a->attproj + (size_t)l * BT * C;
        float *res2 = a->res2 + (size_t)l * BT * C, *ln2 = a->ln2 + (size_t)l * BT * C;
        float *fch = a->fch + (size_t)l * BT * F, *fchg = a->fch_gelu + (size_t)l * BT * F;
        float *fcproj = a->fcproj + (size_t)l * BT * C, *res3 = a->res3 + (size_t)l * BT * C;
        layernorm_forward(ln1, a->ln1_mean + (size_t)l * BT, a->ln1_rstd + (size_t)l * BT, res,
                          g->p[P_LN1_W] + l * C, g->p[P_LN1_B] + l * C, BT, C);
        matmul_forward(qkv, ln1, g->p[P_QKV_W] + (size_t)l * 3 * C * C, g->p[P_QKV_B] + l * 3 * C, BT, C, 3 * C);
        AttnJob j = {atty, a->preatt + (size_t)l * B * NH * T * T, a->att + (size_t)l * B * NH * T * T,
                     NULL, NULL, NULL, qkv, NULL, T, C, NH};
        par_for(B * NH, 1, attention_forward_range, &j);
        matmul_forward(attproj, atty, g->p[P_PROJ_W] + (size_t)l * C * C, g->p[P_PROJ_B] + l * C, BT, C, C);
        residual_forward(res2, res, attproj, (size_t)BT * C);
        layernorm_forward(ln2, a->ln2_mean + (size_t)l * BT, a->ln2_rstd + (size_t)l * BT, res2,
                          g->p[P_LN2_W] + l * C, g->p[P_LN2_B] + l * C, BT, C);
        matmul_forward(fch, ln2, g->p[P_UP_W] + (size_t)l * F * C, g->p[P_UP_B] + l * F, BT, C, F);
        gelu_forward(fchg, fch, (size_t)BT * F);
        matmul_forward(fcproj, fchg, g->p[P_DOWN_W] + (size_t)l * C * F, g->p[P_DOWN_B] + l * C, BT, F, C);
        residual_forward(res3, res2, fcproj, (size_t)BT * C);
    }
    float *last = a->res3 + (size_t)(L - 1) * BT * C;
    layernorm_forward(a->lnf, a->lnf_mean, a->lnf_rstd, last, g->p[P_LNF_W], g->p[P_LNF_B], BT, C);
    matmul_forward(a->logits, a->lnf, g->p[P_HEAD_W], g->p[P_HEAD_B], BT, C, V);
    if (!tgt) return -1.0f;
    double sum = 0.0;
    for (int bt = 0; bt < BT; bt++) {
        float *p = a->probs + (size_t)bt * V;
        softmax_row(p, a->logits + (size_t)bt * V, V);
        a->losses[bt] = -logf(p[tgt[bt]]);
        sum += a->losses[bt];
    }
    return (float)(sum / BT);
}

void gpt_zero_grad(GPT *g) { memset(g->grads, 0, g->n * sizeof(float)); }

/* Accumulates into g->grads; loss_scale is 1/accum_steps. */
void gpt_backward(GPT *g, const int *inp, const int *tgt, int B, int T, float loss_scale) {
    const Shape s = g->s;
    int L = s.L, C = s.C, F = s.F, V = s.V, NH = s.NH, BT = B * T;
    Acts *a = &g->a, *d = &g->da;
    memset(g->dact_mem, 0, g->act_floats * sizeof(float));
    float dloss = loss_scale / BT;
    for (int bt = 0; bt < BT; bt++) {
        const float *p = a->probs + (size_t)bt * V;
        float *dl = d->logits + (size_t)bt * V;
        for (int i = 0; i < V; i++) dl[i] = (p[i] - (i == tgt[bt] ? 1.0f : 0.0f)) * dloss;
    }
    matmul_backward(g, d->lnf, g->g[P_HEAD_W], g->g[P_HEAD_B], d->logits, a->lnf, g->p[P_HEAD_W], BT, C, V);
    float *last = a->res3 + (size_t)(L - 1) * BT * C, *dlast = d->res3 + (size_t)(L - 1) * BT * C;
    layernorm_backward(dlast, g->g[P_LNF_W], g->g[P_LNF_B], d->lnf, last, g->p[P_LNF_W], a->lnf_mean, a->lnf_rstd, BT, C);
    for (int l = L - 1; l >= 0; l--) {
        size_t oC = (size_t)l * BT * C, oF = (size_t)l * BT * F, o3 = (size_t)l * BT * 3 * C, oT = (size_t)l * BT;
        size_t oA = (size_t)l * B * NH * T * T;
        float *res = l == 0 ? a->encoded : a->res3 + (size_t)(l - 1) * BT * C;
        float *dres = l == 0 ? d->encoded : d->res3 + (size_t)(l - 1) * BT * C;
        residual_backward(d->res2 + oC, d->fcproj + oC, d->res3 + oC, (size_t)BT * C);
        matmul_backward(g, d->fch_gelu + oF, g->g[P_DOWN_W] + (size_t)l * C * F, g->g[P_DOWN_B] + l * C,
                        d->fcproj + oC, a->fch_gelu + oF, g->p[P_DOWN_W] + (size_t)l * C * F, BT, F, C);
        gelu_backward(d->fch + oF, a->fch + oF, d->fch_gelu + oF, (size_t)BT * F);
        matmul_backward(g, d->ln2 + oC, g->g[P_UP_W] + (size_t)l * F * C, g->g[P_UP_B] + l * F,
                        d->fch + oF, a->ln2 + oC, g->p[P_UP_W] + (size_t)l * F * C, BT, C, F);
        layernorm_backward(d->res2 + oC, g->g[P_LN2_W] + l * C, g->g[P_LN2_B] + l * C, d->ln2 + oC,
                           a->res2 + oC, g->p[P_LN2_W] + l * C, a->ln2_mean + oT, a->ln2_rstd + oT, BT, C);
        residual_backward(dres, d->attproj + oC, d->res2 + oC, (size_t)BT * C);
        matmul_backward(g, d->atty + oC, g->g[P_PROJ_W] + (size_t)l * C * C, g->g[P_PROJ_B] + l * C,
                        d->attproj + oC, a->atty + oC, g->p[P_PROJ_W] + (size_t)l * C * C, BT, C, C);
        AttnJob j = {NULL, NULL, a->att + oA, d->qkv + o3, d->preatt + oA, d->att + oA,
                     a->qkv + o3, d->atty + oC, T, C, NH};
        par_for(B * NH, 1, attention_backward_range, &j);
        matmul_backward(g, d->ln1 + oC, g->g[P_QKV_W] + (size_t)l * 3 * C * C, g->g[P_QKV_B] + l * 3 * C,
                        d->qkv + o3, a->ln1 + oC, g->p[P_QKV_W] + (size_t)l * 3 * C * C, BT, C, 3 * C);
        layernorm_backward(dres, g->g[P_LN1_W] + l * C, g->g[P_LN1_B] + l * C, d->ln1 + oC,
                           res, g->p[P_LN1_W] + l * C, a->ln1_mean + oT, a->ln1_rstd + oT, BT, C);
    }
    encoder_backward(g->g[P_TOK_EMB], g->g[P_POS_EMB], d->encoded, inp, BT, T, C);
}

/* torch.nn.utils.clip_grad_norm_ semantics. */
double gpt_clip_grads(GPT *g, double max_norm) {
    double ss = 0.0;
    for (size_t i = 0; i < g->n; i++) ss += (double)g->grads[i] * g->grads[i];
    double norm = sqrt(ss), coef = max_norm / (norm + 1e-6);
    if (coef < 1.0)
        for (size_t i = 0; i < g->n; i++) g->grads[i] = (float)(g->grads[i] * coef);
    return norm;
}

/* torch.optim.AdamW update order; weight decay only on 2-D tensors. t counts from 1. */
void gpt_adamw(GPT *g, float lr, float b1, float b2, float eps, float wd, long t) {
    float bc1 = 1.0f - powf(b1, (float)t), bc2_sqrt = sqrtf(1.0f - powf(b2, (float)t));
    size_t off = 0;
    for (int k = 0; k < P_COUNT; k++) {
        float decay = PARAM_DECAY[k] ? 1.0f - lr * wd : 1.0f;
        for (size_t i = off; i < off + g->size[k]; i++) {
            float gr = g->grads[i];
            g->m[i] = b1 * g->m[i] + (1.0f - b1) * gr;
            g->v[i] = b2 * g->v[i] + (1.0f - b2) * gr * gr;
            float denom = sqrtf(g->v[i]) / bc2_sqrt + eps;
            g->params[i] = g->params[i] * decay - (lr / bc1) * g->m[i] / denom;
        }
        off += g->size[k];
    }
}

/* ---------- KV-cache decoder for sampling ---------- */

int dec_alloc(Decoder *d, const GPT *g) {
    memset(d, 0, sizeof *d);
    const Shape s = g->s;
    size_t kv = (size_t)s.L * s.T * s.C;
    d->g = g;
    d->k = calloc(2 * kv, sizeof(float));
    d->v = d->k + kv;
    size_t n = (size_t)s.C * 7 + s.F + s.T + s.V + 2;
    d->x = calloc(n, sizeof(float));
    if (!d->k || !d->x) return -1;
    d->xn = d->x + s.C; d->qkv = d->xn + s.C; d->y = d->qkv + 3 * s.C;
    d->tmp = d->y + s.C; d->ff = d->tmp + s.C;
    d->scores = d->ff + s.F; d->logits = d->scores + s.T; d->stat = d->logits + s.V;
    return 0;
}

void dec_free(Decoder *d) { free(d->k); free(d->x); }

static void matvec(float *out, const float *w, const float *x, const float *b, int rows, int cols) {
    sgemm_nt(1, rows, cols, x, cols, w, cols, out, rows, 0, b);
}

const float *dec_step(Decoder *d, int token) {
    const GPT *g = d->g;
    const Shape s = g->s;
    int C = s.C, F = s.F, H = s.NH, hs = C / H, t = d->pos;
    float scale = 1.0f / sqrtf((float)hs);
    for (int i = 0; i < C; i++) d->x[i] = g->p[P_TOK_EMB][(size_t)token * C + i] + g->p[P_POS_EMB][(size_t)t * C + i];
    for (int l = 0; l < s.L; l++) {
        float *K = d->k + (size_t)l * s.T * C, *Vc = d->v + (size_t)l * s.T * C;
        layernorm_forward(d->xn, d->stat, d->stat + 1, d->x, g->p[P_LN1_W] + l * C, g->p[P_LN1_B] + l * C, 1, C);
        matvec(d->qkv, g->p[P_QKV_W] + (size_t)l * 3 * C * C, d->xn, g->p[P_QKV_B] + l * 3 * C, 3 * C, C);
        memcpy(K + (size_t)t * C, d->qkv + C, C * sizeof(float));
        memcpy(Vc + (size_t)t * C, d->qkv + 2 * C, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            const float *q = d->qkv + h * hs;
            float mx = -10000.0f, sum = 0.0f;
            for (int p = 0; p <= t; p++) {
                const float *k = K + (size_t)p * C + h * hs;
                float v = 0.0f;
                for (int i = 0; i < hs; i++) v += q[i] * k[i];
                d->scores[p] = v * scale;
                if (d->scores[p] > mx) mx = d->scores[p];
            }
            for (int p = 0; p <= t; p++) { d->scores[p] = expf(d->scores[p] - mx); sum += d->scores[p]; }
            float *y = d->y + h * hs;
            for (int i = 0; i < hs; i++) y[i] = 0.0f;
            for (int p = 0; p <= t; p++) {
                const float *v = Vc + (size_t)p * C + h * hs;
                float w = d->scores[p] / sum;
                for (int i = 0; i < hs; i++) y[i] += w * v[i];
            }
        }
        matvec(d->tmp, g->p[P_PROJ_W] + (size_t)l * C * C, d->y, g->p[P_PROJ_B] + l * C, C, C);
        for (int i = 0; i < C; i++) d->x[i] += d->tmp[i];
        layernorm_forward(d->xn, d->stat, d->stat + 1, d->x, g->p[P_LN2_W] + l * C, g->p[P_LN2_B] + l * C, 1, C);
        matvec(d->ff, g->p[P_UP_W] + (size_t)l * F * C, d->xn, g->p[P_UP_B] + l * F, F, C);
        gelu_forward(d->ff, d->ff, (size_t)F);
        matvec(d->tmp, g->p[P_DOWN_W] + (size_t)l * C * F, d->ff, g->p[P_DOWN_B] + l * C, C, F);
        for (int i = 0; i < C; i++) d->x[i] += d->tmp[i];
    }
    layernorm_forward(d->xn, d->stat, d->stat + 1, d->x, g->p[P_LNF_W], g->p[P_LNF_B], 1, C);
    matvec(d->logits, g->p[P_HEAD_W], d->xn, g->p[P_HEAD_B], s.V, C);
    d->pos++;
    return d->logits;
}
