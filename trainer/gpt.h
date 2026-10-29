/*
 * gpt.h - the train/model.py TinyGPT in C: forward, backward, AdamW, KV-cache decoder.
 * Structure follows llm.c train_gpt2.c (fp32 CPU reference), untied head, erf GELU.
 * Parameters live in one buffer in the exact order inference/tinygpt.c loads, so a
 * header plus that buffer is a tinygpt.c weights file.
 */
#ifndef GPT_H
#define GPT_H

#include <stddef.h>
#include <stdint.h>

typedef struct { int32_t V, T, L, NH, C, F; } Shape;   /* vocab, ctx, layers, heads, d_model, d_ff */

enum {
    P_TOK_EMB, P_POS_EMB, P_LN1_W, P_LN1_B, P_QKV_W, P_QKV_B, P_PROJ_W, P_PROJ_B,
    P_LN2_W, P_LN2_B, P_UP_W, P_UP_B, P_DOWN_W, P_DOWN_B, P_LNF_W, P_LNF_B,
    P_HEAD_W, P_HEAD_B, P_COUNT
};

#define ACT_LIST(X) \
    X(encoded, BT * C) X(ln1, L * BT * C) X(ln1_mean, L * BT) X(ln1_rstd, L * BT) \
    X(qkv, L * BT * 3 * C) X(atty, L * BT * C) X(preatt, L * B * NH * T * T) \
    X(att, L * B * NH * T * T) X(attproj, L * BT * C) X(res2, L * BT * C) \
    X(ln2, L * BT * C) X(ln2_mean, L * BT) X(ln2_rstd, L * BT) X(fch, L * BT * F) \
    X(fch_gelu, L * BT * F) X(fcproj, L * BT * C) X(res3, L * BT * C) X(lnf, BT * C) \
    X(lnf_mean, BT) X(lnf_rstd, BT) X(logits, BT * V) X(probs, BT * V) X(losses, BT)

typedef struct {
#define X(name, n) float *name;
    ACT_LIST(X)
#undef X
} Acts;

typedef struct {
    Shape s;
    int B;                     /* micro-batch the activations are sized for */
    size_t size[P_COUNT], n;   /* floats per tensor, total */
    float *params, *grads, *m, *v;
    float *p[P_COUNT], *g[P_COUNT];
    Acts a, da;
    float *act_mem, *dact_mem;
    size_t act_floats;
    float *scratch_w, *scratch_a, *scratch_b;   /* transposes for matmul backward */
} GPT;

extern const char *const PARAM_NAMES[P_COUNT];
extern const int PARAM_DECAY[P_COUNT];

void par_init(int nthreads);
void sgemm_nt(int M, int N, int K, const float *A, int lda, const float *B, int ldb,
              float *C, int ldc, int acc, const float *bias);

int  gpt_alloc(GPT *g, Shape s, int B);
void gpt_free(GPT *g);
void gpt_random_init(GPT *g, uint64_t seed);
float gpt_forward(GPT *g, const int *inp, const int *tgt, int B, int T);
void gpt_zero_grad(GPT *g);
void gpt_backward(GPT *g, const int *inp, const int *tgt, int B, int T, float loss_scale);
double gpt_clip_grads(GPT *g, double max_norm);
void gpt_adamw(GPT *g, float lr, float b1, float b2, float eps, float wd, long t);

typedef struct {
    const GPT *g;
    int pos;
    float *k, *v;   /* [L][T][C] */
    float *x, *xn, *qkv, *y, *tmp, *ff, *scores, *logits, *stat;
} Decoder;

int  dec_alloc(Decoder *d, const GPT *g);
void dec_free(Decoder *d);
const float *dec_step(Decoder *d, int token);   /* feeds token at d->pos, returns logits */

uint64_t rng_next(uint64_t *s);
float rng_uniform(uint64_t *s);

#endif
