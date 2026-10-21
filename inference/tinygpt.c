/*
 * tinygpt.c - minimal GPT inference in pure C.
 * Loads weights exported by train/model.py, generates text byte by byte.
 * No dependencies beyond libc and libm. Runs on ARM (Kindle) and x86.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    int vocab_size;
    int context_len;
    int n_layers;
    int n_heads;
    int d_model;
    int d_ff;
    int head_dim;
} Config;

typedef struct {
    float *tok_emb;   /* [vocab_size, d_model] */
    float *pos_emb;   /* [context_len, d_model] */
    float *ln1_w;     /* [n_layers, d_model] */
    float *ln1_b;     /* [n_layers, d_model] */
    float *qkv_w;     /* [n_layers, 3*d_model, d_model] */
    float *qkv_b;     /* [n_layers, 3*d_model] */
    float *proj_w;    /* [n_layers, d_model, d_model] */
    float *proj_b;    /* [n_layers, d_model] */
    float *ln2_w;     /* [n_layers, d_model] */
    float *ln2_b;     /* [n_layers, d_model] */
    float *up_w;      /* [n_layers, d_ff, d_model] */
    float *up_b;      /* [n_layers, d_ff] */
    float *down_w;    /* [n_layers, d_model, d_ff] */
    float *down_b;    /* [n_layers, d_model] */
    float *ln_f_w;    /* [d_model] */
    float *ln_f_b;    /* [d_model] */
    float *head_w;    /* [vocab_size, d_model] */
    float *head_b;    /* [vocab_size] */
} Weights;

/* scratch buffers allocated once */
typedef struct {
    float *x;         /* [context_len, d_model] - residual stream */
    float *ln_out;    /* [d_model] */
    float *qkv;       /* [3 * d_model] */
    float *all_q;     /* [context_len, d_model] */
    float *all_k;     /* [context_len, d_model] */
    float *all_v;     /* [context_len, d_model] */
    float *attn_vec;  /* [d_model] */
    float *scores;    /* [context_len] */
    float *proj_out;  /* [d_model] */
    float *ff_up;     /* [d_ff] */
    float *ff_down;   /* [d_model] */
    float *logits;    /* [vocab_size] */
} Scratch;

typedef struct {
    Config cfg;
    Weights w;
    Scratch s;
} Model;

/* ---------- math ---------- */

static void layernorm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float s = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * s * w[i] + b[i];
}

static void matvec(float *out, const float *w, const float *x, const float *b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = b ? b[i] : 0.0f;
        const float *wi = w + (long)i * cols;
        for (int j = 0; j < cols; j++) sum += wi[j] * x[j];
        out[i] = sum;
    }
}

static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax_n(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ---------- loading ---------- */

static float *read_floats(FILE *f, long count) {
    float *p = (float *)malloc((size_t)count * sizeof(float));
    if (!p) { fprintf(stderr, "oom (%ld floats)\n", count); exit(1); }
    if ((long)fread(p, sizeof(float), (size_t)count, f) != count) {
        fprintf(stderr, "truncated weights\n"); exit(1);
    }
    return p;
}

static float *alloc_f(long n) {
    float *p = (float *)calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "oom\n"); exit(1); }
    return p;
}

static Model *load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    Model *m = (Model *)calloc(1, sizeof(Model));
    Config *c = &m->cfg;
    int hdr[6];
    if (fread(hdr, sizeof(int), 6, f) != 6) { fprintf(stderr, "bad header\n"); exit(1); }
    c->vocab_size = hdr[0]; c->context_len = hdr[1]; c->n_layers = hdr[2];
    c->n_heads = hdr[3]; c->d_model = hdr[4]; c->d_ff = hdr[5];
    c->head_dim = c->d_model / c->n_heads;

    Weights *w = &m->w;
    long L = c->n_layers, D = c->d_model, F = c->d_ff, V = c->vocab_size, T = c->context_len;
    w->tok_emb = read_floats(f, V * D);
    w->pos_emb = read_floats(f, T * D);
    w->ln1_w   = read_floats(f, L * D);
    w->ln1_b   = read_floats(f, L * D);
    w->qkv_w   = read_floats(f, L * 3 * D * D);
    w->qkv_b   = read_floats(f, L * 3 * D);
    w->proj_w  = read_floats(f, L * D * D);
    w->proj_b  = read_floats(f, L * D);
    w->ln2_w   = read_floats(f, L * D);
    w->ln2_b   = read_floats(f, L * D);
    w->up_w    = read_floats(f, L * F * D);
    w->up_b    = read_floats(f, L * F);
    w->down_w  = read_floats(f, L * D * F);
    w->down_b  = read_floats(f, L * D);
    w->ln_f_w  = read_floats(f, D);
    w->ln_f_b  = read_floats(f, D);
    w->head_w  = read_floats(f, V * D);
    w->head_b  = read_floats(f, V);
    fclose(f);

    Scratch *s = &m->s;
    s->x        = alloc_f(T * D);
    s->ln_out   = alloc_f(D);
    s->qkv      = alloc_f(3 * D);
    s->all_q    = alloc_f(T * D);
    s->all_k    = alloc_f(T * D);
    s->all_v    = alloc_f(T * D);
    s->attn_vec = alloc_f(D);
    s->scores   = alloc_f(T);
    s->proj_out = alloc_f(D);
    s->ff_up    = alloc_f(F);
    s->ff_down  = alloc_f(D);
    s->logits   = alloc_f(V);

    return m;
}

/* ---------- forward (full sequence, returns logits for last position) ---------- */

static void forward(Model *m, const int *tokens, int seq_len) {
    Config *c = &m->cfg;
    Weights *w = &m->w;
    Scratch *s = &m->s;
    int D = c->d_model, H = c->n_heads, hd = c->head_dim;
    int T = seq_len;

    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            s->x[t * D + d] = w->tok_emb[tokens[t] * D + d] + w->pos_emb[t * D + d];

    for (int l = 0; l < c->n_layers; l++) {
        long lD = (long)l * D, l3DD = (long)l * 3 * D * D, l3D = (long)l * 3 * D;
        long lDD = (long)l * D * D, lFD = (long)l * c->d_ff * D, lF = (long)l * c->d_ff;
        long lDF = (long)l * D * c->d_ff;

        for (int t = 0; t < T; t++) {
            layernorm(s->ln_out, s->x + t * D, w->ln1_w + lD, w->ln1_b + lD, D);
            matvec(s->qkv, w->qkv_w + l3DD, s->ln_out, w->qkv_b + l3D, 3 * D, D);
            memcpy(s->all_q + t * D, s->qkv,         D * sizeof(float));
            memcpy(s->all_k + t * D, s->qkv + D,     D * sizeof(float));
            memcpy(s->all_v + t * D, s->qkv + 2 * D, D * sizeof(float));
        }

        for (int t = 0; t < T; t++) {
            memset(s->attn_vec, 0, D * sizeof(float));
            for (int h = 0; h < H; h++) {
                int off = h * hd;
                float scale = 1.0f / sqrtf((float)hd);
                for (int p = 0; p <= t; p++) {
                    float dot = 0;
                    for (int d = 0; d < hd; d++)
                        dot += s->all_q[t * D + off + d] * s->all_k[p * D + off + d];
                    s->scores[p] = dot * scale;
                }
                softmax_n(s->scores, t + 1);
                for (int p = 0; p <= t; p++)
                    for (int d = 0; d < hd; d++)
                        s->attn_vec[off + d] += s->scores[p] * s->all_v[p * D + off + d];
            }
            matvec(s->proj_out, w->proj_w + lDD, s->attn_vec, w->proj_b + lD, D, D);
            for (int d = 0; d < D; d++) s->x[t * D + d] += s->proj_out[d];
        }

        for (int t = 0; t < T; t++) {
            layernorm(s->ln_out, s->x + t * D, w->ln2_w + lD, w->ln2_b + lD, D);
            matvec(s->ff_up, w->up_w + lFD, s->ln_out, w->up_b + lF, c->d_ff, D);
            for (int i = 0; i < c->d_ff; i++) s->ff_up[i] = gelu_f(s->ff_up[i]);
            matvec(s->ff_down, w->down_w + lDF, s->ff_up, w->down_b + lD, D, c->d_ff);
            for (int d = 0; d < D; d++) s->x[t * D + d] += s->ff_down[d];
        }
    }

    layernorm(s->ln_out, s->x + (T - 1) * D, w->ln_f_w, w->ln_f_b, D);
    matvec(s->logits, w->head_w, s->ln_out, w->head_b, c->vocab_size, D);
}

/* ---------- sampling ---------- */

static int sample_top_k(float *logits, int n, float temp, int top_k) {
    for (int i = 0; i < n; i++) logits[i] /= temp;

    if (top_k > 0 && top_k < n) {
        float *tmp = (float *)malloc(n * sizeof(float));
        memcpy(tmp, logits, n * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++)
                if (tmp[j] > tmp[best]) best = j;
            float t = tmp[i]; tmp[i] = tmp[best]; tmp[best] = t;
        }
        float thr = tmp[top_k - 1];
        free(tmp);
        for (int i = 0; i < n; i++)
            if (logits[i] < thr) logits[i] = -1e9f;
    }

    softmax_n(logits, n);
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < n; i++) { cum += logits[i]; if (cum >= r) return i; }
    return n - 1;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <weights.bin> [-n count] [-t temp] [-k topk] [-p prompt] [-s seed] [--char]\n",
            argv[0]);
        return 1;
    }

    int max_tokens = 256;
    float temp = 0.8f;
    int top_k = 40;
    const char *prompt = "\n";
    int seed = -1, char_mode = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--char")) char_mode = 1;
    }

    if (seed < 0) seed = (int)time(NULL);
    srand((unsigned)seed);

    Model *m = load_model(argv[1]);
    Config *c = &m->cfg;
    fprintf(stderr, "model: vocab=%d ctx=%d layers=%d heads=%d d=%d ff=%d\n",
        c->vocab_size, c->context_len, c->n_layers, c->n_heads, c->d_model, c->d_ff);

    int prompt_len = (int)strlen(prompt);
    int max_seq = c->context_len + max_tokens;
    int *tokens = (int *)malloc(max_seq * sizeof(int));
    int seq_len = 0;
    for (int i = 0; i < prompt_len && seq_len < c->context_len; i++)
        tokens[seq_len++] = (unsigned char)prompt[i];

    for (int i = 0; i < max_tokens; i++) {
        int ctx_start = 0, ctx_len = seq_len;
        if (ctx_len > c->context_len) {
            ctx_start = ctx_len - c->context_len;
            ctx_len = c->context_len;
        }
        forward(m, tokens + ctx_start, ctx_len);

        float *logits_copy = (float *)malloc(c->vocab_size * sizeof(float));
        memcpy(logits_copy, m->s.logits, c->vocab_size * sizeof(float));
        int next = sample_top_k(logits_copy, c->vocab_size, temp, top_k);
        free(logits_copy);

        tokens[seq_len++] = next;
        if (char_mode) { putchar(next); fflush(stdout); }
    }

    if (!char_mode)
        for (int i = prompt_len; i < seq_len; i++) putchar(tokens[i]);
    putchar('\n');

    free(tokens);
    return 0;
}
