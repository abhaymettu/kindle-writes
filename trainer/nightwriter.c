/*
 * nightwriter - overnight char-level trainer for the Kindle.
 *
 *   nightwriter train     --data book.txt --run DIR [options]   train, checkpoint, write pages
 *   nightwriter sample    --init page.bin --data book.txt        re-sample a page from its weights
 *   nightwriter gradcheck --init w.bin --batch b.bin --out g.bin (used by trainer/gradcheck.py)
 *   nightwriter bench     [shape options]                        tok/s of a full training step
 *
 * Determinism: fixed seeds, no -ffast-math, a fixed two-way thread split whose results do
 * not depend on the split. A run replays bit-identically on the same build.
 */
#define _GNU_SOURCE
#include "gpt.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef GIT_HASH
#define GIT_HASH "unknown"
#endif

/* PLACEHOLDER thermal and battery thresholds. Replace from the M0 thermal trace.
 * Measured so far (2026-10-23): SoC steady at 32 C over 60 min of scalar SGEMM on the
 * charger. Nothing has been measured under the 2-thread NEON load these guard. */
#define SOC_PAUSE_C   60.0   /* PLACEHOLDER */
#define SOC_RESUME_C  50.0   /* PLACEHOLDER */
#define BATT_PAUSE_C  42.0   /* PLACEHOLDER, plan value */
#define BATT_RESUME_C 38.0   /* PLACEHOLDER, plan value */
#define BATT_STOP_PCT 15     /* plan value: stop when below this and discharging */

#define EXIT_STOPPED 3       /* deliberate stop (battery); a supervisor should not restart */

/* ---------- config ---------- */

typedef struct {   /* everything that shapes the math; stored in every checkpoint */
    Shape shape;
    int32_t B, accum, warmup, eval_every, eval_seqs, ckpt_every, top_k, page_chars;
    float lr, lr_min, b1, b2, eps, wd, clip, budget_hours, val_frac, temp, time_scale;
    int32_t token_mode;   /* 0: bytes; 1: BPE ids from --tokens/--vocab (corpus_hash covers both files) */
    int64_t total_steps, corpus_len;
    uint64_t init_seed, data_seed, sample_seed, corpus_hash;
    char prompt[64], final_prompt[64];
} TrainCfg;

typedef struct {
    int64_t step;
    uint64_t data_rng;
    double train_s, paused_s, pause_note_s, last_val;
    int32_t next_page, pad0;
    uint64_t writes;   /* checkpoint sequence number; the higher one is newest, even at equal steps */
} State;

typedef struct {
    char magic[8];
    uint32_t version, header_bytes;
    TrainCfg cfg;
    State st;
    uint64_t nparams;
} CkptHeader;

typedef struct {   /* operational knobs; not stored, may change between restarts */
    const char *data, *run, *init, *sysroot, *tokens, *vocab;
    int threads;
    long stop_after;
    int64_t extend_to;   /* >0: raise a finished run's total_steps to this, at lr_min */
    double soc_pause, soc_resume, batt_pause, batt_resume;
    int batt_stop;
} Ops;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "nightwriter: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void unescape(char *dst, const char *src, size_t cap) {
    size_t n = 0;
    for (; *src && n + 1 < cap; src++) {
        if (src[0] == '\\' && src[1] == 'n') { dst[n++] = '\n'; src++; }
        else dst[n++] = *src;
    }
    dst[n] = 0;
}

static void escape(char *dst, const char *src, size_t len, size_t cap) {
    size_t n = 0;
    for (size_t i = 0; i < len && n + 3 < cap; i++) {
        if (src[i] == '\n') { dst[n++] = '\\'; dst[n++] = 'n'; }
        else if (src[i] == '"') { dst[n++] = '\\'; dst[n++] = '"'; }
        else dst[n++] = src[i];
    }
    dst[n] = 0;
}

/* ---------- corpus ---------- */

#define MAXV 4096

/* Token id -> bytes. Byte mode is the identity vocabulary with no merges. In token mode id
 * 256+i is merges[i] = (a, b), the concatenation of ids a and b. */
typedef struct {
    int V, n_merges;
    uint8_t **str;
    int *len;
    int32_t (*merges)[2];
} Vocab;

typedef struct {
    int32_t *tok;           /* token ids; equal to the bytes in byte mode */
    int64_t n, n_train;     /* in tokens: train = [0, n_train), held-out = [n_train, n) */
    uint8_t *buf;           /* the text as bytes (decoded tokens in token mode), for the copy audit */
    int64_t nbytes, nbytes_train;
    uint8_t mask[MAXV];     /* ids that occur in the training split */
    Vocab voc;
    int token_mode;
    uint64_t hash;
    int32_t *sa;            /* suffix array of buf, for the copy audit */
} Corpus;

static const uint8_t *g_sa_text;
static int64_t g_sa_n;
#define SA_CAP 4096

static int sa_cmp(const void *pa, const void *pb) {
    int64_t a = *(const int32_t *)pa, b = *(const int32_t *)pb;
    int64_t la = g_sa_n - a, lb = g_sa_n - b, len = la < lb ? la : lb;
    if (len > SA_CAP) len = SA_CAP;
    int r = memcmp(g_sa_text + a, g_sa_text + b, (size_t)len);
    if (r) return r;
    if (len == SA_CAP) return a < b ? -1 : 1;
    return la < lb ? -1 : 1;
}

static uint64_t fnv1a(uint64_t h, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}

static void vocab_bytes(Vocab *v) {
    v->V = 256; v->n_merges = 0; v->merges = NULL;
    v->str = malloc(256 * sizeof *v->str);
    v->len = malloc(256 * sizeof *v->len);
    for (int i = 0; i < 256; i++) { v->str[i] = malloc(1); v->str[i][0] = (uint8_t)i; v->len[i] = 1; }
}

static void corpus_load(Corpus *c, const char *path, float val_frac, int T) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    c->n = ftell(f);
    fseek(f, 0, SEEK_SET);
    c->buf = malloc((size_t)c->n);
    if (!c->buf || (int64_t)fread(c->buf, 1, (size_t)c->n, f) != c->n) die("cannot read %s", path);
    fclose(f);
    c->n_train = c->n - (int64_t)(c->n * (double)val_frac);
    if (c->n_train < 4 * (T + 1) || c->n - c->n_train < 2 * (T + 1))
        die("corpus %s too small for ctx %d and val_frac %.3f", path, T, val_frac);
    c->hash = fnv1a(1469598103934665603ULL, c->buf, (size_t)c->n);
    c->nbytes = c->n; c->nbytes_train = c->n_train;
    c->tok = malloc((size_t)c->n * sizeof(int32_t));
    if (!c->tok) die("oom (corpus)");
    for (int64_t i = 0; i < c->n; i++) c->tok[i] = c->buf[i];
    vocab_bytes(&c->voc);
    c->token_mode = 0;
    memset(c->mask, 0, sizeof c->mask);
    for (int64_t i = 0; i < c->n_train; i++) c->mask[c->buf[i]] = 1;
    c->sa = NULL;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)len + 1);
    if (!b || fread(b, 1, (size_t)len, f) != (size_t)len) die("cannot read %s", path);
    fclose(f);
    *n = (size_t)len;
    return b;
}

static void vocab_load(Vocab *v, const uint8_t *b, size_t n, const char *path) {
    uint32_t hv[2];
    if (n < 16 || memcmp(b, "KWVOCAB1", 8)) die("%s is not a vocab file", path);
    memcpy(hv, b + 8, 8);
    v->V = (int)hv[0]; v->n_merges = (int)hv[1];
    if (v->V < 256 || v->V > MAXV || v->n_merges != v->V - 256) die("bad vocab header in %s", path);
    v->str = malloc((size_t)v->V * sizeof *v->str);
    v->len = malloc((size_t)v->V * sizeof *v->len);
    v->merges = malloc((size_t)v->n_merges * sizeof *v->merges);
    size_t at = 16;
    for (int i = 0; i < v->V; i++) {
        uint16_t l;
        if (at + 2 > n) die("truncated %s", path);
        memcpy(&l, b + at, 2); at += 2;
        if (!l || at + l > n) die("truncated %s", path);
        v->str[i] = malloc(l);
        memcpy(v->str[i], b + at, l); at += l;
        v->len[i] = l;
    }
    for (int i = 0; i < v->n_merges; i++) {
        uint16_t p[2];
        if (at + 4 > n) die("truncated %s", path);
        memcpy(p, b + at, 4); at += 4;
        v->merges[i][0] = p[0]; v->merges[i][1] = p[1];
        int a = p[0], c = p[1], id = 256 + i;
        if (a >= id || c >= id || v->len[a] + v->len[c] != v->len[id] ||
            memcmp(v->str[a], v->str[id], (size_t)v->len[a]) || memcmp(v->str[c], v->str[id] + v->len[a], (size_t)v->len[c]))
            die("merge %d in %s does not match its vocab entry", i, path);
    }
    if (at != n) die("trailing bytes in %s", path);
}

/* Token mode: ids and the train/held-out seam come from the tokens file. */
static void corpus_load_tokens(Corpus *c, const char *tok_path, const char *voc_path, int T) {
    size_t nt, nv;
    uint8_t *tb = read_file(tok_path, &nt), *vb = read_file(voc_path, &nv);
    vocab_load(&c->voc, vb, nv, voc_path);
    uint32_t h32[2];
    uint64_t h64[2];
    if (nt < 32 || memcmp(tb, "KWTOKS1", 8)) die("%s is not a tokens file", tok_path);
    memcpy(h32, tb + 8, 8);
    memcpy(h64, tb + 16, 16);
    if ((int)h32[0] != c->voc.V) die("%s has V %u but %s has V %d", tok_path, h32[0], voc_path, c->voc.V);
    c->n = (int64_t)h64[0]; c->n_train = (int64_t)h64[1];
    if (nt != 32 + 2 * (size_t)c->n || c->n_train > c->n) die("bad size or header in %s", tok_path);
    if (c->n_train < 4 * (T + 1) || c->n - c->n_train < 2 * (T + 1)) die("%s too small for ctx %d", tok_path, T);
    c->tok = malloc((size_t)c->n * sizeof(int32_t));
    c->nbytes = 0;
    for (int64_t i = 0; i < c->n; i++) {
        uint16_t id;
        memcpy(&id, tb + 32 + 2 * i, 2);
        if (id >= c->voc.V) die("token %u out of range in %s", id, tok_path);
        c->tok[i] = id;
        c->nbytes += c->voc.len[id];
        if (i + 1 == c->n_train) c->nbytes_train = c->nbytes;
    }
    c->buf = malloc((size_t)c->nbytes);
    for (int64_t i = 0, at = 0; i < c->n; i++) {
        memcpy(c->buf + at, c->voc.str[c->tok[i]], (size_t)c->voc.len[c->tok[i]]);
        at += c->voc.len[c->tok[i]];
    }
    c->hash = fnv1a(fnv1a(1469598103934665603ULL, tb, nt), vb, nv);
    c->token_mode = 1;
    memset(c->mask, 0, sizeof c->mask);
    for (int64_t i = 0; i < c->n_train; i++) c->mask[c->tok[i]] = 1;
    c->sa = NULL;
    free(tb); free(vb);
}

static void corpus_open(Corpus *c, const Ops *o, float val_frac, int T) {
    memset(c, 0, sizeof *c);
    if (o->tokens || o->vocab) {
        if (!o->tokens || !o->vocab) die("token mode needs both --tokens and --vocab");
        corpus_load_tokens(c, o->tokens, o->vocab, T);
    } else {
        if (!o->data) die("give --data, or --tokens and --vocab");
        corpus_load(c, o->data, val_frac, T);
    }
}

/* Same chunking as presplit() in prepare_tokens.py: letters, digits or other non-space,
 * each with an optional leading space; or whitespace, minus a final space that leads a chunk. */
static int byte_class(int b) {
    if (b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\f' || b == '\v') return 0;
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) return 1;
    if (b >= '0' && b <= '9') return 2;
    return 3;
}

static int64_t chunk_end(const uint8_t *s, int64_t i, int64_t n) {
    int64_t j;
    if (s[i] == ' ' && i + 1 < n && byte_class(s[i + 1])) {
        int k = byte_class(s[i + 1]);
        for (j = i + 2; j < n && byte_class(s[j]) == k; j++) {}
    } else if (byte_class(s[i])) {
        int k = byte_class(s[i]);
        for (j = i + 1; j < n && byte_class(s[j]) == k; j++) {}
    } else {
        for (j = i + 1; j < n && !byte_class(s[j]); j++) {}
        if (j < n && s[j - 1] == ' ' && j - 1 > i) j--;
    }
    return j;
}

/* BPE encode, the same rule as prepare_tokens.py: within each chunk, repeatedly merge every
 * occurrence of the lowest-rank adjacent pair. Byte mode (no merges) is the identity. */
static int64_t vocab_encode(const Vocab *v, const uint8_t *s, int64_t n, int32_t *out) {
    int64_t m = 0;
    for (int64_t i = 0; i < n;) {
        int64_t j = v->n_merges ? chunk_end(s, i, n) : n, k0 = m;
        for (int64_t t = i; t < j; t++) out[m++] = s[t];
        for (;;) {
            int best = v->n_merges;
            for (int64_t t = k0; t + 1 < m; t++)
                for (int r = 0; r < best; r++)
                    if (v->merges[r][0] == out[t] && v->merges[r][1] == out[t + 1]) { best = r; break; }
            if (best == v->n_merges) break;
            int64_t w = k0;
            for (int64_t t = k0; t < m; t++) {
                if (t + 1 < m && out[t] == v->merges[best][0] && out[t + 1] == v->merges[best][1]) { out[w++] = 256 + best; t++; }
                else out[w++] = out[t];
            }
            m = w;
        }
        i = j;
    }
    return m;
}

static void corpus_build_sa(Corpus *c) {
    c->sa = malloc((size_t)c->nbytes * sizeof(int32_t));
    if (!c->sa) die("oom (suffix array)");
    for (int64_t i = 0; i < c->nbytes; i++) c->sa[i] = (int32_t)i;
    g_sa_text = c->buf; g_sa_n = c->nbytes;
    qsort(c->sa, (size_t)c->nbytes, sizeof(int32_t), sa_cmp);
}

static int sa_char(const Corpus *c, int32_t suf, int64_t L) {
    return suf + L < c->nbytes ? c->buf[suf + L] : -1;
}

/* Longest substring of s that occurs verbatim in the corpus. */
static int64_t longest_copy(const Corpus *c, const uint8_t *s, int64_t m, int64_t *at_s, int64_t *at_c) {
    int64_t best = 0;
    *at_s = *at_c = 0;
    for (int64_t i = 0; i < m; i++) {
        int64_t lo = 0, hi = c->nbytes, L = 0;
        while (i + L < m && L < SA_CAP - 1) {
            int ch = s[i + L];
            int64_t a = lo, b = hi;
            while (a < b) { int64_t mid = (a + b) / 2; if (sa_char(c, c->sa[mid], L) < ch) a = mid + 1; else b = mid; }
            int64_t first = a;
            b = hi;
            while (a < b) { int64_t mid = (a + b) / 2; if (sa_char(c, c->sa[mid], L) <= ch) a = mid + 1; else b = mid; }
            if (first == a) break;
            lo = first; hi = a; L++;
        }
        if (L > best) { best = L; *at_s = i; *at_c = c->sa[lo]; }
    }
    return best;
}

/* ---------- sampling ---------- */

static int pick(const float *logits, const uint8_t *mask, int V, float temp, int top_k, uint64_t *rng) {
    float z[MAXV], sorted[MAXV];
    int n = 0;
    for (int i = 0; i < V; i++) {
        z[i] = (!mask || mask[i]) ? logits[i] / temp : -INFINITY;
        if (z[i] > -INFINITY) sorted[n++] = z[i];
    }
    float thr = -INFINITY;
    if (top_k > 0 && top_k < n) {
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++) if (sorted[j] > sorted[best]) best = j;
            float t = sorted[i]; sorted[i] = sorted[best]; sorted[best] = t;
        }
        thr = sorted[top_k - 1];
    }
    float mx = -INFINITY, sum = 0.0f;
    for (int i = 0; i < V; i++) if (z[i] >= thr && z[i] > mx) mx = z[i];
    for (int i = 0; i < V; i++) { z[i] = (z[i] >= thr && z[i] > -INFINITY) ? expf(z[i] - mx) : 0.0f; sum += z[i]; }
    float r = rng_uniform(rng) * sum, cum = 0.0f;
    int last = 0;
    for (int i = 0; i < V; i++) {
        if (z[i] <= 0.0f) continue;
        cum += z[i]; last = i;
        if (cum > r) return i;
    }
    return last;
}

/* ---------- lexicon decoding ----------
 * Optional decoder constraint: every run of letters the sampler emits must be a word
 * that occurs in the training split (or its capitalised form). The model still picks
 * every byte; disallowed bytes are masked like bytes absent from the training split.
 * Built from [0, n_train) only, so the held-out text never shapes the decoder. */

typedef struct { char **w; int n; } Lexicon;
static Lexicon g_lex;

static int is_letter(int ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void lexicon_build(const Corpus *c) {
    int cap = 1 << 16, n = 0;
    char **w = malloc((size_t)cap * sizeof *w);
    for (int64_t i = 0; i < c->nbytes_train;) {
        if (!is_letter(c->buf[i])) { i++; continue; }
        int64_t j = i;
        while (j < c->nbytes_train && is_letter(c->buf[j])) j++;
        for (int v = 0; v < 2; v++) {
            if (v && !(c->buf[i] >= 'a' && c->buf[i] <= 'z')) break;
            if (n == cap) { cap *= 2; w = realloc(w, (size_t)cap * sizeof *w); }
            char *s = malloc((size_t)(j - i + 1));
            memcpy(s, c->buf + i, (size_t)(j - i));
            s[j - i] = 0;
            if (v) s[0] = (char)(s[0] - 'a' + 'A');
            w[n++] = s;
        }
        i = j;
    }
    qsort(w, (size_t)n, sizeof *w, cmp_str);
    int u = 0;
    for (int i = 0; i < n; i++) if (!u || strcmp(w[u - 1], w[i])) w[u++] = w[i];
    g_lex.w = w; g_lex.n = u;
}

/* 2: s is a word, 1: a proper prefix of a word, 0: neither. */
static int lexicon_find(const char *s, int len) {
    int lo = 0, hi = g_lex.n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (strncmp(g_lex.w[mid], s, (size_t)len) < 0) lo = mid + 1; else hi = mid;
    }
    if (lo == g_lex.n || strncmp(g_lex.w[lo], s, (size_t)len)) return 0;
    return g_lex.w[lo][len] ? 1 : 2;
}

/* Narrows base to the bytes that keep the current word (the letters at the end of out) legal. */
static void lexicon_mask(const uint8_t *base, const uint8_t *out, int64_t len, uint8_t *m) {
    int64_t s = len;
    while (s > 0 && is_letter(out[s - 1])) s--;
    int wl = (int)(len - s);
    char buf[64];
    if (wl >= (int)sizeof buf - 1) { memcpy(m, base, 256); return; }
    memcpy(buf, out + s, (size_t)wl);
    int state = wl ? lexicon_find(buf, wl) : 2;
    for (int ch = 0; ch < 256; ch++) {
        if (!base[ch]) { m[ch] = 0; continue; }
        if (!is_letter(ch)) { m[ch] = state != 1; continue; }
        if (wl && !state) { m[ch] = 1; continue; }   /* prompt ended inside an unknown word */
        buf[wl] = (char)ch;
        m[ch] = lexicon_find(buf, wl + 1) != 0;
    }
    int any = 0;
    for (int ch = 0; ch < 256; ch++) any |= m[ch];
    if (!any) memcpy(m, base, 256);
}

/* Token mode: keeps a token only if its bytes, appended to out, complete no letter run
 * outside the lexicon and leave the trailing letter run a prefix of some lexicon word.
 * A word the prompt left unfinished and unknown runs free until it ends, as in byte mode. */
static void lexicon_mask_tokens(const Vocab *voc, const uint8_t *base, const uint8_t *out, int64_t len, uint8_t *m) {
    int64_t s = len;
    while (s > 0 && is_letter(out[s - 1])) s--;
    int wl0 = (int)(len - s), any = 0;
    char buf[64];
    if (wl0 >= (int)sizeof buf - 1) { memcpy(m, base, (size_t)voc->V); return; }
    memcpy(buf, out + s, (size_t)wl0);
    int free0 = wl0 && !lexicon_find(buf, wl0);
    for (int id = 0; id < voc->V; id++) {
        m[id] = 0;
        if (!base[id]) continue;
        int wl = wl0, freew = free0, ok = 1;
        for (int i = 0; i < voc->len[id] && ok; i++) {
            int ch = voc->str[id][i];
            if (is_letter(ch)) {
                if (freew) continue;
                if (wl >= (int)sizeof buf - 1) { freew = 1; continue; }
                buf[wl++] = (char)ch;
                ok = lexicon_find(buf, wl) != 0;
            } else {
                if (wl && !freew) ok = lexicon_find(buf, wl) == 2;
                wl = 0; freew = 0;
            }
        }
        m[id] = (uint8_t)ok;
        any |= ok;
        memcpy(buf, out + s, (size_t)wl0);
    }
    if (!any) memcpy(m, base, (size_t)voc->V);
}

/* Writes prompt + generated text to out, as bytes, stopping once at least n bytes follow the
 * prompt (n tokens in byte mode). The prompt is BPE-encoded with the corpus vocab. Past the
 * context the window slides by half: the cache restarts on the most recent T/2 tokens at
 * positions 0..T/2-1. out needs room for prompt + n + the longest vocab entry. */
static int64_t sample_text(const GPT *g, const Vocab *voc, const char *prompt, int n, uint64_t seed, float temp,
                           int top_k, const uint8_t *mask, uint8_t *out) {
    uint8_t lmask[MAXV];
    Decoder d;
    if (dec_alloc(&d, g)) die("oom (decoder)");
    int T = g->s.T, half = T / 2;
    int64_t plen = (int64_t)strlen(prompt), len = 0, nt = 0;
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(plen + n + 1));
    uint64_t rng = seed ? seed : 1;
    const float *lg = NULL;
    int64_t np = vocab_encode(voc, (const uint8_t *)prompt, plen, ids);
    memcpy(out, prompt, (size_t)plen);
    len = plen;
    for (nt = 0; nt < np; nt++) lg = dec_step(&d, ids[nt]);
    if (!lg) lg = dec_step(&d, '\n');
    while (len - plen < n) {
        const uint8_t *m = mask;
        if (g_lex.n && voc->n_merges) { lexicon_mask_tokens(voc, mask, out, len, lmask); m = lmask; }
        else if (g_lex.n) { lexicon_mask(mask, out, len, lmask); m = lmask; }
        int tok = pick(lg, m, g->s.V, temp, top_k, &rng);
        ids[nt++] = tok;
        memcpy(out + len, voc->str[tok], (size_t)voc->len[tok]);
        len += voc->len[tok];
        if (d.pos == T) {
            d.pos = 0;
            for (int64_t i = nt - half; i < nt - 1; i++) dec_step(&d, ids[i]);
        }
        lg = dec_step(&d, tok);
    }
    free(ids);
    dec_free(&d);
    return len;
}

/* ---------- weights files (inference/tinygpt.c format) ---------- */

static void export_bin(const GPT *g, const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) die("cannot write %s", tmp);
    int32_t hdr[6] = {g->s.V, g->s.T, g->s.L, g->s.NH, g->s.C, g->s.F};
    fwrite(hdr, sizeof hdr, 1, f);
    fwrite(g->params, sizeof(float), g->n, f);
    if (ferror(f) || fflush(f) || fsync(fileno(f)) || fclose(f) || rename(tmp, path)) die("cannot write %s", path);
}

static Shape read_bin_shape(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    int32_t h[6];
    if (fread(h, sizeof h, 1, f) != 1) die("bad header in %s", path);
    fclose(f);
    Shape s = {h[0], h[1], h[2], h[3], h[4], h[5]};
    return s;
}

static void load_bin(GPT *g, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 24, SEEK_SET);
    if (fread(g->params, sizeof(float), g->n, f) != g->n) die("truncated weights in %s", path);
    fclose(f);
}

/* ---------- checkpoints: two slots, CRC32, newest valid wins ---------- */

static uint32_t crc_table[256];

static uint32_t crc32_update(uint32_t crc, const void *data, size_t n) {
    if (!crc_table[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            crc_table[i] = c;
        }
    const uint8_t *p = data;
    crc = ~crc;
    while (n--) crc = crc_table[(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return ~crc;
}

static const char *const SLOT_NAME[2] = {"A", "B"};

static void ckpt_write(const char *run, int slot, const TrainCfg *cfg, const State *st, const GPT *g) {
    char tmp[1024], dst[1024];
    snprintf(tmp, sizeof tmp, "%s/ckpt/%s.tmp", run, SLOT_NAME[slot]);
    snprintf(dst, sizeof dst, "%s/ckpt/%s.ckpt", run, SLOT_NAME[slot]);
    CkptHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, "KWCKPT1", 8);
    h.version = 1; h.header_bytes = sizeof h; h.cfg = *cfg; h.st = *st; h.nparams = g->n;
    FILE *f = fopen(tmp, "wb");
    if (!f) die("cannot write %s: %s", tmp, strerror(errno));
    uint32_t crc = crc32_update(0, &h, sizeof h);
    fwrite(&h, sizeof h, 1, f);
    const float *parts[3] = {g->params, g->m, g->v};
    for (int i = 0; i < 3; i++) {
        crc = crc32_update(crc, parts[i], g->n * sizeof(float));
        fwrite(parts[i], sizeof(float), g->n, f);
    }
    fwrite(&crc, sizeof crc, 1, f);
    if (fflush(f) || fsync(fileno(f)) || fclose(f)) die("cannot write %s: %s", tmp, strerror(errno));
    if (rename(tmp, dst)) die("cannot rename %s: %s", tmp, strerror(errno));
    snprintf(tmp, sizeof tmp, "%s/ckpt", run);
    int dfd = open(tmp, O_RDONLY);
    if (dfd >= 0) { fsync(dfd); close(dfd); }
}

/* Returns the whole file if magic, size and CRC check out, else NULL. */
static uint8_t *ckpt_read(const char *path, CkptHeader *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = n > (long)(sizeof *h + 4) ? malloc((size_t)n) : NULL;
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    memcpy(h, buf, sizeof *h);
    uint32_t crc;
    memcpy(&crc, buf + n - 4, 4);
    if (memcmp(h->magic, "KWCKPT1", 8) || h->header_bytes != sizeof *h ||
        (long)(sizeof *h + 3 * h->nparams * sizeof(float) + 4) != n || crc32_update(0, buf, (size_t)n - 4) != crc) {
        fprintf(stderr, "nightwriter: %s is invalid (torn write or wrong build), ignoring\n", path);
        free(buf);
        return NULL;
    }
    return buf;
}

/* ---------- sensors (paths and values confirmed on device 2026-10-23) ---------- */

typedef struct {
    double soc_c, batt_c;
    int batt_pct;
    char batt_status[24], wifi[64], soc_src[48];
    long cpu_khz;
    const char *missing;   /* "soc" or "battery" when a guard cannot be read; training pauses */
} Sensors;

static int read_line(const char *root, const char *path, char *out, size_t cap) {
    char p[1024];
    snprintf(p, sizeof p, "%s%s", root, path);
    FILE *f = fopen(p, "r");
    if (!f) return -1;
    if (!fgets(out, (int)cap, f)) { fclose(f); return -1; }
    fclose(f);
    out[strcspn(out, "\r\n")] = 0;
    return 0;
}

static char g_batt_dir[512];

/* Prefers a supply with a capacity file. Failing that, keeps one with a status file so a
 * missing gauge can still be told apart from "on the charger". */
static void find_battery(const char *root) {
    char p[1024], line[64], f[600];
    snprintf(p, sizeof p, "%s/sys/class/power_supply", root);
    DIR *d = opendir(p);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        snprintf(f, sizeof f, "/sys/class/power_supply/%s/capacity", e->d_name);
        if (read_line(root, f, line, sizeof line) == 0) {
            snprintf(g_batt_dir, sizeof g_batt_dir, "/sys/class/power_supply/%s", e->d_name);
            break;
        }
        snprintf(f, sizeof f, "/sys/class/power_supply/%s/status", e->d_name);
        if (!g_batt_dir[0] && read_line(root, f, line, sizeof line) == 0)
            snprintf(g_batt_dir, sizeof g_batt_dir, "/sys/class/power_supply/%s", e->d_name);
    }
    closedir(d);
}

/* SoC temperature by zone type, never max-over-all: on the Kindle mtktscpu reads ~29 C while
 * mtktspcb2, mtktspcb3 and mtktsAP sit at a 125 C placeholder. Order: type "mtktscpu", then
 * the hottest zone whose type contains "cpu", then the hottest zone under 100 C. NAN if none,
 * which pauses training as sensor_missing. Named zones are trusted at any reading, so real
 * heat on the CPU still pauses. The "bd71827_bat" thermal zone (2800, unknown units) is not
 * the battery gauge; battery decisions read power_supply only. */
static void read_soc(const char *root, Sensors *s) {
    char p[96], type[32], line[32];
    double named = NAN, cpu = NAN, other = NAN;
    char cpu_src[48] = "", other_src[48] = "";
    for (int z = 0; z < 32; z++) {
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/temp", z);
        if (read_line(root, p, line, sizeof line) || !line[0]) continue;
        double t = atof(line);
        if (t > 1000 || t < -1000) t /= 1000.0;
        if (t < -40.0) continue;   /* error codes, not a temperature */
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/type", z);
        if (read_line(root, p, type, sizeof type)) type[0] = 0;
        char lower[32];
        size_t i = 0;
        for (; type[i]; i++) lower[i] = (char)(type[i] >= 'A' && type[i] <= 'Z' ? type[i] + 32 : type[i]);
        lower[i] = 0;
        if (!strcmp(type, "mtktscpu") && isnan(named)) {
            named = t;
            snprintf(s->soc_src, sizeof s->soc_src, "zone%d:%s", z, type);
        } else if (strstr(lower, "cpu")) {
            if (isnan(cpu) || t > cpu) { cpu = t; snprintf(cpu_src, sizeof cpu_src, "zone%d:%s", z, type); }
        } else if (t < 100.0) {
            if (isnan(other) || t > other) { other = t; snprintf(other_src, sizeof other_src, "zone%d:%s", z, type); }
        }
    }
    if (!isnan(named)) s->soc_c = named;
    else if (!isnan(cpu)) { s->soc_c = cpu; snprintf(s->soc_src, sizeof s->soc_src, "%s", cpu_src); }
    else if (!isnan(other)) { s->soc_c = other; snprintf(s->soc_src, sizeof s->soc_src, "%s", other_src); }
}

static void read_sensors(const char *root, Sensors *s) {
    char line[128], p[700];
    s->soc_c = s->batt_c = NAN;
    s->batt_pct = -1;
    s->cpu_khz = -1;
    strcpy(s->batt_status, "na");
    strcpy(s->soc_src, "none");
    read_soc(root, s);
    if (g_batt_dir[0]) {
        snprintf(p, sizeof p, "%s/capacity", g_batt_dir);
        if (!read_line(root, p, line, sizeof line)) s->batt_pct = atoi(line);
        snprintf(p, sizeof p, "%s/temp", g_batt_dir);
        if (!read_line(root, p, line, sizeof line)) {
            double t = atof(line);
            /* Kindle bd71827_bat/temp reads whole degrees C (29-31 on device, 2026-10-23);
             * values above 150 are taken as tenths, the convention on most other drivers. */
            s->batt_c = t > 150 ? t / 10.0 : t;
        }
        snprintf(p, sizeof p, "%s/status", g_batt_dir);
        if (!read_line(root, p, line, sizeof line)) snprintf(s->batt_status, sizeof s->batt_status, "%s", line);
    }
    /* Fail-safe: no SoC reading pauses. A battery whose charge or temperature cannot be read
     * pauses unless it reports Charging. */
    s->missing = isnan(s->soc_c) ? "soc"
               : (s->batt_pct < 0 || isnan(s->batt_c)) && strcmp(s->batt_status, "Charging") ? "battery" : NULL;
    if (!read_line(root, "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", line, sizeof line))
        s->cpu_khz = atol(line);
    char wlan[32] = "na", lipc[16] = "na";
    read_line(root, "/sys/class/net/wlan0/operstate", wlan, sizeof wlan);
    if (!root[0]) {
        FILE *f = popen("lipc-get-prop com.lab126.cmd wirelessEnable 2>/dev/null", "r");
        if (f) {
            if (fgets(line, sizeof line, f)) { line[strcspn(line, "\r\n ")] = 0; if (line[0]) snprintf(lipc, sizeof lipc, "%s", line); }
            pclose(f);
        }
    }
    snprintf(s->wifi, sizeof s->wifi, "wlan0=%s;wirelessEnable=%s", wlan, lipc);
}

/* ---------- run ---------- */

typedef struct {
    TrainCfg cfg;
    State st;
    Ops ops;
    GPT g;
    Corpus corpus;
    FILE *log;
    int slot;            /* slot holding the newest checkpoint, -1 if none */
    int *inp, *tgt;
    const char *pause_note;   /* cause of pauses since the last page; not checkpointed */
    double val_char;          /* held-out nats per char from the last eval_val */
} Run;

static volatile sig_atomic_t g_stop;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static void log_row(Run *r, const char *event, float lr, float train, float val, double tok_s, double gnorm,
                    const Sensors *s) {
    char ts[32];
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tm);
    fprintf(r->log, "%s\t%.1f\t%s\t%lld\t%.3g\t%.4f\t%.4f\t%.1f\t%.3f\t%.1f\t%d\t%.1f\t%s\t%ld\t%s\n", ts,
            r->st.train_s, event, (long long)r->st.step, lr, train, val, tok_s, gnorm, s->soc_c, s->batt_pct,
            s->batt_c, s->batt_status, s->cpu_khz, s->wifi);
    fflush(r->log);
}

static float lr_at(const TrainCfg *c, int64_t step) {
    if (step < c->warmup) return c->lr * (float)(step + 1) / (float)c->warmup;
    if (c->total_steps <= c->warmup) return c->lr;
    double p = (double)(step - c->warmup) / (double)(c->total_steps - c->warmup);
    if (p > 1.0) p = 1.0;
    return c->lr_min + 0.5f * (c->lr - c->lr_min) * (float)(1.0 + cos(M_PI * p));
}

static double page_due_s(const TrainCfg *c, int page) {
    static const double first[] = {0, 300, 900};
    double s = page < 3 ? first[page] : 1800.0 * (page - 2);
    return s * c->time_scale;
}

static void fill_window(const Corpus *c, int64_t start, int T, int *inp, int *tgt) {
    for (int t = 0; t < T; t++) { inp[t] = c->tok[start + t]; tgt[t] = c->tok[start + t + 1]; }
}

/* Mean loss per token over eval_seqs windows spaced evenly through the held-out split.
 * Also sets r->val_char: the same nats summed over every target, divided by the bytes those
 * target tokens decode to. Equal to the per-token loss in byte mode. */
static float eval_val(Run *r) {
    int T = r->cfg.shape.T, B = r->cfg.B, n = r->cfg.eval_seqs;
    int64_t span = r->corpus.n - r->corpus.n_train - T - 1, chars = 0;
    double sum = 0.0;
    for (int i0 = 0; i0 < n; i0 += B) {
        int nb = n - i0 < B ? n - i0 : B;
        for (int b = 0; b < nb; b++) {
            int64_t start = r->corpus.n_train + (n > 1 ? span * (i0 + b) / (n - 1) : 0);
            fill_window(&r->corpus, start, T, r->inp + b * T, r->tgt + b * T);
        }
        sum += (double)gpt_forward(&r->g, r->inp, r->tgt, nb, T) * nb;
        for (int i = 0; i < nb * T; i++) chars += r->corpus.voc.len[r->tgt[i]];
    }
    r->val_char = sum * T / (double)chars;
    return (float)(sum / n);
}

static float train_step(Run *r, double *gnorm) {
    int T = r->cfg.shape.T, B = r->cfg.B;
    int64_t span = r->corpus.n_train - T - 1;
    double loss = 0.0;
    gpt_zero_grad(&r->g);
    for (int a = 0; a < r->cfg.accum; a++) {
        for (int b = 0; b < B; b++)
            fill_window(&r->corpus, (int64_t)(rng_next(&r->st.data_rng) % (uint64_t)span), T,
                        r->inp + b * T, r->tgt + b * T);
        loss += gpt_forward(&r->g, r->inp, r->tgt, B, T);
        gpt_backward(&r->g, r->inp, r->tgt, B, T, 1.0f / r->cfg.accum);
    }
    *gnorm = gpt_clip_grads(&r->g, r->cfg.clip);
    gpt_adamw(&r->g, lr_at(&r->cfg, r->st.step), r->cfg.b1, r->cfg.b2, r->cfg.eps, r->cfg.wd, r->st.step + 1);
    return (float)(loss / r->cfg.accum);
}

static void checkpoint(Run *r) {
    int slot = r->slot == 0 ? 1 : 0;
    r->st.writes++;
    ckpt_write(r->ops.run, slot, &r->cfg, &r->st, &r->g);
    r->slot = slot;
}

static void write_page(Run *r, const char *prompt, const char *note, const Sensors *s) {
    double t0 = mono_s();
    const TrainCfg *c = &r->cfg;
    float val = eval_val(r);
    r->st.last_val = val;
    size_t plen = strlen(prompt);
    uint8_t *text = malloc(plen + (size_t)c->page_chars + MAXV);
    int64_t len = sample_text(&r->g, &r->corpus.voc, prompt, c->page_chars, c->sample_seed, c->temp, c->top_k, r->corpus.mask, text);
    int64_t at_s, at_c, copy = longest_copy(&r->corpus, text + plen, len - (int64_t)plen, &at_s, &at_c);

    char path[1024], esc_prompt[160], esc_copy[SA_CAP * 2 + 8];
    int page = r->st.next_page;
    snprintf(path, sizeof path, "%s/pages/%02d.bin", r->ops.run, page);
    export_bin(&r->g, path);
    snprintf(path, sizeof path, "%s/pages/%02d.txt", r->ops.run, page);
    FILE *f = fopen(path, "w");
    if (!f) die("cannot write %s", path);
    int64_t mins = (int64_t)(r->st.train_s / 60);
    double chars = (double)r->st.step * c->B * c->accum * c->shape.T;
    if (c->token_mode)
        fprintf(f, "page %02d | step %lld | train time %02lld:%02lld | %.2fM tokens read | held-out loss %.2f nats/char\n\n",
                page, (long long)r->st.step, (long long)(mins / 60), (long long)(mins % 60), chars / 1e6, r->val_char);
    else
        fprintf(f, "page %02d | step %lld | train time %02lld:%02lld | %.2fM chars read | held-out loss %.2f\n\n",
                page, (long long)r->st.step, (long long)(mins / 60), (long long)(mins % 60), chars / 1e6, val);
    fwrite(text, 1, (size_t)len, f);
    escape(esc_copy, (const char *)text + plen + at_s, (size_t)copy, sizeof esc_copy);
    escape(esc_prompt, prompt, plen, sizeof esc_prompt);
    fprintf(f, "\n\n---\n");
    if (c->token_mode)
        fprintf(f, "held-out loss %.4f nats/char (%.3f bits/char; %.4f nats/token), %d windows of %d tokens from the held-out tokens, never trained on\n",
                r->val_char, r->val_char / M_LN2, val, c->eval_seqs, c->shape.T);
    else
        fprintf(f, "held-out loss %.4f nats/char (%.3f bits/char), %d windows of %d chars from the last %.0f%% of the book, never trained on\n",
                val, val / M_LN2, c->eval_seqs, c->shape.T, c->val_frac * 100);
    fprintf(f, "longest copied span: %lld chars, at book offset %lld: \"%s\"\n", (long long)copy, (long long)at_c, esc_copy);
    fprintf(f, "prompt \"%s\" | seed %llu | temp %.2f | top-k %d | weights pages/%02d.bin\n", esc_prompt,
            (unsigned long long)c->sample_seed, c->temp, c->top_k, page);
    if (note && note[0]) fprintf(f, "%s\n", note);
    if (ferror(f) || fflush(f) || fsync(fileno(f)) || fclose(f)) die("cannot write %s", path);
    free(text);
    r->st.next_page++;
    r->st.train_s += mono_s() - t0;
    log_row(r, "page", lr_at(c, r->st.step), NAN, val, 0, 0, s);
    if (c->token_mode) log_row(r, "val_char", lr_at(c, r->st.step), NAN, (float)r->val_char, 0, 0, s);
    fprintf(stderr, "page %02d at step %lld: held-out %.4f, longest copy %lld chars\n", page,
            (long long)r->st.step, c->token_mode ? r->val_char : val, (long long)copy);
    checkpoint(r);
}

static void write_run_json(const Run *r) {
    char path[1024];
    snprintf(path, sizeof path, "%s/run.json", r->ops.run);
    if (access(path, F_OK) == 0) return;
    FILE *f = fopen(path, "w");
    if (!f) die("cannot write %s", path);
    const TrainCfg *c = &r->cfg;
    char p1[160], p2[160];
    escape(p1, c->prompt, strlen(c->prompt), sizeof p1);
    escape(p2, c->final_prompt, strlen(c->final_prompt), sizeof p2);
    fprintf(f,
            "{\n  \"git\": \"%s\",\n  \"data\": \"%s\",\n  \"corpus_bytes\": %lld,\n  \"corpus_fnv1a64\": \"%016llx\",\n"
            "  \"val_frac\": %.4f,\n  \"shape\": {\"vocab\": %d, \"ctx\": %d, \"layers\": %d, \"heads\": %d, \"d_model\": %d, \"d_ff\": %d},\n"
            "  \"params\": %zu,\n  \"micro_batch\": %d,\n  \"accum\": %d,\n  \"lr\": %g,\n  \"lr_min\": %g,\n  \"warmup\": %d,\n"
            "  \"betas\": [%g, %g],\n  \"eps\": %g,\n  \"weight_decay\": %g,\n  \"clip\": %g,\n  \"total_steps\": %lld,\n"
            "  \"budget_hours\": %g,\n  \"init_seed\": %llu,\n  \"data_seed\": %llu,\n  \"sample_seed\": %llu,\n"
            "  \"temp\": %g,\n  \"top_k\": %d,\n  \"page_chars\": %d,\n  \"prompt\": \"%s\",\n  \"final_prompt\": \"%s\",\n"
            "  \"time_scale\": %g,\n  \"threads\": %d,\n  \"sampling_mask\": \"bytes present in the training split\"\n}\n",
            GIT_HASH, r->ops.data, (long long)r->corpus.n, (unsigned long long)r->corpus.hash, c->val_frac,
            c->shape.V, c->shape.T, c->shape.L, c->shape.NH, c->shape.C, c->shape.F, r->g.n, c->B, c->accum, c->lr,
            c->lr_min, c->warmup, c->b1, c->b2, c->eps, c->wd, c->clip, (long long)c->total_steps, c->budget_hours,
            (unsigned long long)c->init_seed, (unsigned long long)c->data_seed, (unsigned long long)c->sample_seed,
            c->temp, c->top_k, c->page_chars, p1, p2, c->time_scale, r->ops.threads);
    if (c->token_mode) {
        fseek(f, -3, SEEK_CUR);   /* reopen the object: drop "\n}\n" */
        fprintf(f, ",\n  \"token_mode\": \"bpe\",\n  \"tokens\": \"%s\",\n  \"vocab\": \"%s\",\n"
                   "  \"tokens_vocab_fnv1a64\": \"%016llx\",\n  \"n_tokens\": %lld,\n  \"n_train_tokens\": %lld,\n"
                   "  \"text_bytes\": %lld,\n  \"train_bytes\": %lld,\n  \"val_loss_unit\": \"val_char log rows and pages: nats/char\"\n}\n",
                r->ops.tokens, r->ops.vocab, (unsigned long long)r->corpus.hash, (long long)r->corpus.n,
                (long long)r->corpus.n_train, (long long)r->corpus.nbytes, (long long)r->corpus.nbytes_train);
    }
    fclose(f);
}

/* Blocks while too warm or while a guard sensor cannot be read. Returns seconds paused. */
static double thermal_pause(Run *r, Sensors *s) {
    const Ops *o = &r->ops;
    if (!(s->missing || s->soc_c >= o->soc_pause || s->batt_c >= o->batt_pause)) return 0;
    double t0 = mono_s();
    if (s->missing) {
        log_row(r, "sensor_missing", 0, NAN, NAN, 0, 0, s);
        fprintf(stderr, "sensor_missing: no readable %s sensor, pausing\n", s->missing);
        r->pause_note = "sensor missing";
    }
    log_row(r, "pause", 0, NAN, NAN, 0, 0, s);
    fprintf(stderr, "pause: soc %.1f C (%s), battery %.1f C\n", s->soc_c, s->soc_src, s->batt_c);
    while (!g_stop) {
        sleep(30);
        read_sensors(o->sysroot, s);
        log_row(r, "paused", 0, NAN, NAN, 0, 0, s);
        if (!s->missing && !(s->soc_c > o->soc_resume) && !(s->batt_c > o->batt_resume)) break;
    }
    double dt = mono_s() - t0;
    log_row(r, "resume", 0, NAN, NAN, 0, 0, s);
    return dt;
}

static void train_main(TrainCfg cfg, Ops ops) {
    Run r;
    memset(&r, 0, sizeof r);
    r.ops = ops;
    r.slot = -1;
    r.pause_note = "warm";
    char path[1024];
    mkdir(ops.run, 0755);
    snprintf(path, sizeof path, "%s/ckpt", ops.run); mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/pages", ops.run); mkdir(path, 0755);

    CkptHeader h[2];
    uint8_t *buf[2] = {NULL, NULL};
    int any_file = 0;
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof path, "%s/ckpt/%s.ckpt", ops.run, SLOT_NAME[i]);
        any_file |= access(path, F_OK) == 0;
        buf[i] = ckpt_read(path, &h[i]);
    }
    int extended = 0;
    int pick_slot = buf[0] && (!buf[1] || h[0].st.writes > h[1].st.writes) ? 0 : buf[1] ? 1 : -1;
    snprintf(path, sizeof path, "%s/pages/01.txt", ops.run);
    if (pick_slot < 0 && any_file && access(path, F_OK) != 0) {
        fprintf(stderr, "nightwriter: no valid checkpoint and no page after 00; starting fresh\n");
        any_file = 0;
    }
    if (pick_slot < 0 && any_file) die("checkpoint files exist in %s/ckpt but none is valid; move them aside to start fresh", ops.run);

    if (pick_slot >= 0) {
        r.cfg = h[pick_slot].cfg;
        r.st = h[pick_slot].st;
        r.slot = pick_slot;
        fprintf(stderr, "resuming from slot %s at step %lld (stored config overrides command-line training flags)\n",
                SLOT_NAME[pick_slot], (long long)r.st.step);
        /* A second night on the same book. The target is absolute so a restart with the same
         * flag is a plain resume. Only a finished run extends, and the extra steps hold lr at
         * lr_min (lr = lr_min with no warmup makes lr_at constant) rather than restarting the
         * cosine with a jump. The change is stored in the checkpoint, not in run.json. */
        if (ops.extend_to > 0 && ops.extend_to != r.cfg.total_steps) {
            if (ops.extend_to < r.cfg.total_steps)
                die("--extend-to %lld is below the run's total_steps %lld", (long long)ops.extend_to, (long long)r.cfg.total_steps);
            if (r.cfg.total_steps == 0 || r.st.step < r.cfg.total_steps)
                die("run is not finished (step %lld of %lld); resume it without --extend-to first",
                    (long long)r.st.step, (long long)r.cfg.total_steps);
            fprintf(stderr, "extending total_steps %lld -> %lld at lr_min %g\n", (long long)r.cfg.total_steps,
                    (long long)ops.extend_to, r.cfg.lr_min);
            r.cfg.total_steps = ops.extend_to;
            r.cfg.lr = r.cfg.lr_min;
            r.cfg.warmup = 0;
            extended = 1;
        }
    } else {
        if (ops.extend_to > 0) die("--extend-to needs an existing run to resume");
        r.cfg = cfg;
        r.st.data_rng = cfg.data_seed ? cfg.data_seed : 1;
    }
    if ((int)strlen(r.cfg.prompt) >= r.cfg.shape.T / 2 || (int)strlen(r.cfg.final_prompt) >= r.cfg.shape.T / 2)
        die("prompts must be shorter than ctx/2 (%d)", r.cfg.shape.T / 2);
    corpus_open(&r.corpus, &ops, r.cfg.val_frac, r.cfg.shape.T);
    if (pick_slot >= 0 && r.cfg.token_mode != r.corpus.token_mode) die("this run was started in the other corpus mode");
    r.cfg.token_mode = r.corpus.token_mode;
    if (r.corpus.voc.V != r.cfg.shape.V) die("vocab %d does not match model vocab %d", r.corpus.voc.V, r.cfg.shape.V);
    if (pick_slot >= 0 && (r.corpus.hash != r.cfg.corpus_hash || r.corpus.n != r.cfg.corpus_len))
        die("%s is not the corpus this run was started on", ops.tokens ? ops.tokens : ops.data);
    r.cfg.corpus_hash = r.corpus.hash;
    r.cfg.corpus_len = r.corpus.n;
    if (gpt_alloc(&r.g, r.cfg.shape, r.cfg.B)) die("oom (model)");
    if (pick_slot >= 0) {
        const float *p = (const float *)(buf[pick_slot] + sizeof(CkptHeader));
        memcpy(r.g.params, p, r.g.n * sizeof(float));
        memcpy(r.g.m, p + r.g.n, r.g.n * sizeof(float));
        memcpy(r.g.v, p + 2 * r.g.n, r.g.n * sizeof(float));
    } else if (ops.init) {
        load_bin(&r.g, ops.init);
    } else {
        gpt_random_init(&r.g, r.cfg.init_seed);
    }
    free(buf[0]); free(buf[1]);
    r.inp = malloc(sizeof(int) * r.cfg.B * r.cfg.shape.T);
    r.tgt = malloc(sizeof(int) * r.cfg.B * r.cfg.shape.T);
    par_init(ops.threads);
    write_run_json(&r);

    snprintf(path, sizeof path, "%s/train.log", ops.run);
    int fresh_log = access(path, F_OK) != 0;
    r.log = fopen(path, "a");
    if (!r.log) die("cannot open %s", path);
    if (fresh_log)
        fprintf(r.log, "time\ttrain_s\tevent\tstep\tlr\ttrain_loss\tval_loss\ttok_s\tgrad_norm\tsoc_c\tbatt_pct\tbatt_c\tbatt_status\tcpu_khz\twifi\n");

    double t_sa = mono_s();
    corpus_build_sa(&r.corpus);
    fprintf(stderr, "params %zu | corpus %lld %s (%lld train, %lld held-out) | suffix array %.1fs | threads %d\n",
            r.g.n, (long long)r.corpus.n, r.corpus.token_mode ? "tokens" : "bytes", (long long)r.corpus.n_train,
            (long long)(r.corpus.n - r.corpus.n_train), mono_s() - t_sa, ops.threads);

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    find_battery(ops.sysroot);
    Sensors s;
    read_sensors(ops.sysroot, &s);
    fprintf(stderr, "sensors: soc %.1f C from %s | battery %s (%d%%, %.1f C, %s)\n", s.soc_c, s.soc_src,
            g_batt_dir[0] ? g_batt_dir : "none", s.batt_pct, s.batt_c, s.batt_status);
    log_row(&r, "start", 0, NAN, NAN, 0, 0, &s);
    if (extended) log_row(&r, "extend", r.cfg.lr, NAN, NAN, 0, 0, &s);

    if (r.st.next_page == 0) write_page(&r, r.cfg.prompt, "", &s);

    long done_here = 0;
    int tokens_per_step = r.cfg.B * r.cfg.accum * r.cfg.shape.T;
    while (r.cfg.total_steps == 0 || r.st.step < r.cfg.total_steps) {
        read_sensors(ops.sysroot, &s);
        if (s.batt_pct >= 0 && s.batt_pct < ops.batt_stop && !strcmp(s.batt_status, "Discharging")) {
            char note[128];
            snprintf(note, sizeof note, "stopped: battery %d%% and discharging at step %lld", s.batt_pct, (long long)r.st.step);
            write_page(&r, r.cfg.prompt, note, &s);
            log_row(&r, "stop", 0, NAN, NAN, 0, 0, &s);
            exit(EXIT_STOPPED);
        }
        double paused = thermal_pause(&r, &s);
        r.st.paused_s += paused;
        r.st.pause_note_s += paused;
        if (g_stop) break;

        double t0 = mono_s(), gnorm;
        float lr = lr_at(&r.cfg, r.st.step);
        float loss = train_step(&r, &gnorm);
        double dt = mono_s() - t0;
        r.st.step++;
        r.st.train_s += dt;
        done_here++;
        float val = NAN;
        if (r.cfg.eval_every > 0 && r.st.step % r.cfg.eval_every == 0) {
            double te = mono_s();
            val = r.st.last_val = eval_val(&r);
            r.st.train_s += mono_s() - te;
            if (r.cfg.token_mode) {
                log_row(&r, "val_char", lr, NAN, (float)r.val_char, 0, 0, &s);
                fprintf(stderr, "step %lld | held-out %.4f nats/token, %.4f nats/char\n", (long long)r.st.step, val, r.val_char);
            }
        }
        log_row(&r, "step", lr, loss, val, tokens_per_step / dt, gnorm, &s);
        fprintf(stderr, "step %lld | loss %.4f | lr %.2e | %.1f tok/s | soc %.1f C\n", (long long)r.st.step, loss, lr,
                tokens_per_step / dt, s.soc_c);

        if (r.cfg.total_steps == 0 && r.st.train_s >= 180.0) {
            double rate = r.st.step / r.st.train_s;
            r.cfg.total_steps = (int64_t)(0.9 * r.cfg.budget_hours * 3600.0 * rate);
            if (r.cfg.total_steps <= r.st.step) r.cfg.total_steps = r.st.step + 1;
            fprintf(stderr, "calibrated: %.4f steps/s -> total_steps %lld\n", rate, (long long)r.cfg.total_steps);
            log_row(&r, "calibrated", lr, NAN, NAN, 0, 0, &s);
        }

        int last = r.cfg.total_steps > 0 && r.st.step >= r.cfg.total_steps;
        if (last || r.st.train_s >= page_due_s(&r.cfg, r.st.next_page)) {
            char note[96] = "";
            if (r.st.pause_note_s > 0) snprintf(note, sizeof note, "paused %.0f min: %s", r.st.pause_note_s / 60, r.pause_note);
            r.st.pause_note_s = 0;
            r.pause_note = "warm";
            write_page(&r, last && r.cfg.final_prompt[0] ? r.cfg.final_prompt : r.cfg.prompt, note, &s);
        } else if (r.cfg.ckpt_every > 0 && r.st.step % r.cfg.ckpt_every == 0) {
            checkpoint(&r);
        }
        if (ops.stop_after > 0 && done_here >= ops.stop_after) break;
    }
    checkpoint(&r);
    log_row(&r, g_stop ? "signal" : "exit", 0, NAN, NAN, 0, 0, &s);
    fclose(r.log);
}

/* ---------- other subcommands ---------- */

static void gradcheck_main(const char *init, const char *batch, const char *out, float lr, int threads) {
    par_init(threads);
    FILE *f = fopen(batch, "rb");
    int32_t bt[2];
    if (!f || fread(bt, sizeof bt, 1, f) != 1) die("cannot read %s", batch);
    int B = bt[0], T = bt[1];
    int *inp = malloc(sizeof(int) * B * T), *tgt = malloc(sizeof(int) * B * T);
    if (fread(inp, sizeof(int), (size_t)B * T, f) != (size_t)B * T || fread(tgt, sizeof(int), (size_t)B * T, f) != (size_t)B * T)
        die("truncated %s", batch);
    fclose(f);
    GPT g;
    if (gpt_alloc(&g, read_bin_shape(init), B)) die("oom");
    if (T > g.s.T) die("batch T %d exceeds ctx %d", T, g.s.T);
    load_bin(&g, init);
    float loss = gpt_forward(&g, inp, tgt, B, T);
    gpt_zero_grad(&g);
    gpt_backward(&g, inp, tgt, B, T, 1.0f);
    FILE *o = fopen(out, "wb");
    if (!o) die("cannot write %s", out);
    fwrite(&loss, sizeof loss, 1, o);
    fwrite(g.grads, sizeof(float), g.n, o);
    gpt_clip_grads(&g, 1.0);
    gpt_adamw(&g, lr, 0.9f, 0.99f, 1e-8f, 0.1f, 1);
    fwrite(g.params, sizeof(float), g.n, o);
    fclose(o);
    printf("loss %.6f\n", loss);
}

static void sample_main(const Ops *o, float val_frac, const char *prompt, uint64_t seed,
                        float temp, int top_k, int chars, int lexicon) {
    const char *init = o->init;
    int threads = o->threads;
    par_init(threads);
    GPT g;
    if (gpt_alloc(&g, read_bin_shape(init), 1)) die("oom");
    load_bin(&g, init);
    Corpus c;
    if ((int)strlen(prompt) >= g.s.T / 2) die("prompt must be shorter than ctx/2 (%d)", g.s.T / 2);
    corpus_open(&c, o, val_frac, g.s.T);
    if (c.voc.V != g.s.V) die("vocab %d does not match model vocab %d", c.voc.V, g.s.V);
    corpus_build_sa(&c);
    if (lexicon) {
        lexicon_build(&c);
        fprintf(stderr, "lexicon: %d words from the training split\n", g_lex.n);
    }
    size_t plen = strlen(prompt);
    uint8_t *text = malloc(plen + (size_t)chars + MAXV);
    int64_t len = sample_text(&g, &c.voc, prompt, chars, seed, temp, top_k, c.mask, text);
    fwrite(text, 1, (size_t)len, stdout);
    int64_t at_s, at_c, copy = longest_copy(&c, text + plen, len - (int64_t)plen, &at_s, &at_c);
    fprintf(stderr, "\nlongest copied span: %lld chars at book offset %lld\n", (long long)copy, (long long)at_c);
}

static void bench_main(TrainCfg cfg, int threads, int steps) {
    par_init(threads);
    GPT g;
    if (gpt_alloc(&g, cfg.shape, cfg.B)) die("oom");
    gpt_random_init(&g, 42);
    int T = cfg.shape.T, n = cfg.B * T;
    int *inp = malloc(sizeof(int) * n), *tgt = malloc(sizeof(int) * n);
    uint64_t rng = 7;
    for (int i = 0; i < n; i++) { inp[i] = (int)(rng_next(&rng) % 96) + 32; tgt[i] = (int)(rng_next(&rng) % 96) + 32; }
    double t0 = 0;
    for (int s = 0; s <= steps; s++) {
        if (s == 1) t0 = mono_s();
        gpt_zero_grad(&g);
        gpt_forward(&g, inp, tgt, cfg.B, T);
        gpt_backward(&g, inp, tgt, cfg.B, T, 1.0f);
        gpt_adamw(&g, 1e-4f, 0.9f, 0.99f, 1e-8f, 0.1f, s + 1);
    }
    double dt = mono_s() - t0;
    const Shape *sh = &cfg.shape;
    double fwd = 2.0 * sh->L * ((double)sh->C * 3 * sh->C + (double)sh->C * sh->C + 2.0 * sh->C * sh->F) +
                 2.0 * sh->C * sh->V + 2.0 * sh->L * sh->C * (sh->T + 1.0);
    double tok_s = (double)n * steps / dt;
    printf("threads %d | micro-batch %dx%d | %.1f tok/s | %.3f s/micro-step | ~%.3f GFLOP/s (3x forward FLOP count)\n",
           threads, cfg.B, T, tok_s, dt / steps, 3.0 * fwd * tok_s / 1e9);
    printf("rss: params+grads+adam %.1f MB, activations+grads %.1f MB\n", 4.0 * g.n * 4 / 1e6,
           2.0 * g.act_floats * 4 / 1e6);
}

/* ---------- cli ---------- */

static TrainCfg default_cfg(void) {
    TrainCfg c;
    memset(&c, 0, sizeof c);
    Shape s = {256, 128, 4, 4, 128, 512};
    c.shape = s;
    c.B = 4; c.accum = 4; c.warmup = 100; c.eval_every = 25; c.eval_seqs = 32; c.ckpt_every = 10;
    c.top_k = 40; c.page_chars = 1800;
    c.lr = 1e-3f; c.lr_min = 1e-4f; c.b1 = 0.9f; c.b2 = 0.99f; c.eps = 1e-8f; c.wd = 0.1f; c.clip = 1.0f;
    c.val_frac = 0.05f; c.temp = 0.8f; c.time_scale = 1.0f;
    c.init_seed = 1337; c.data_seed = 2027; c.sample_seed = 1342;
    /* Fixed demo prompt for every page: the book's last heading, "CHAPTER LXI.", as it occurs
     * in data/pride.txt (byte 711746, inside the held-out tail, so never trained on). The plan
     * named "CHAPTER I." and "CHAPTER LXII.", which do not occur: this edition has no plain
     * CHAPTER I. heading (only the "Heading to Chapter I." caption) and ends at LXI. */
    strcpy(c.prompt, "CHAPTER LXI.\n\n");
    return c;
}

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  nightwriter train --data FILE --run DIR (--steps N | --budget-hours H) [--threads 2]\n"
        "      [--layers 4 --dmodel 128 --heads 4 --dff 512 --ctx 128] [--batch 4 --accum 4]\n"
        "      [--lr 1e-3 --lr-min 1e-4 --warmup 100 --wd 0.1] [--eval-every 25 --eval-seqs 32 --ckpt-every 10]\n"
        "      [--seed 1337 --data-seed 2027 --sample-seed 1342 --prompt 'CHAPTER LXI.\\n\\n' --final-prompt P]\n"
        "      [--temp 0.8 --top-k 40 --page-chars 1800 --time-scale 1] [--init w.bin] [--stop-after N]\n"
        "      [--extend-to N]   resume a finished run and train on to step N at lr_min\n"
        "      [--sysroot DIR --soc-pause C --soc-resume C --batt-pause C --batt-resume C --batt-stop PCT]\n"
        "  nightwriter sample --init w.bin --data FILE [--prompt P --sample-seed S --temp T --top-k K --page-chars N --lexicon 0|1]\n"
        "  nightwriter gradcheck --init w.bin --batch b.bin --out g.bin [--lr 1e-3] [--threads N]\n"
        "  nightwriter bench [--threads N] [--batch B] [--bench-steps S] [shape flags] [--tokens F]\n"
        "  token mode: --tokens F --vocab F in place of --data (train, sample); see prepare_tokens.py\n"
        "  nightwriter encode --vocab F --data FILE   print the BPE ids of FILE, one per line (parity check)\n");
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    const char *cmd = argv[1];
    TrainCfg c = default_cfg();
    Ops o = {NULL, NULL, NULL, "", NULL, NULL, 2, 0, 0, SOC_PAUSE_C, SOC_RESUME_C, BATT_PAUSE_C, BATT_RESUME_C, BATT_STOP_PCT};
    const char *batch = NULL, *out = NULL;
    int bench_steps = 3, lexicon = 0;
    for (int i = 2; i < argc; i++) {
        const char *k = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) die("missing value for %s", k);
        i++;
        if (!strcmp(k, "--data")) o.data = v;
        else if (!strcmp(k, "--run")) o.run = v;
        else if (!strcmp(k, "--init")) o.init = v;
        else if (!strcmp(k, "--sysroot")) o.sysroot = v;
        else if (!strcmp(k, "--tokens")) o.tokens = v;
        else if (!strcmp(k, "--vocab")) o.vocab = v;
        else if (!strcmp(k, "--threads")) o.threads = atoi(v);
        else if (!strcmp(k, "--stop-after")) o.stop_after = atol(v);
        else if (!strcmp(k, "--extend-to")) o.extend_to = atoll(v);
        else if (!strcmp(k, "--soc-pause")) o.soc_pause = atof(v);
        else if (!strcmp(k, "--soc-resume")) o.soc_resume = atof(v);
        else if (!strcmp(k, "--batt-pause")) o.batt_pause = atof(v);
        else if (!strcmp(k, "--batt-resume")) o.batt_resume = atof(v);
        else if (!strcmp(k, "--batt-stop")) o.batt_stop = atoi(v);
        else if (!strcmp(k, "--batch-file") || !strcmp(k, "--batch") ) {
            if (!strcmp(cmd, "gradcheck")) batch = v; else c.B = atoi(v);
        }
        else if (!strcmp(k, "--out")) out = v;
        else if (!strcmp(k, "--bench-steps")) bench_steps = atoi(v);
        else if (!strcmp(k, "--lexicon")) lexicon = atoi(v);
        else if (!strcmp(k, "--steps")) c.total_steps = atoll(v);
        else if (!strcmp(k, "--budget-hours")) c.budget_hours = (float)atof(v);
        else if (!strcmp(k, "--layers")) c.shape.L = atoi(v);
        else if (!strcmp(k, "--dmodel")) c.shape.C = atoi(v);
        else if (!strcmp(k, "--heads")) c.shape.NH = atoi(v);
        else if (!strcmp(k, "--dff")) c.shape.F = atoi(v);
        else if (!strcmp(k, "--ctx")) c.shape.T = atoi(v);
        else if (!strcmp(k, "--accum")) c.accum = atoi(v);
        else if (!strcmp(k, "--lr")) c.lr = (float)atof(v);
        else if (!strcmp(k, "--lr-min")) c.lr_min = (float)atof(v);
        else if (!strcmp(k, "--warmup")) c.warmup = atoi(v);
        else if (!strcmp(k, "--wd")) c.wd = (float)atof(v);
        else if (!strcmp(k, "--val-frac")) c.val_frac = (float)atof(v);
        else if (!strcmp(k, "--eval-every")) c.eval_every = atoi(v);
        else if (!strcmp(k, "--eval-seqs")) c.eval_seqs = atoi(v);
        else if (!strcmp(k, "--ckpt-every")) c.ckpt_every = atoi(v);
        else if (!strcmp(k, "--seed")) c.init_seed = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--data-seed")) c.data_seed = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--sample-seed")) c.sample_seed = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--prompt")) unescape(c.prompt, v, sizeof c.prompt);
        else if (!strcmp(k, "--final-prompt")) unescape(c.final_prompt, v, sizeof c.final_prompt);
        else if (!strcmp(k, "--temp")) c.temp = (float)atof(v);
        else if (!strcmp(k, "--top-k")) c.top_k = atoi(v);
        else if (!strcmp(k, "--page-chars")) c.page_chars = atoi(v);
        else if (!strcmp(k, "--time-scale")) c.time_scale = (float)atof(v);
        else die("unknown flag %s", k);
    }
    if (c.shape.C % c.shape.NH) die("d_model must divide by heads");
    if (o.tokens) {
        FILE *tf = fopen(o.tokens, "rb");
        uint32_t hv[4];
        if (!tf || fread(hv, sizeof hv, 1, tf) != 1 || memcmp(hv, "KWTOKS1", 8)) die("cannot read tokens header of %s", o.tokens);
        fclose(tf);
        c.shape.V = (int32_t)hv[2];
        c.token_mode = 1;
    }

    if (!strcmp(cmd, "train")) {
        if (!(o.data || o.tokens) || !o.run) usage();
        if (o.init) {
            Shape s = read_bin_shape(o.init);
            c.shape = s;
        }
        if (c.total_steps <= 0 && c.budget_hours <= 0 && o.extend_to <= 0) die("give --steps or --budget-hours");
        train_main(c, o);
    } else if (!strcmp(cmd, "sample")) {
        if (!o.init || !(o.data || o.tokens)) usage();
        sample_main(&o, c.val_frac, c.prompt, c.sample_seed, c.temp, c.top_k, c.page_chars, lexicon);
    } else if (!strcmp(cmd, "gradcheck")) {
        if (!o.init || !batch || !out) usage();
        gradcheck_main(o.init, batch, out, c.lr, o.threads);
    } else if (!strcmp(cmd, "encode")) {
        if (!o.vocab || !o.data) usage();
        size_t nv, nd;
        uint8_t *vb = read_file(o.vocab, &nv), *db = read_file(o.data, &nd);
        Vocab v;
        vocab_load(&v, vb, nv, o.vocab);
        int32_t *ids = malloc(sizeof(int32_t) * (nd + 1));
        int64_t m = vocab_encode(&v, db, (int64_t)nd, ids);
        for (int64_t i = 0; i < m; i++) printf("%d\n", ids[i]);
    } else if (!strcmp(cmd, "bench")) {
        bench_main(c, o.threads, bench_steps);
    } else {
        usage();
    }
    return 0;
}
