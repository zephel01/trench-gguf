// tools/hessian_dump.cpp (trench-gguf) - dumps what the per-expert Hessians of a MoE model are built from, through
// llama.cpp's eval callback, in one pass over a calibration text (the same chunking as llama-imatrix: -c <ctx>, several
// chunks a batch as separate sequences).
//
//   hessian_dump <model.gguf> <calib.txt> <out dir> [--gpu] [--threads=N] [--ctx=2048] [--ub=8192] [--chunks=N]
//                [--dump-down] [--layers=a-b] [--dense[=<regex>]] [--dense-chunks=N] [--no-moe]
//
// For every MUL_MAT_ID whose weight is blk.L.ffn_gate_exps.weight it writes the op's input rows (the gate/up input of
// layer L, one row a token, the same for all experts) and the routing ids:
//   <out>/blk.L.x.bf16      n_tokens x n_embd      bf16 (ggml's round-to-nearest)
//   <out>/blk.L.ids.i32     n_tokens x n_expert_used int32
//   <out>/blk.L.w_<v>.f32   n_tokens x n_expert_used f32, the routing weights: v = raw (ffn_moe_weights), norm (after
//                           normalization), scaled (after the model's scale) - whichever the graph names; the last
//                           stage present is what multiplies the expert outputs
// --dump-down also writes the input of blk.L.ffn_down_exps.weight (one row a token and slot, 3-4x the bytes of x):
//   <out>/blk.L.a.bf16      n_tokens x n_expert_used x n_ff
// Without it, the offline side recomputes a = silu(gate x) * (up x) from x and the original gate/up weights (the ones
// being quantized) - cheaper on disk: DeepSeek V4 Flash at 1M tokens is 8 KB a token a layer for x (344 GB in all)
// against 24 KB for a.
// The rows of every file are in the same order (batch order), so row r of blk.L.x and blk.L.ids belong together.
// <out>/info.txt records the token count, the shapes and the chunking.
//
// --dense[=<regex>] also writes the inputs of the plain MUL_MATs whose weight is a per-layer dense tensor
// (blk.L.<name>.weight, the name matched by <regex>; default: the attention projections, the indexer projections and
// the shared expert's down projection - see DENSE_DEFAULT), for GPTQ / LDLQ of the dense tensors:
//   <out>/blk.L.din.<name>.bf16   rows x [groups x] ne0 bf16, one row a src1 row (a token; the indexer's attn_k: a
//                                 compressed position). A weight used as a batched 3-D matmul (DeepSeek V4's grouped
//                                 attn_output_a, [ne0, rank, groups]) gets token-major rows of all groups:
//                                 [token][group][ne0], so group g's Hessian is the sum over tokens of its slice.
// Several weights that read the same input tensor (attn_q_a, attn_kv, the compressor and indexer.proj all read
// attn_norm's output) share one file. <out>/dense.txt maps every weight to its file: "<weight> <file> <ne0> <groups>
// <rows>". The shared experts' gate/up and the router read the MoE input, which blk.L.x already holds (the offline
// side takes their Hessian from there). --dense-chunks=N dumps the dense inputs during the first N chunks only (the
// rows are big: DeepSeek V4.1 ~100 KB a token a layer; 16 chunks = 32K tokens is 4x the widest input).
// --no-moe skips the expert inputs (a dense-only pass).
//
// --gpu: dense tensors on the GPU, experts on the host (as tools/imatrix.py run --gpu / teacher --gpu); the inputs
// are read back from wherever the op ran. Needs a ROCm/CUDA llama.cpp build (the patched one of setup.sh for models
// bigger than RAM); without --gpu a CPU build is enough.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

static const char * DENSE_DEFAULT =
    "attn_(q_a|q_b|kv|output_a|output_b|compressor_kv|compressor_gate)|indexer\\.(proj|attn_q_b|attn_k)|ffn_down_shexp";

struct DenseFile {
    std::string file;
    int64_t ne0 = 0, groups = 1, rows = 0;
};

struct Dump {
    std::string dir;
    bool down = false;
    bool moe = true;
    bool dense = false;
    bool dense_active = false;          // set by main: still within --dense-chunks
    std::regex dense_re;
    std::map<std::string, DenseFile> dense_files;     // file name -> shape and rows written
    std::map<std::string, std::string> dense_weight;  // weight name -> file name
    std::map<const ggml_tensor *, std::string> dense_seen;   // this batch: input tensor -> file (cleared per decode)
    int lo = 0, hi = 1 << 30;
    std::map<std::string, FILE *> files;
    std::map<int, int64_t> rows;        // tokens written per layer (from the gate op)
    std::map<int, std::string> wname;   // the last ffn_moe_weights* node name seen per layer
    int n_embd = 0, n_ff = 0, n_ids = 0;
    std::vector<char> buf;
    std::vector<uint16_t> bf;
    std::vector<int32_t> ids;

    FILE * file(const std::string & name) {
        auto it = files.find(name);
        if (it != files.end()) return it->second;
        FILE * f = fopen((dir + "/" + name).c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s/%s\n", dir.c_str(), name.c_str()); exit(1); }
        files[name] = f;
        return f;
    }
};

static int layer_of(const char * name, const char * kind) {
    // "blk.12.ffn_gate_exps.weight" -> 12 when kind matches, else -1. The scheduler's copy of a weight to another
    // backend (op offload of the host experts to the GPU) is named "<backend>#blk.12....#<n>": look inside the '#'s.
    const char * h = strchr(name, '#');
    if (h) name = h + 1;
    if (strncmp(name, "blk.", 4)) return -1;
    char * end;
    long L = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.') return -1;
    return strncmp(end + 1, kind, strlen(kind)) == 0 ? (int) L : -1;
}

// "blk.12.attn_q_b.weight" (or a backend copy "<backend>#blk.12.attn_q_b.weight#0") -> layer 12, short "attn_q_b"
static int dense_of(const char * name, std::string & shortname) {
    const char * h = strchr(name, '#');
    std::string n = h ? std::string(h + 1) : std::string(name);
    if (h) { const size_t e = n.find('#'); if (e != std::string::npos) n.resize(e); }
    if (n.compare(0, 4, "blk.")) return -1;
    char * end;
    long L = strtol(n.c_str() + 4, &end, 10);
    if (end == n.c_str() + 4 || *end != '.') return -1;
    std::string rest(end + 1);
    const std::string suf = ".weight";
    if (rest.size() <= suf.size() || rest.compare(rest.size() - suf.size(), suf.size(), suf)) return -1;
    shortname = rest.substr(0, rest.size() - suf.size());
    if (shortname.find("_exps") != std::string::npos) return -1;
    return (int) L;
}

static void get_tensor(const ggml_tensor * t, std::vector<char> & buf) {
    buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    Dump & d = *(Dump *) ud;
    static std::map<std::string, int> seen;
    static const bool debug = getenv("HESSIAN_DUMP_DEBUG") != nullptr;
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT) {
            if (!d.dense || !d.dense_active) return false;
            std::string sn;
            const int L = dense_of(t->src[0]->name, sn);
            if (L < d.lo || L > d.hi || !std::regex_match(sn, d.dense_re)) return false;
            if (debug && seen[t->src[0]->name]++ == 0)
                fprintf(stderr, "mul_mat src0 '%s' [%ld %ld %ld] src1 '%s' [%ld %ld %ld %ld] %s -> layer %d dense %s\n", t->src[0]->name,
                        (long) t->src[0]->ne[0], (long) t->src[0]->ne[1], (long) t->src[0]->ne[2], t->src[1]->name,
                        (long) t->src[1]->ne[0], (long) t->src[1]->ne[1], (long) t->src[1]->ne[2], (long) t->src[1]->ne[3],
                        ggml_type_name(t->src[1]->type), L, sn.c_str());
            return true;
        }
        if (t->op == GGML_OP_MUL_MAT_ID) {
            if (!d.moe) return false;
            const int Lg = layer_of(t->src[0]->name, "ffn_gate_exps"), Ld = d.down ? layer_of(t->src[0]->name, "ffn_down_exps") : -1;
            const int L = Lg >= 0 ? Lg : Ld;
            if (debug && seen[t->src[0]->name]++ == 0)
                fprintf(stderr, "mul_mat_id src0 '%s' src1 [%ld %ld %ld] %s ids [%ld %ld] -> layer %d %s\n", t->src[0]->name,
                        (long) t->src[1]->ne[0], (long) t->src[1]->ne[1], (long) t->src[1]->ne[2], ggml_type_name(t->src[1]->type),
                        (long) t->src[2]->ne[0], (long) t->src[2]->ne[1], L, L >= d.lo && L <= d.hi ? "DUMP" : "skip");
            return L >= d.lo && L <= d.hi;
        }
        if (!strncmp(t->name, "ffn_moe_weights", 15)) {   // the routing weights: raw, after normalization, after scaling
            if (!d.moe) return false;
            int L = -1; char tail[32] = {0};
            if (sscanf(t->name, "ffn_moe_weights-%d%31s", &L, tail) < 1 || tail[0]) {
                if (sscanf(t->name, "ffn_moe_weights_norm-%d%31s", &L, tail) < 1 || tail[0]) {
                    if (sscanf(t->name, "ffn_moe_weights_scaled-%d%31s", &L, tail) < 1 || tail[0]) return false;
                }
            }
            if (L < d.lo || L > d.hi) return false;
            if (d.wname[L].find(t->name) == std::string::npos) d.wname[L] += std::string(t->name) + " ";
            return true;
        }
        return false;
    }
    if (t->op == GGML_OP_MUL_MAT) {   // a dense weight's input (the ask phase checked the name)
        std::string sn;
        const int L = dense_of(t->src[0]->name, sn);
        std::string wname = "blk." + std::to_string(L) + "." + sn + ".weight";
        const ggml_tensor * src1 = t->src[1];
        auto it = d.dense_seen.find(src1);
        if (it != d.dense_seen.end()) {   // another weight on the same input: share its file
            if (!d.dense_weight.count(wname)) d.dense_weight[wname] = it->second;
            return true;
        }
        std::string fname;
        auto wit = d.dense_weight.find(wname);
        fname = wit != d.dense_weight.end() ? wit->second : "blk." + std::to_string(L) + ".din." + sn + ".bf16";
        d.dense_weight[wname] = fname;
        d.dense_seen[src1] = fname;
        const int64_t ne0 = src1->ne[0], ne1 = src1->ne[1], ne2 = src1->ne[2], ne3 = src1->ne[3];
        const bool batched = t->src[0]->ne[2] > 1 && ne2 == t->src[0]->ne[2];   // grouped weight [ne0, rank, groups]
        if (src1->nb[0] != ggml_type_size(src1->type) || ggml_blck_size(src1->type) != 1) {
            fprintf(stderr, "%s: input '%s' is not row-contiguous (%s): not dumped\n", wname.c_str(), src1->name, ggml_type_name(src1->type));
            return true;
        }
        get_tensor(src1, d.buf);
        const auto * tr = ggml_get_type_traits(src1->type);
        std::vector<float> row(ne0);
        std::vector<uint16_t> out;
        out.reserve((size_t) ne0 * ne1 * ne2 * ne3);
        auto put = [&](int64_t i1, int64_t i2, int64_t i3) {
            const char * p = d.buf.data() + i1 * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3];
            if (src1->type == GGML_TYPE_F32) memcpy(row.data(), p, ne0 * sizeof(float));
            else if (tr && tr->to_float) tr->to_float(p, row.data(), ne0);
            else { fprintf(stderr, "%s: input type %s not readable\n", wname.c_str(), ggml_type_name(src1->type)); exit(1); }
            const size_t at = out.size();
            out.resize(at + ne0);
            ggml_fp32_to_bf16_row_ref(row.data(), (ggml_bf16_t *) out.data() + at, ne0);
        };
        if (batched) { for (int64_t i3 = 0; i3 < ne3; ++i3) for (int64_t i1 = 0; i1 < ne1; ++i1) for (int64_t i2 = 0; i2 < ne2; ++i2) put(i1, i2, i3); }
        else         { for (int64_t i3 = 0; i3 < ne3; ++i3) for (int64_t i2 = 0; i2 < ne2; ++i2) for (int64_t i1 = 0; i1 < ne1; ++i1) put(i1, i2, i3); }
        fwrite(out.data(), 2, out.size(), d.file(fname));
        DenseFile & df = d.dense_files[fname];
        df.file = fname; df.ne0 = ne0; df.groups = batched ? ne2 : 1;
        df.rows += batched ? ne1 * ne3 : ne1 * ne2 * ne3;
        return true;
    }
    if (t->op != GGML_OP_MUL_MAT_ID) {   // a weights node: [1, n_ids, n_tokens] or [n_ids, n_tokens] f32
        const char * dash = strrchr(t->name, '-');
        const int L = atoi(dash + 1);
        if (t->type != GGML_TYPE_F32) return true;
        const char * suffix = !strncmp(t->name, "ffn_moe_weights_scaled", 22) ? "scaled" :
                              !strncmp(t->name, "ffn_moe_weights_norm", 20) ? "norm" : "raw";
        get_tensor(t, d.buf);
        const bool lead1 = t->ne[0] == 1;   // [1, n_ids, n_tok] (llama.cpp's weights) or [n_ids, n_tok]
        const int64_t n_ids = lead1 ? t->ne[1] : t->ne[0], n_tok = lead1 ? t->ne[2] : t->ne[1];
        const size_t nb_j = lead1 ? t->nb[1] : t->nb[0], nb_r = lead1 ? t->nb[2] : t->nb[1];
        if (debug && seen[t->name]++ == 0)
            fprintf(stderr, "weights '%s' [%ld %ld %ld %ld] %s\n", t->name, (long) t->ne[0], (long) t->ne[1], (long) t->ne[2],
                    (long) t->ne[3], ggml_type_name(t->type));
        std::vector<float> w(n_ids * n_tok);
        for (int64_t r = 0; r < n_tok; ++r)
            for (int64_t j = 0; j < n_ids; ++j)
                w[r * n_ids + j] = *(const float *) (d.buf.data() + r * nb_r + j * nb_j);
        fwrite(w.data(), 4, w.size(), d.file("blk." + std::to_string(L) + ".w_" + suffix + ".f32"));
        return true;
    }
    const ggml_tensor * src1 = t->src[1];
    const ggml_tensor * ids = t->src[2];
    const int Lg = layer_of(t->src[0]->name, "ffn_gate_exps");
    const int L = Lg >= 0 ? Lg : layer_of(t->src[0]->name, "ffn_down_exps");
    if (src1->type != GGML_TYPE_F32) { fprintf(stderr, "%s: input is %s, not f32\n", t->src[0]->name, ggml_type_name(src1->type)); exit(1); }
    const int64_t ne0 = src1->ne[0], ne1 = src1->ne[1], n_tok = src1->ne[2], n_ids = ids->ne[0];
    get_tensor(src1, d.buf);
    const std::string pre = "blk." + std::to_string(L) + ".";
    if (Lg >= 0) {   // x: the same row for every slot (ne1 == 1), or repeated per slot (take slot 0)
        d.n_embd = (int) ne0; d.n_ids = (int) n_ids;
        d.bf.resize(n_tok * ne0);
        for (int64_t r = 0; r < n_tok; ++r)
            ggml_fp32_to_bf16_row_ref((const float *) (d.buf.data() + r * src1->nb[2]), (ggml_bf16_t *) d.bf.data() + r * ne0, ne0);
        fwrite(d.bf.data(), 2, d.bf.size(), d.file(pre + "x.bf16"));
        std::vector<char> ib;
        get_tensor(ids, ib);
        d.ids.resize(n_tok * n_ids);
        for (int64_t r = 0; r < n_tok; ++r)
            for (int64_t j = 0; j < n_ids; ++j)
                d.ids[r * n_ids + j] = *(const int32_t *) (ib.data() + r * ids->nb[1] + j * ids->nb[0]);
        fwrite(d.ids.data(), 4, d.ids.size(), d.file(pre + "ids.i32"));
        d.rows[L] += n_tok;
    } else {         // a: one row a token and slot
        d.n_ff = (int) ne0;
        if (ne1 != n_ids) { fprintf(stderr, "%s: input has %ld slots, ids %ld\n", t->src[0]->name, (long) ne1, (long) n_ids); exit(1); }
        d.bf.resize(n_tok * n_ids * ne0);
        for (int64_t r = 0; r < n_tok; ++r)
            for (int64_t j = 0; j < n_ids; ++j)
                ggml_fp32_to_bf16_row_ref((const float *) (d.buf.data() + r * src1->nb[2] + j * src1->nb[1]),
                                          (ggml_bf16_t *) d.bf.data() + (r * n_ids + j) * ne0, ne0);
        fwrite(d.bf.data(), 2, d.bf.size(), d.file(pre + "a.bf16"));
    }
    return true;
}

int main(int argc, char ** argv) {
    std::vector<const char *> args;
    int nt = 16, n_ctx = 2048, ub = 8192, max_chunks = 0, dense_chunks = 0;
    std::string dense_re;
    bool gpu = false;
    Dump d;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--threads=", 10)) nt = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--ctx=", 6)) n_ctx = atoi(argv[i] + 6);
        else if (!strncmp(argv[i], "--ub=", 5)) ub = atoi(argv[i] + 5);
        else if (!strncmp(argv[i], "--chunks=", 9)) max_chunks = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--layers=", 9)) { d.lo = atoi(argv[i] + 9); const char * p = strchr(argv[i] + 9, '-'); d.hi = p ? atoi(p + 1) : d.lo; }
        else if (!strcmp(argv[i], "--dump-down")) d.down = true;
        else if (!strcmp(argv[i], "--dense")) { d.dense = true; dense_re = DENSE_DEFAULT; }
        else if (!strncmp(argv[i], "--dense=", 8)) { d.dense = true; dense_re = argv[i] + 8; }
        else if (!strncmp(argv[i], "--dense-chunks=", 15)) dense_chunks = atoi(argv[i] + 15);
        else if (!strcmp(argv[i], "--no-moe")) d.moe = false;
        else if (!strcmp(argv[i], "--gpu")) gpu = true;
        else args.push_back(argv[i]);
    }
    if (args.size() != 3) {
        fprintf(stderr, "usage: hessian_dump <model.gguf> <calib.txt> <out dir> [--gpu] [--threads=N] [--ctx=2048] [--ub=8192] "
                        "[--chunks=N] [--dump-down] [--layers=a-b] [--dense[=<regex>]] [--dense-chunks=N] [--no-moe]\n");
        return 2;
    }
    d.dir = args[2];
    if (d.dense) d.dense_re = std::regex(dense_re);
    if (!d.moe && !d.dense) { fprintf(stderr, "--no-moe without --dense: nothing to dump\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_model_tensor_buft_override ov[2] = {{"exps", ggml_backend_dev_buffer_type(cpu)}, {nullptr, nullptr}};
    mp.n_gpu_layers = gpu ? 999 : 0;
    if (gpu) mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(args[0], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", args[0]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ifstream in(args[1]);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), true, false);
    if (n < 0) { tokens.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), true, false); }
    tokens.resize(n);
    int n_chunk = n / n_ctx;
    if (max_chunks > 0 && n_chunk > max_chunks) n_chunk = max_chunks;
    const int n_seq = ub >= n_ctx ? ub / n_ctx : 1;
    if (ub < n_ctx) ub = n_ctx;
    printf("%d tokens, %d chunks of %d, %d chunks a batch (bos: %s)\n", n, n_chunk, n_ctx, n_seq,
           llama_vocab_get_add_bos(vocab) ? "yes" : "no");
    fflush(stdout);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_seq * n_ctx; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = n_seq;
    cp.n_threads = nt; cp.n_threads_batch = nt;
    cp.cb_eval = cb; cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot make the context\n"); return 1; }
    llama_batch b = llama_batch_init(ub, 0, n_seq);
    for (int c0 = 0; c0 < n_chunk; c0 += n_seq) {
        const int ns = std::min(n_seq, n_chunk - c0);
        llama_memory_clear(llama_get_memory(ctx), true);
        d.dense_active = d.dense && (dense_chunks <= 0 || c0 < dense_chunks);
        d.dense_seen.clear();
        if (!d.moe && !d.dense_active) break;   // (a dense-only pass past its chunks: done)
        b.n_tokens = 0;
        for (int s = 0; s < ns; ++s)
            for (int k = 0; k < n_ctx; ++k) {
                const int i = b.n_tokens++;
                b.token[i] = tokens[(size_t) (c0 + s) * n_ctx + k]; b.pos[i] = k; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = 0;
            }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed at chunk %d\n", c0); fflush(stderr); return 1; }
        printf("chunks %d-%d of %d done\n", c0, c0 + ns - 1, n_chunk);
        fflush(stdout);
    }
    for (auto & kv : d.files) fclose(kv.second);
    FILE * f = fopen((d.dir + "/info.txt").c_str(), "w");
    int64_t rows = d.rows.empty() ? 0 : d.rows.begin()->second;
    fprintf(f, "model %s\ntext %s\nctx %d\nchunks %d\ntokens %lld\nn_embd %d\nn_ff %d\nn_expert_used %d\nlayers", args[0], args[1],
            n_ctx, n_chunk, (long long) rows, d.n_embd, d.n_ff, d.n_ids);
    for (auto & kv : d.rows) fprintf(f, " %d", kv.first);
    fprintf(f, "\nweights");
    for (auto & kv : d.wname) fprintf(f, " %d:%s", kv.first, kv.second.c_str());
    fprintf(f, "\n");
    fclose(f);
    if (d.dense) {
        FILE * g = fopen((d.dir + "/dense.txt").c_str(), "w");
        fprintf(g, "# weight file ne0 groups rows (tools/hessian_dump --dense; regex %s, dense chunks %d)\n", dense_re.c_str(), dense_chunks);
        for (auto & kv : d.dense_weight) {
            const DenseFile & df = d.dense_files[kv.second];
            fprintf(g, "%s %s %lld %lld %lld\n", kv.first.c_str(), kv.second.c_str(), (long long) df.ne0, (long long) df.groups, (long long) df.rows);
        }
        fclose(g);
        printf("dense: %zu weights, %zu input files\n", d.dense_weight.size(), d.dense_files.size());
    }
    printf("wrote %lld rows a layer for %zu layers to %s\n", (long long) rows, d.rows.size(), d.dir.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
