// tools/teacher.cpp (trench-gguf) - the quality check of a quantized GGUF, with llama.cpp as the library.
//
//   teacher ref <model.gguf> <out dir> <seq.ids> [<seq.ids> ...] [--k=32] [--gpu] [--threads=N]
//       Runs the model (the original, or the biggest quant you have) over each token-id file and writes
//       <out dir>/<seq name>.topk: the top-k log-probs of every position (int32 k, then per position k int32 ids and
//       k float32 log-probs; position i predicts token i+1). Abyss reads the same files (STRATA_GLM_SCORE_TOPK).
//   teacher kl <model.gguf> <ref dir> <seq.ids> [<seq.ids> ...] [--gpu] [--threads=N]
//       Runs a quantized model over the same files and prints, against the reference: KL (the reference's top-k plus
//       one bucket for the rest of the vocabulary, the same formula as Abyss's score path), the largest KL of one
//       position, the 99th percentile, the share of positions with the same top token, and nll.
//
//   --gpu: dense tensors on the GPU (-ngl 99), routed experts stay memory-mapped on the host (exps -> CPU buffer) and
//   llama.cpp copies them to the GPU for each batch (op offload), as llama-imatrix --gpu in tools/imatrix.py does.
//   Needs a ROCm or CUDA build of llama.cpp; without --gpu a CPU build is enough. A whole file is one batch.
//   Token-id files: comma-separated ids, no BOS is added (write it into the file if the model wants one).
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static std::vector<llama_token> read_ids(const char * path) {
    std::ifstream in(path);
    std::string s((std::istreambuf_iterator<char>(in)), {}), t;
    std::vector<llama_token> ids;
    std::stringstream ss(s);
    while (std::getline(ss, t, ',')) if (t.find_first_not_of(" \r\n\t") != std::string::npos) ids.push_back(atoi(t.c_str()));
    return ids;
}

static std::string stem(const char * path) {
    std::string p(path);
    const size_t a = p.find_last_of('/');
    if (a != std::string::npos) p = p.substr(a + 1);
    const size_t b = p.rfind('.');
    return b == std::string::npos ? p : p.substr(0, b);
}

int main(int argc, char ** argv) {
    std::vector<const char *> args;
    int K = 32, nt = 16;
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--k=", 4)) K = atoi(argv[i] + 4);
        else if (!strncmp(argv[i], "--threads=", 10)) nt = atoi(argv[i] + 10);
        else if (!strcmp(argv[i], "--gpu")) gpu = true;
        else args.push_back(argv[i]);
    }
    const bool ref = args.size() >= 4 && !strcmp(args[0], "ref");
    const bool kl = args.size() >= 4 && !strcmp(args[0], "kl");
    if (!ref && !kl) {
        fprintf(stderr, "usage: teacher ref <model.gguf> <out dir> <seq.ids>... [--k=32] [--gpu] [--threads=N]\n"
                        "       teacher kl  <model.gguf> <ref dir> <seq.ids>... [--gpu] [--threads=N]\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_model_tensor_buft_override ov[2] = {{"exps", ggml_backend_dev_buffer_type(cpu)}, {nullptr, nullptr}};
    mp.n_gpu_layers = gpu ? 999 : 0;
    if (gpu) mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(args[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", args[1]); return 1; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::string dir = args[2];

    double kl_sum = 0, kl_max = 0, nll_sum = 0;
    size_t npos = 0, same = 0;
    std::vector<double> kls;
    std::vector<int> ord(nv);
    std::vector<float> ls(nv);
    for (size_t a = 3; a < args.size(); ++a) {
        const std::vector<llama_token> ids = read_ids(args[a]);
        const int n = (int) ids.size();
        if (n < 2) { fprintf(stderr, "%s: fewer than 2 ids\n", args[a]); return 1; }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n + 16; cp.n_batch = n; cp.n_ubatch = n; cp.n_threads = nt; cp.n_threads_batch = nt;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "cannot make a context of %d tokens\n", n); return 1; }
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        const std::string tk = dir + "/" + stem(args[a]) + ".topk";
        FILE * f = fopen(tk.c_str(), ref ? "wb" : "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", tk.c_str()); return 1; }
        int32_t k32 = K;
        if (ref) fwrite(&k32, 4, 1, f);
        else if (fread(&k32, 4, 1, f) != 1) { fprintf(stderr, "%s: empty\n", tk.c_str()); return 1; }
        std::vector<int32_t> ti(k32);
        std::vector<float> tv(k32);
        double nll = 0;
        for (int i = 0; i + 1 < n; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            const float mx = *std::max_element(lg, lg + nv);
            double z = 0;
            for (int j = 0; j < nv; ++j) z += std::exp((double) lg[j] - mx);
            const double lse = mx + std::log(z);
            for (int j = 0; j < nv; ++j) ls[j] = (float) (lg[j] - lse);
            nll -= ls[ids[i + 1]];
            if (ref) {
                std::iota(ord.begin(), ord.end(), 0);
                std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) { return lg[x] > lg[y]; });
                for (int j = 0; j < K; ++j) { ti[j] = ord[j]; tv[j] = ls[ord[j]]; }
                fwrite(ti.data(), 4, K, f); fwrite(tv.data(), 4, K, f);
                continue;
            }
            if (fread(ti.data(), 4, k32, f) != (size_t) k32 || fread(tv.data(), 4, k32, f) != (size_t) k32) {
                fprintf(stderr, "%s: shorter than %s\n", tk.c_str(), args[a]); return 1;
            }
            double k = 0, pr = 0, pt = 0;
            for (int j = 0; j < k32; ++j) {
                const double p = std::exp((double) tv[j]);
                k += p * ((double) tv[j] - (double) ls[ti[j]]);
                pr += p;
                pt += std::exp((double) ls[ti[j]]);
            }
            const double rr = std::max(1.0 - pr, 1e-12), rt = std::max(1.0 - pt, 1e-12);
            if (pr < 1.0) k += rr * (std::log(rr) - std::log(rt));
            int am = 0;
            for (int j = 1; j < nv; ++j) if (lg[j] > lg[am]) am = j;
            kl_sum += k; kl_max = std::max(kl_max, k); kls.push_back(k);
            same += ti[0] == am;
        }
        fclose(f);
        nll_sum += nll; npos += n - 1;
        printf("%s: %d positions, nll %.5f%s\n", args[a], n - 1, nll / (n - 1), ref ? (" -> " + tk).c_str() : "");
        fflush(stdout);
        llama_batch_free(b);
        llama_free(ctx);
    }
    if (kl && npos) {
        std::sort(kls.begin(), kls.end());
        printf("all: %zu positions, nll %.5f | KL %.6f (99th pct %.4f, max %.4f), same top token %.2f%%\n", npos,
               nll_sum / npos, kl_sum / npos, kls[(size_t) (0.99 * (kls.size() - 1))], kl_max, 100.0 * same / npos);
    }
    llama_model_free(model);
    return 0;
}
