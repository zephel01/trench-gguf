// gpuq/gpuq_tables.c (trench-gguf) - writes the i-quant search tables (grid, map, neighbours) that ggml builds in
// iq2xs_init_impl to a file for the GPU quantizer, so the GPU searches exactly the tables the CPU searches.
//   cc -O2 -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src gpuq_tables.c -L<build>/bin -lggml-base -lm -o gpuq_tables
//   ./gpuq_tables gpuq_tables.bin
// File: per type (IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S): int32 ggml type, grid_size, map_size, neigh_len, then
// uint64 grid[grid_size], int32 map[map_size], uint16 neighbours[neigh_len].
#include "ggml-quants.c"

static void dump(FILE * f, enum ggml_type type) {
    iq2xs_init_impl(type);
    const int g = iq2_data_index(type);
    const int grid_size = iq2_grid_size(type);
    const int map_size = 43692;
    const int * map = iq2_data[g].map;
    int neigh_len = 0;
    for (int u = 0; u < map_size; ++u) {
        if (map[u] < 0) {
            const int off = -map[u] - 1;
            const int end = off + 1 + iq2_data[g].neighbours[off];
            if (end > neigh_len) neigh_len = end;
        }
    }
    const int32_t hdr[4] = {(int32_t) type, grid_size, map_size, neigh_len};
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(iq2_data[g].grid, sizeof(uint64_t), grid_size, f);
    fwrite(map, sizeof(int32_t), map_size, f);
    fwrite(iq2_data[g].neighbours, sizeof(uint16_t), neigh_len, f);
    fprintf(stderr, "type %d: grid %d, map %d, neighbours %d\n", (int) type, grid_size, map_size, neigh_len);
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out.bin\n", argv[0]); return 2; }
    FILE * f = fopen(argv[1], "wb");
    if (!f) { perror(argv[1]); return 1; }
    dump(f, GGML_TYPE_IQ2_XXS);
    dump(f, GGML_TYPE_IQ2_XS);
    dump(f, GGML_TYPE_IQ1_S);
    dump(f, GGML_TYPE_IQ2_S);
    fclose(f);
    return 0;
}
