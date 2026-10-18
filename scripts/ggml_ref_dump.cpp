// Differential dequant reference: reads a GGUF tensor row, calls ggml's
// dequantize_row_*, prints f32 values as binary to stdout.
//
// Build (a checkout of llama.cpp is required; only ggml-quants.c is used):
//
//   gcc -O1 -mf16c -mavx2 -DGGML_COMMON_IMPL_CPP \
//       -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//       -c llama.cpp/ggml/src/ggml-quants.c -o ggml-quants.o
//   gcc -O1 -c -x c - <<'EOF' -o stub.o   # stubs for symbols pulled in but unused
//   void ggml_abort(const char*f,int l,const char*m,...){(void)f;(void)l;(void)m;__builtin_trap();}
//   size_t ggml_row_size(int t,long long n){(void)t;(void)n;return 0;}
//   size_t ggml_type_size(int t){(void)t;return 0;}
//   const char*ggml_type_name(int t){(void)t;return "?";}
//   EOF
//   g++ -O1 -mf16c -mavx2 -DGGML_COMMON_IMPL_CPP \
//       -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//       scripts/ggml_ref_dump.cpp ggml-quants.o stub.o -o ggml_ref_dump
//
// Usage: ggml_ref_dump <model.gguf> <tensor.name> <row> > row.f32
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "ggml-quants.h"

static uint64_t read_u64(const uint8_t*& p) { uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
static uint32_t read_u32(const uint8_t*& p) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <gguf> <tensor> <row>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) return 1;
    fclose(f);
    const uint8_t* p = buf.data();
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); return 1; }
    p += 4;
    uint32_t version = read_u32(p);
    uint64_t n_tensors = read_u64(p), n_kv = read_u64(p);
    (void)version;
    auto skip_val = [&](uint32_t t) {
        switch (t) {
            case 0: case 1: case 7: p += 1; break;
            case 2: case 3: p += 2; break;
            case 4: case 5: case 6: p += 4; break;
            case 10: case 11: case 12: p += 8; break;
            case 8: { uint64_t n = read_u64(p); p += n; break; }
            case 9: {
                uint32_t et = read_u32(p); uint64_t n = read_u64(p);
                if (et == 8) { for (uint64_t i = 0; i < n; i++) { uint64_t m = read_u64(p); p += m; } }
                else { size_t isz = et==0||et==1||et==7?1:et==2||et==3?2:et==4||et==5||et==6?4:8; p += isz*n; }
                break;
            }
            default: fprintf(stderr, "bad vtype %u\n", t); exit(1);
        }
    };
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = read_u64(p); p += klen;
        uint32_t t = read_u32(p); skip_val(t);
    }
    struct Info { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
    std::vector<Info> infos;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen = read_u64(p); std::string name((const char*)p, nlen); p += nlen;
        uint32_t nd = read_u32(p);
        Info inf; inf.name = name;
        for (uint32_t d = 0; d < nd; d++) inf.dims.push_back(read_u64(p));
        inf.type = read_u32(p); inf.off = read_u64(p);
        infos.push_back(inf);
    }
    uint64_t data_start = (uint64_t)(p - buf.data());
    data_start = (data_start + 31) / 32 * 32;
    // second pass on original buffer with computed data_start
    for (auto& inf : infos) {
        if (inf.name != argv[2]) continue;
        uint64_t row = strtoull(argv[3], nullptr, 10);
        uint64_t cols = 1; for (size_t d = 1; d < inf.dims.size(); d++) cols *= inf.dims[d]; // row-major logical
        // file order: dims fastest first; logical shape reversed
        uint64_t logical_cols = inf.dims[0];
        for (size_t d = 1; d + 1 < inf.dims.size(); d++) logical_cols *= 1; // rows product unused
        const uint8_t* base = buf.data() + data_start + inf.off;
        std::vector<float> y(inf.dims[0]);
        const uint8_t* rowp = base + row * 0; // placeholder
        (void)rowp; (void)y; (void)cols; (void)logical_cols;
        size_t row_bytes = 0;
        switch (inf.type) {
            case 8: row_bytes = inf.dims[0] / 32 * 34; break;
            case 10: row_bytes = inf.dims[0] / 256 * 84; break;
            case 12: row_bytes = inf.dims[0] / 256 * 144; break;
            case 14: row_bytes = inf.dims[0] / 256 * 210; break;
            case 21: row_bytes = inf.dims[0] / 256 * 110; break;
            case 22: row_bytes = inf.dims[0] / 256 * 82; break;
            case 23: row_bytes = inf.dims[0] / 256 * 136; break;
            case 0: row_bytes = inf.dims[0] * 4; break;
            case 30: row_bytes = inf.dims[0] * 2; break;
            default: fprintf(stderr, "type %u unsupported\n", inf.type); return 1;
        }
        const uint8_t* rp = base + row * row_bytes;
        std::vector<float> out(inf.dims[0]);
        switch (inf.type) {
            case 8: dequantize_row_q8_0((const block_q8_0*)rp, out.data(), inf.dims[0]); break;
            case 10: dequantize_row_q2_K((const block_q2_K*)rp, out.data(), inf.dims[0]); break;
            case 12: dequantize_row_q4_K((const block_q4_K*)rp, out.data(), inf.dims[0]); break;
            case 14: dequantize_row_q6_K((const block_q6_K*)rp, out.data(), inf.dims[0]); break;
            case 21: dequantize_row_iq3_s((const block_iq3_s*)rp, out.data(), inf.dims[0]); break;
            case 22: dequantize_row_iq2_s((const block_iq2_s*)rp, out.data(), inf.dims[0]); break;
            case 23: dequantize_row_iq4_xs((const block_iq4_xs*)rp, out.data(), inf.dims[0]); break;
            case 0: { for (uint64_t i = 0; i < inf.dims[0]; i++) { float v; memcpy(&v, rp + 4*i, 4); out[i] = v; } } break;
            case 30: { for (uint64_t i = 0; i < inf.dims[0]; i++) { uint32_t h; memcpy(&h, rp + 2*i, 2); uint32_t bits = (h & 0x8000) << 16 | ((h >> 7) & 1 ? 0x7F800000|(h&0x7F)<<16 : (h&0x7F)<<16); uint32_t e = (h >> 7) & 0xFF; bits = (h & 0x8000) << 16; if (e == 0) { bits |= (h & 0x7F) << 16; } else { bits |= (e + 112) << 23 | (h & 0x7F) << 16; } float v; memcpy(&v, &bits, 4); out[i] = v; } } break;
        }
        fwrite(out.data(), sizeof(float), out.size(), stdout);
        return 0;
    }
    fprintf(stderr, "tensor not found\n");
    return 1;
}
