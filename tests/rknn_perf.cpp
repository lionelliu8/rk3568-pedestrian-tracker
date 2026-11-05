// Per-layer NPU/CPU timing for any RKNN model, using a zero input.
// Usage: rknn_perf MODEL [runs]
// Prints the runtime table (which ops run on CPU vs NPU) and the mean rknn_run time.
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "stopwatch.h"

namespace {

void check(int ret, const char* what) {
    if (ret != RKNN_SUCC) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(ret));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " MODEL [runs]\n";
        return 2;
    }
    try {
        const int runs = argc > 2 ? std::stoi(argv[2]) : 10;
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        std::vector<char> blob(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(blob.data(), static_cast<std::streamsize>(blob.size()));

        rknn_context ctx = 0;
        check(rknn_init(&ctx, blob.data(), static_cast<uint32_t>(blob.size()), RKNN_FLAG_COLLECT_PERF_MASK, nullptr),
              "rknn_init");
        rknn_tensor_attr in{};
        in.index = 0;
        check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in)), "rknn_query(input)");
        std::vector<uint8_t> zeros(in.n_elems, 0);
        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<uint32_t>(zeros.size());
        input.buf = zeros.data();

        double total = 0;
        for (int i = 0; i < runs; ++i) {
            check(rknn_inputs_set(ctx, 1, &input), "rknn_inputs_set");
            Stopwatch watch;
            check(rknn_run(ctx, nullptr), "rknn_run");
            if (i > 0) total += watch.elapsed();  // skip warm-up
        }
        rknn_perf_detail detail{};
        if (rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)) == RKNN_SUCC && detail.perf_data) {
            std::cout << std::string(detail.perf_data, detail.data_len) << "\n";
        }
        std::cout << "mean rknn_run ms (with perf collection): " << total / std::max(1, runs - 1) << "\n";
        rknn_destroy(ctx);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
