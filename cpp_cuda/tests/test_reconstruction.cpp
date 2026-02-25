#include "holographic_reconstruction.h"
#include "cpu_baseline.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

static void generate_test_hologram(float* data, int H, int W) {
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float x = 2.0f * j / (float)W - 1.0f;
            float y = 2.0f * i / (float)H - 1.0f;
            float val = 0.0f;
            float scales[] = {2.0f, 5.0f, 10.0f, 20.0f};
            for (float s : scales) {
                val += sinf(s * (float)M_PI * (x * x + y * y));
                val += cosf(s * (float)M_PI * x * y);
            }
            data[i * W + j] = val;
        }
    }
    float mn = *std::min_element(data, data + H * W);
    float mx = *std::max_element(data, data + H * W);
    float range = mx - mn + 1e-8f;
    for (int i = 0; i < H * W; ++i) {
        data[i] = 255.0f * (data[i] - mn) / range;
        data[i] = floorf(data[i]);
    }
}

static bool test_output_range(const float* output, int size, const char* name) {
    float mn = *std::min_element(output, output + size);
    float mx = *std::max_element(output, output + size);
    bool pass = (mn >= -0.001f) && (mx <= 1.001f);
    printf("  [%s] Range: [%.6f, %.6f] -> %s\n", name, mn, mx, pass ? "PASS" : "FAIL");
    return pass;
}

static float compute_relative_error(const float* a, const float* b, int n) {
    double max_err = 0;
    double max_val = 0;
    for (int i = 0; i < n; ++i) {
        max_err = std::max(max_err, (double)std::abs(a[i] - b[i]));
        max_val = std::max(max_val, (double)std::max(std::abs(a[i]), std::abs(b[i])));
    }
    return (float)(max_err / (max_val + 1e-8));
}

int main() {
    int passed = 0, failed = 0;

    // Test configurations
    struct TestConfig {
        int size;
        int depth;
    };
    TestConfig configs[] = {
        {128, 1}, {128, 3}, {256, 3}, {256, 5}, {512, 3}
    };

    printf("=== Holographic Reconstruction Tests ===\n\n");

    for (auto& cfg : configs) {
        printf("Config: %dx%d, %d planes\n", cfg.size, cfg.size, cfg.depth);

        std::vector<float> hologram(cfg.size * cfg.size);
        generate_test_hologram(hologram.data(), cfg.size, cfg.size);

        int out_size = cfg.size * cfg.size * cfg.depth;

        // CUDA test
        std::vector<float> cuda_output(out_size);
        {
            CUDAHolographicReconstructor::Params p;
            p.num_planes = cfg.depth;
            CUDAHolographicReconstructor recon(p);
            recon.reconstruct_intensity(hologram.data(), cfg.size, cfg.size,
                                         cuda_output.data());
        }
        if (test_output_range(cuda_output.data(), out_size, "CUDA")) passed++; else failed++;

        // CPU test (4 threads)
        std::vector<float> cpu_output(out_size);
        {
            CPUHolographicReconstructor::Params p;
            p.num_planes = cfg.depth;
            p.num_threads = 4;
            CPUHolographicReconstructor recon(p);
            recon.reconstruct_intensity(hologram.data(), cfg.size, cfg.size,
                                         cpu_output.data());
        }
        if (test_output_range(cpu_output.data(), out_size, "CPU ")) passed++; else failed++;

        // Cross-validation: CUDA vs CPU
        float rel_err = compute_relative_error(cuda_output.data(), cpu_output.data(), out_size);
        bool cross_pass = rel_err < 1e-2f;  // Relaxed for standalone test (tighter in validate_cuda.py)
        printf("  [CROSS] CUDA vs CPU relative error: %.6e -> %s\n",
               rel_err, cross_pass ? "PASS" : "FAIL");
        if (cross_pass) passed++; else failed++;

        printf("\n");
    }

    printf("=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
