#include "holographic_reconstruction.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

// Generate synthetic hologram (matches Python's generate_test_hologram)
static void generate_test_hologram(float* data, int H, int W) {
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float x = 2.0f * j / (float)W - 1.0f;
            float y = 2.0f * i / (float)H - 1.0f;
            float val = 0.0f;
            float scales[] = {2.0f, 5.0f, 10.0f, 20.0f};
            for (float s : scales) {
                val += sinf(s * M_PI * (x * x + y * y));
                val += cosf(s * M_PI * x * y);
            }
            // Simple pseudo-random noise (deterministic)
            float noise = sinf(i * 12.9898f + j * 78.233f) * 43758.5453f;
            noise = noise - floorf(noise);
            val += 0.1f * (2.0f * noise - 1.0f);
            data[i * W + j] = val;
        }
    }
    // Normalize to [0, 255]
    float mn = *std::min_element(data, data + H * W);
    float mx = *std::max_element(data, data + H * W);
    float range = mx - mn + 1e-8f;
    for (int i = 0; i < H * W; ++i) {
        data[i] = 255.0f * (data[i] - mn) / range;
        data[i] = floorf(data[i]);  // byte-quantize like Python
    }
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --size N        Image size NxN (default: 256)\n");
    printf("  --depth N       Number of depth planes (default: 3)\n");
    printf("  --warmup N      Warmup iterations (default: 5)\n");
    printf("  --iterations N  Benchmark iterations (default: 20)\n");
    printf("  --gpu N         GPU device ID (default: 0)\n");
}

int main(int argc, char* argv[]) {
    int size = 256;
    int depth = 3;
    int warmup = 5;
    int iterations = 20;
    int gpu_id = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--size" && i + 1 < argc)       size = atoi(argv[++i]);
        else if (arg == "--depth" && i + 1 < argc)  depth = atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)  warmup = atoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc)    gpu_id = atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    printf("CUDA Holographic Reconstruction Benchmark\n");
    printf("==========================================\n");
    printf("Image size: %dx%d\n", size, size);
    printf("Depth planes: %d\n", depth);
    printf("GPU device: %d\n", gpu_id);
    printf("Warmup: %d, Iterations: %d\n\n", warmup, iterations);

    // Generate test data
    std::vector<float> hologram(size * size);
    generate_test_hologram(hologram.data(), size, size);

    // Setup reconstructor
    CUDAHolographicReconstructor::Params params;
    params.num_planes = depth;
    params.gpu_id = gpu_id;

    CUDAHolographicReconstructor recon(params);
    std::vector<float> output(size * size * depth);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < warmup; ++i) {
        recon.reconstruct_intensity(hologram.data(), size, size, output.data());
    }
    cudaDeviceSynchronize();

    // Benchmark
    printf("Benchmarking...\n");
    std::vector<double> latencies(iterations);

    for (int i = 0; i < iterations; ++i) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        recon.reconstruct_intensity(hologram.data(), size, size, output.data());
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        latencies[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Statistics
    std::sort(latencies.begin(), latencies.end());
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / iterations;
    double median = latencies[iterations / 2];
    double p95 = latencies[(int)(iterations * 0.95)];
    double p99 = latencies[(int)(iterations * 0.99)];
    double min_lat = latencies.front();
    double max_lat = latencies.back();

    printf("\nResults:\n");
    printf("  Mean:   %.3f ms\n", mean);
    printf("  Median: %.3f ms\n", median);
    printf("  P95:    %.3f ms\n", p95);
    printf("  P99:    %.3f ms\n", p99);
    printf("  Min:    %.3f ms\n", min_lat);
    printf("  Max:    %.3f ms\n", max_lat);
    printf("  Throughput: %.1f ops/s\n", 1000.0 / mean);

    // Verify output sanity
    float out_min = *std::min_element(output.begin(), output.end());
    float out_max = *std::max_element(output.begin(), output.end());
    float out_sum = std::accumulate(output.begin(), output.end(), 0.0f);
    float out_mean = out_sum / output.size();
    printf("\nOutput sanity check:\n");
    printf("  Min: %.6f, Max: %.6f, Mean: %.6f\n", out_min, out_max, out_mean);
    printf("  Expected range: [0.0, 1.0]\n");

    return 0;
}
