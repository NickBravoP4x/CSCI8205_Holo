#include "cpu_baseline.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

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
            float noise = sinf(i * 12.9898f + j * 78.233f) * 43758.5453f;
            noise = noise - floorf(noise);
            val += 0.1f * (2.0f * noise - 1.0f);
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

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --size N        Image size NxN (default: 256)\n");
    printf("  --depth N       Number of depth planes (default: 3)\n");
    printf("  --threads N     Number of threads (default: 1)\n");
    printf("  --warmup N      Warmup iterations (default: 5)\n");
    printf("  --iterations N  Benchmark iterations (default: 20)\n");
    printf("  --sweep         Run thread scaling sweep (1,2,4,8,16,32,64)\n");
}

int main(int argc, char* argv[]) {
    int size = 256;
    int depth = 3;
    int threads = 1;
    int warmup = 5;
    int iterations = 20;
    bool sweep = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--size" && i + 1 < argc)        size = atoi(argv[++i]);
        else if (arg == "--depth" && i + 1 < argc)   depth = atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc)  threads = atoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc)   warmup = atoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (arg == "--sweep") sweep = true;
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    printf("CPU (OpenMP + FFTW) Holographic Reconstruction Benchmark\n");
    printf("=========================================================\n");
    printf("Image size: %dx%d\n", size, size);
    printf("Depth planes: %d\n", depth);
    printf("Warmup: %d, Iterations: %d\n\n", warmup, iterations);

    std::vector<float> hologram(size * size);
    generate_test_hologram(hologram.data(), size, size);
    std::vector<float> output(size * size * depth);

    auto run_benchmark = [&](int num_threads) {
        CPUHolographicReconstructor::Params params;
        params.num_planes = depth;
        params.num_threads = num_threads;

        CPUHolographicReconstructor recon(params);

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            recon.reconstruct_intensity(hologram.data(), size, size, output.data());
        }

        // Benchmark
        std::vector<double> latencies(iterations);
        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            recon.reconstruct_intensity(hologram.data(), size, size, output.data());
            auto end = std::chrono::high_resolution_clock::now();
            latencies[i] = std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::sort(latencies.begin(), latencies.end());
        double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / iterations;
        double median = latencies[iterations / 2];
        double p95 = latencies[(int)(iterations * 0.95)];

        printf("Threads: %2d | Mean: %8.3f ms | Median: %8.3f ms | P95: %8.3f ms | Throughput: %7.1f ops/s\n",
               num_threads, mean, median, p95, 1000.0 / mean);

        return mean;
    };

    if (sweep) {
        printf("Thread Scaling Sweep:\n");
        printf("%-10s %-12s %-12s %-12s %-12s %-12s\n",
               "Threads", "Mean (ms)", "Median (ms)", "P95 (ms)", "Throughput", "Speedup");

        int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
        double base_latency = 0;
        for (int tc : thread_counts) {
            double mean = run_benchmark(tc);
            if (tc == 1) base_latency = mean;
            double speedup = base_latency / mean;
            printf("  Speedup: %.2fx\n", speedup);
        }
    } else {
        run_benchmark(threads);
    }

    // Output sanity check
    float out_min = *std::min_element(output.begin(), output.end());
    float out_max = *std::max_element(output.begin(), output.end());
    float out_sum = std::accumulate(output.begin(), output.end(), 0.0f);
    float out_mean = out_sum / output.size();
    printf("\nOutput sanity: Min=%.6f Max=%.6f Mean=%.6f\n", out_min, out_max, out_mean);

    return 0;
}
