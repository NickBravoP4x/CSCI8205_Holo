#include "pipeline.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <stdexcept>

namespace fs = std::filesystem;

// ─── Statistics helper ───────────────────────────────────────────────────────
struct Stats {
    double mean_ms = 0, median_ms = 0, std_ms = 0;
    double min_ms = 0, max_ms = 0;
    double p95_ms = 0, p99_ms = 0;
    double throughput_fps = 0;
};

static Stats compute_stats(std::vector<double>& times) {
    Stats s;
    if (times.empty()) return s;

    std::sort(times.begin(), times.end());
    int n = (int)times.size();

    s.min_ms = times.front();
    s.max_ms = times.back();
    s.median_ms = (n % 2 == 0)
        ? (times[n/2-1] + times[n/2]) / 2.0
        : times[n/2];

    double sum = 0;
    for (auto t : times) sum += t;
    s.mean_ms = sum / n;

    double var = 0;
    for (auto t : times) var += (t - s.mean_ms) * (t - s.mean_ms);
    s.std_ms = std::sqrt(var / n);

    s.p95_ms = times[std::min((int)(n * 0.95), n - 1)];
    s.p99_ms = times[std::min((int)(n * 0.99), n - 1)];
    s.throughput_fps = 1000.0 / s.mean_ms;

    return s;
}

// ─── JSON escape helper ─────────────────────────────────────────────────────
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

static std::string stats_to_json(const Stats& s, int indent = 6) {
    std::string pad(indent, ' ');
    std::ostringstream os;
    os.precision(4);
    os << std::fixed;
    os << "{\n";
    os << pad << "  \"mean_ms\": " << s.mean_ms << ",\n";
    os << pad << "  \"median_ms\": " << s.median_ms << ",\n";
    os << pad << "  \"std_ms\": " << s.std_ms << ",\n";
    os << pad << "  \"min_ms\": " << s.min_ms << ",\n";
    os << pad << "  \"max_ms\": " << s.max_ms << ",\n";
    os << pad << "  \"p95_ms\": " << s.p95_ms << ",\n";
    os << pad << "  \"p99_ms\": " << s.p99_ms << ",\n";
    os << pad << "  \"throughput_fps\": " << s.throughput_fps << "\n";
    os << pad << "}";
    return os.str();
}

// ─── Collect image paths ─────────────────────────────────────────────────────
static std::vector<std::string> collect_images(const std::string& dir, int max_images) {
    std::vector<std::string> paths;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (max_images > 0 && (int)paths.size() > max_images) {
        paths.resize(max_images);
    }
    return paths;
}

// ─── Parse CLI args ──────────────────────────────────────────────────────────
struct CLIArgs {
    std::string data_dir;
    std::string heatmap_engine;
    std::string classifier_engine;
    std::string output_path;
    int num_images = 0;    // 0 = all
    int warmup_iters = 20;
    int gpu_id = 0;
};

static CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) args.data_dir = argv[++i];
        else if (arg == "--heatmap-engine" && i + 1 < argc) args.heatmap_engine = argv[++i];
        else if (arg == "--classifier-engine" && i + 1 < argc) args.classifier_engine = argv[++i];
        else if (arg == "--output" && i + 1 < argc) args.output_path = argv[++i];
        else if (arg == "--num-images" && i + 1 < argc) args.num_images = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) args.warmup_iters = std::stoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc) args.gpu_id = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_pipeline [options]\n"
                      << "  --data-dir DIR           Input image directory\n"
                      << "  --heatmap-engine PATH    Heatmap TRT engine file\n"
                      << "  --classifier-engine PATH Classifier TRT engine file\n"
                      << "  --output PATH            Output JSON path\n"
                      << "  --num-images N           Number of images (0=all)\n"
                      << "  --warmup N               TRT warmup iterations (default 20)\n"
                      << "  --gpu ID                 GPU device ID (default 0)\n";
            exit(0);
        }
    }
    return args;
}

// ═══════════════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    CLIArgs args = parse_args(argc, argv);

    if (args.data_dir.empty() || args.heatmap_engine.empty() ||
        args.classifier_engine.empty()) {
        std::cerr << "Error: --data-dir, --heatmap-engine, and --classifier-engine "
                  << "are required.\nUse --help for usage.\n";
        return 1;
    }

    // ─── System info ─────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));

    std::cout << "=== C++ Pipeline Benchmark ===" << std::endl;
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem >> 20)
              << " MB)" << std::endl;

    // ─── Collect images ──────────────────────────────────────────────────
    auto image_paths = collect_images(args.data_dir, args.num_images);
    int num_images = (int)image_paths.size();
    std::cout << "Images: " << num_images << std::endl;

    if (num_images == 0) {
        std::cerr << "No images found in " << args.data_dir << std::endl;
        return 1;
    }

    // ─── Create pipeline ─────────────────────────────────────────────────
    PipelineConfig config;
    config.heatmap_engine_path = args.heatmap_engine;
    config.classifier_engine_path = args.classifier_engine;
    config.gpu_id = args.gpu_id;

    HolographicPipeline pipeline(config);
    std::cout << "Pipeline created. Warming up..." << std::endl;
    pipeline.warmup(args.warmup_iters);
    std::cout << "Warmup complete." << std::endl;

    // ─── Create CUDA events for GPU timing ───────────────────────────────
    cudaEvent_t ev_start, ev_s1, ev_s2, ev_s3, ev_s4, ev_s5, ev_s6;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_s1));
    CUDA_CHECK(cudaEventCreate(&ev_s2));
    CUDA_CHECK(cudaEventCreate(&ev_s3));
    CUDA_CHECK(cudaEventCreate(&ev_s4));
    CUDA_CHECK(cudaEventCreate(&ev_s5));
    CUDA_CHECK(cudaEventCreate(&ev_s6));

    // ─── Timing storage ──────────────────────────────────────────────────
    std::vector<double> t_load, t_ema, t_recon, t_detect, t_classify, t_postproc, t_e2e;
    int total_detections = 0, total_filtered = 0;
    int pd_count = 0, ec_count = 0;

    // ─── Benchmark loop ──────────────────────────────────────────────────
    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_images; ++i) {
        cudaStream_t s = pipeline.stream();

        // Record start
        CUDA_CHECK(cudaEventRecord(ev_start, s));

        // Stage 1: Load & Resize (CPU-heavy, use chrono)
        auto cpu_s1 = std::chrono::high_resolution_clock::now();
        pipeline.stage1_load_resize(image_paths[i]);
        auto cpu_e1 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_s1, s));

        // Stage 2: EMA
        pipeline.stage2_ema();
        CUDA_CHECK(cudaEventRecord(ev_s2, s));

        // Stage 3: Reconstruction
        pipeline.stage3_reconstruct();
        CUDA_CHECK(cudaEventRecord(ev_s3, s));

        // Stage 4: Heatmap + Peaks
        pipeline.stage4_detect();
        CUDA_CHECK(cudaEventRecord(ev_s4, s));

        // Stage 5: Classification
        pipeline.stage5_classify();
        CUDA_CHECK(cudaEventRecord(ev_s5, s));

        // Stage 6: Post-processing
        auto detections = pipeline.stage6_postprocess();
        CUDA_CHECK(cudaEventRecord(ev_s6, s));

        // Synchronize and collect timings
        CUDA_CHECK(cudaEventSynchronize(ev_s6));

        float ms_s1_s2, ms_s2_s3, ms_s3_s4, ms_s4_s5, ms_s5_s6, ms_total;
        CUDA_CHECK(cudaEventElapsedTime(&ms_s1_s2, ev_s1, ev_s2));
        CUDA_CHECK(cudaEventElapsedTime(&ms_s2_s3, ev_s2, ev_s3));
        CUDA_CHECK(cudaEventElapsedTime(&ms_s3_s4, ev_s3, ev_s4));
        CUDA_CHECK(cudaEventElapsedTime(&ms_s4_s5, ev_s4, ev_s5));
        CUDA_CHECK(cudaEventElapsedTime(&ms_s5_s6, ev_s5, ev_s6));
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev_start, ev_s6));

        // Stage 1 timing from chrono (CPU bound)
        double ms_load = std::chrono::duration<double, std::milli>(cpu_e1 - cpu_s1).count();

        t_load.push_back(ms_load);
        t_ema.push_back(ms_s1_s2);
        t_recon.push_back(ms_s2_s3);
        t_detect.push_back(ms_s3_s4);
        t_classify.push_back(ms_s4_s5);
        t_postproc.push_back(ms_s5_s6);
        t_e2e.push_back(ms_total);

        // Detection stats
        total_detections += pipeline.last_raw_detection_count();
        total_filtered += (int)detections.size();
        for (auto& d : detections) {
            if (d.class_name == "PD") pd_count++;
            else if (d.class_name == "EC") ec_count++;
        }

        // Progress
        if ((i + 1) % 100 == 0 || i == num_images - 1) {
            printf("  [%d/%d] e2e=%.2f ms (%.0f FPS)\n",
                   i + 1, num_images, ms_total, 1000.0 / ms_total);
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

    // ─── Compute stats ───────────────────────────────────────────────────
    auto s_load     = compute_stats(t_load);
    auto s_ema      = compute_stats(t_ema);
    auto s_recon    = compute_stats(t_recon);
    auto s_detect   = compute_stats(t_detect);
    auto s_classify = compute_stats(t_classify);
    auto s_postproc = compute_stats(t_postproc);
    auto s_e2e      = compute_stats(t_e2e);

    // GPU memory
    size_t mem_free, mem_total;
    CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
    size_t mem_used_mb = (mem_total - mem_free) >> 20;

    // ─── Print summary ───────────────────────────────────────────────────
    printf("\n=== Results ===\n");
    printf("End-to-end: %.2f ms mean (%.0f FPS), %.2f ms median\n",
           s_e2e.mean_ms, s_e2e.throughput_fps, s_e2e.median_ms);
    printf("Per-stage (mean ms): load=%.3f ema=%.3f recon=%.3f detect=%.3f "
           "classify=%.3f postproc=%.3f\n",
           s_load.mean_ms, s_ema.mean_ms, s_recon.mean_ms,
           s_detect.mean_ms, s_classify.mean_ms, s_postproc.mean_ms);
    printf("Wall clock: %.2f s for %d images\n", wall_seconds, num_images);
    printf("Detections: %d raw, %d filtered (PD=%d, EC=%d)\n",
           total_detections, total_filtered, pd_count, ec_count);
    printf("GPU memory: %zu MB used\n", mem_used_mb);

    // ─── Write JSON ──────────────────────────────────────────────────────
    if (!args.output_path.empty()) {
        // Ensure output directory exists
        fs::path out_path(args.output_path);
        if (out_path.has_parent_path()) {
            fs::create_directories(out_path.parent_path());
        }

        std::ofstream ofs(args.output_path);
        ofs << "{\n";

        // system_info
        ofs << "  \"system_info\": {\n";
        ofs << "    \"gpu\": \"" << json_escape(prop.name) << "\",\n";
        ofs << "    \"gpu_memory_mb\": " << (prop.totalGlobalMem >> 20) << ",\n";
        ofs << "    \"cuda_version\": \"" << (CUDART_VERSION / 1000) << "."
            << ((CUDART_VERSION % 1000) / 10) << "\",\n";
        ofs << "    \"backend\": \"C++ TensorRT\"\n";
        ofs << "  },\n";

        // config
        ofs << "  \"config\": {\n";
        ofs << "    \"num_images\": " << num_images << ",\n";
        ofs << "    \"img_size\": " << config.img_size << ",\n";
        ofs << "    \"backend\": \"cpp_trt\",\n";
        ofs << "    \"device\": \"cuda:" << args.gpu_id << "\"\n";
        ofs << "  },\n";

        // stage_statistics
        ofs << "  \"stage_statistics\": {\n";
        ofs << "    \"stage1_load\": " << stats_to_json(s_load) << ",\n";
        ofs << "    \"stage2_ema\": " << stats_to_json(s_ema) << ",\n";
        ofs << "    \"stage3_holographic\": " << stats_to_json(s_recon) << ",\n";
        ofs << "    \"stage4_heatmap\": " << stats_to_json(s_detect) << ",\n";
        ofs << "    \"stage5_classification\": " << stats_to_json(s_classify) << ",\n";
        ofs << "    \"stage6_postprocessing\": " << stats_to_json(s_postproc) << "\n";
        ofs << "  },\n";

        // end_to_end
        ofs << "  \"end_to_end\": " << stats_to_json(s_e2e, 2) << ",\n";

        // gpu_memory
        ofs << "  \"gpu_memory\": {\n";
        ofs << "    \"used_mb\": " << mem_used_mb << "\n";
        ofs << "  },\n";

        // detection_summary
        ofs << "  \"detection_summary\": {\n";
        ofs << "    \"total_detections\": " << total_detections << ",\n";
        ofs << "    \"total_filtered\": " << total_filtered << ",\n";
        ofs << "    \"pd_count\": " << pd_count << ",\n";
        ofs << "    \"ec_count\": " << ec_count << "\n";
        ofs << "  }\n";

        ofs << "}\n";
        ofs.close();
        std::cout << "Results written to " << args.output_path << std::endl;
    }

    // ─── Cleanup CUDA events ─────────────────────────────────────────────
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_s1);
    cudaEventDestroy(ev_s2);
    cudaEventDestroy(ev_s3);
    cudaEventDestroy(ev_s4);
    cudaEventDestroy(ev_s5);
    cudaEventDestroy(ev_s6);

    return 0;
}
