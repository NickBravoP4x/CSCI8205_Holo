#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>

// ─── Detection result ────────────────────────────────────────────────────────
struct Detection {
    float x;               // in output_scale coordinates (640)
    float y;
    float confidence;      // heatmap peak confidence [0,1]
    std::string class_name;       // "EC" or "PD"
    float class_confidence;       // classifier confidence [0,1]
};

// ─── Pipeline configuration ──────────────────────────────────────────────────
struct PipelineConfig {
    // Image
    int img_size = 448;
    float output_scale = 640.0f;

    // EMA
    float ema_alpha = 0.05f;
    float ema_scale = 2.0f;
    float ema_mean = 105.0f;
    int   ema_warmup = 5;

    // Reconstruction (3-plane)
    float recon_resolution = 0.087f;
    float recon_wavelength = 405.0f / 1.33f / 1000.0f;
    float recon_z_start = 0.0f;
    float recon_z_step = 6.0f;
    int   recon_num_planes = 3;
    float recon_shift_mean = 105.0f;
    int   recon_padding = 32;

    // Heatmap detection
    float heatmap_threshold = 0.3f;
    int   heatmap_nms_distance = 5;
    int   heatmap_topk = 50;

    // Classification
    int   crop_size = 128;
    float class_threshold = 0.25f;
    int   max_batch = 32;

    // Post-processing
    int   filter_max_history = 200;
    float filter_distance_threshold = 50.0f;
    int   filter_min_hits = 5;

    // Engine paths
    std::string heatmap_engine_path;
    std::string classifier_engine_path;

    // GPU
    int gpu_id = 0;
};

// ─── EMA Stage ───────────────────────────────────────────────────────────────

class EMAStage {
public:
    EMAStage(float alpha = 0.05f, float scale = 2.0f,
             float mean = 105.0f, int warmup = 5, int gpu_id = 0);
    ~EMAStage();

    /// Process one frame. d_frame: [H*W] float [0,255], d_output: [H*W] float [0,1]
    void process(const float* d_frame, float* d_output, int H, int W,
                 cudaStream_t stream = nullptr);

    void reset();

private:
    float alpha_, scale_, mean_;
    int warmup_target_;
    int frame_count_ = 0;
    int gpu_id_;

    float* d_background_ = nullptr;
    int buf_size_ = 0;
};

// ─── ImageNet Normalization (CUDA kernel) ────────────────────────────────────
/// Normalize [planes][H][W] intensity in [0,1] to ImageNet-normalized RGB
/// Input: d_intensity [3][H][W] in [0,1]
/// Output: d_normalized [1][3][H][W] ImageNet-normalized
void imagenet_normalize(const float* d_input, float* d_output,
                        int C, int H, int W, cudaStream_t stream = nullptr);

// ─── Peak Detection ──────────────────────────────────────────────────────────

struct Peak {
    float x, y, confidence;
};

/// Max-pool NMS + top-k peak extraction on heatmap
/// d_heatmap: [1][1][H][W], output: host vector of peaks (D2H inside)
std::vector<Peak> detect_peaks(const float* d_heatmap, int H, int W,
                               float threshold, int min_distance, int topk,
                               float output_scale, cudaStream_t stream = nullptr);

// ─── Static Centerpoint Filter (CPU) ─────────────────────────────────────────

class StaticCenterpointFilter {
public:
    StaticCenterpointFilter(int max_history = 200,
                            float distance_threshold = 50.0f,
                            int min_hits = 5);

    /// Returns indices of detections that pass the filter
    std::vector<bool> update_and_filter(const std::vector<Detection>& detections);

    void reset();

private:
    int max_history_;
    float distance_threshold_;
    int min_hits_;

    // Circular buffer of history points
    std::vector<float> history_x_;
    std::vector<float> history_y_;
    int history_size_ = 0;
    int history_start_ = 0;  // oldest entry index
};
