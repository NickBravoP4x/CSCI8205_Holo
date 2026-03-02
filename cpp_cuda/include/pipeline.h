#pragma once

#include "pipeline_stages.h"
#include "tensorrt_engine.h"
#include "holographic_reconstruction.h"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════════
// HolographicPipeline — sequential single-GPU orchestrator
// ═══════════════════════════════════════════════════════════════════════════════

class HolographicPipeline {
public:
    HolographicPipeline(const PipelineConfig& config);
    ~HolographicPipeline();

    HolographicPipeline(const HolographicPipeline&) = delete;
    HolographicPipeline& operator=(const HolographicPipeline&) = delete;

    /// Warmup all stages (TRT engines, EMA, etc.)
    void warmup(int trt_iters = 20);

    /// Reset EMA and post-processing state
    void reset();

    /// Run full pipeline on one image file. Returns filtered detections.
    std::vector<Detection> process_frame(const std::string& image_path);

    // ─── Individual stage methods (for fine-grained timing) ──────────────
    void stage1_load_resize(const std::string& path);
    void stage2_ema();
    void stage3_reconstruct();
    void stage4_detect();
    void stage5_classify();
    std::vector<Detection> stage6_postprocess();

    // ─── Accessors for benchmark timing ──────────────────────────────────
    const std::vector<Detection>& last_detections() const { return last_detections_; }
    int last_raw_detection_count() const { return (int)last_peaks_.size(); }
    cudaStream_t stream() const { return stream_; }

private:
    PipelineConfig config_;

    // CUDA stream
    cudaStream_t stream_ = nullptr;

    // Stage objects
    std::unique_ptr<EMAStage> ema_;
    std::unique_ptr<CUDAHolographicReconstructor> reconstructor_;
    std::unique_ptr<TensorRTEngine> heatmap_engine_;
    std::unique_ptr<TensorRTEngine> classifier_engine_;
    StaticCenterpointFilter filter_;

    // ─── Pre-allocated GPU buffers ───────────────────────────────────────
    float* d_frame_float_ = nullptr;
    float* d_enhanced_ = nullptr;
    float* d_normalized_ = nullptr;
    float* d_heatmap_ = nullptr;
    float* d_crops_ = nullptr;
    float* d_class_out_ = nullptr;

    // Pinned host buffers
    float* h_frame_pinned_ = nullptr;
    uint8_t* h_frame_uint8_ = nullptr;
    float* h_class_out_ = nullptr;

    // ─── Per-frame state ─────────────────────────────────────────────────
    cv::Mat cpu_frame_;
    std::vector<Peak> last_peaks_;
    std::vector<Detection> last_detections_;

    void allocate_buffers();
    void free_buffers();
};
