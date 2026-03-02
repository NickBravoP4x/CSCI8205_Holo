#include "pipeline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════════════

HolographicPipeline::HolographicPipeline(const PipelineConfig& config)
    : config_(config),
      filter_(config.filter_max_history, config.filter_distance_threshold,
              config.filter_min_hits)
{
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Create stage objects
    ema_ = std::make_unique<EMAStage>(
        config_.ema_alpha, config_.ema_scale, config_.ema_mean,
        config_.ema_warmup, config_.gpu_id);

    CUDAReconParams rp;
    rp.resolution  = config_.recon_resolution;
    rp.wavelength  = config_.recon_wavelength;
    rp.z_start     = config_.recon_z_start;
    rp.z_step      = config_.recon_z_step;
    rp.num_planes  = config_.recon_num_planes;
    rp.shift_mean  = config_.recon_shift_mean;
    rp.padding     = config_.recon_padding;
    rp.gpu_id      = config_.gpu_id;
    reconstructor_ = std::make_unique<CUDAHolographicReconstructor>(rp);

    heatmap_engine_ = std::make_unique<TensorRTEngine>(
        config_.heatmap_engine_path, config_.gpu_id);
    classifier_engine_ = std::make_unique<TensorRTEngine>(
        config_.classifier_engine_path, config_.gpu_id);

    allocate_buffers();
}

HolographicPipeline::~HolographicPipeline() {
    free_buffers();
    if (stream_) cudaStreamDestroy(stream_);
}

void HolographicPipeline::allocate_buffers() {
    int S = config_.img_size;
    int HW = S * S;
    int C = config_.recon_num_planes;  // 3
    int hm_size = (S / 4) * (S / 4);  // 112*112
    int crop_elems = config_.max_batch * 3 * config_.crop_size * config_.crop_size;
    int class_out_elems = config_.max_batch * 2;

    // GPU buffers
    CUDA_CHECK(cudaMalloc(&d_frame_float_, HW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enhanced_, HW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normalized_, C * HW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_heatmap_, hm_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_crops_, crop_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_class_out_, class_out_elems * sizeof(float)));

    // Pinned host buffers (sized for max of frame upload or crop batch)
    int pinned_sz = std::max(HW, crop_elems);
    CUDA_CHECK(cudaMallocHost(&h_frame_pinned_, pinned_sz * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_frame_uint8_, HW * sizeof(uint8_t)));
    CUDA_CHECK(cudaMallocHost(&h_class_out_, class_out_elems * sizeof(float)));
}

void HolographicPipeline::free_buffers() {
    auto gpu_free = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto host_free = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };

    gpu_free(d_frame_float_);
    gpu_free(d_enhanced_);
    gpu_free(d_normalized_);
    gpu_free(d_heatmap_);
    gpu_free(d_crops_);
    gpu_free(d_class_out_);

    host_free(h_frame_pinned_);
    host_free(h_frame_uint8_);
    host_free(h_class_out_);
}

void HolographicPipeline::warmup(int trt_iters) {
    heatmap_engine_->warmup(trt_iters);
    classifier_engine_->warmup(trt_iters);

    // Warmup EMA with zeros
    int S = config_.img_size;
    CUDA_CHECK(cudaMemset(d_frame_float_, 0, S * S * sizeof(float)));
    for (int i = 0; i < config_.ema_warmup + 2; ++i) {
        ema_->process(d_frame_float_, d_enhanced_, S, S, stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Reset state after warmup
    reset();
}

void HolographicPipeline::reset() {
    ema_->reset();
    filter_.reset();
    last_peaks_.clear();
    last_detections_.clear();
}

// ─── Stage 1: Load & Resize ─────────────────────────────────────────────────
void HolographicPipeline::stage1_load_resize(const std::string& path) {
    int S = config_.img_size;

    // Load grayscale
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Cannot read image: " + path);
    }

    // Resize to SxS
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(S, S), 0, 0, cv::INTER_AREA);

    // Store uint8 copy for crop extraction (Stage 5)
    cpu_frame_ = resized.clone();
    std::memcpy(h_frame_uint8_, resized.data, S * S);

    // Convert to float32 [0, 255] in pinned buffer
    for (int i = 0; i < S * S; ++i) {
        h_frame_pinned_[i] = static_cast<float>(resized.data[i]);
    }

    // H2D async
    CUDA_CHECK(cudaMemcpyAsync(d_frame_float_, h_frame_pinned_,
                                S * S * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));
}

// ─── Stage 2: EMA Enhancement ───────────────────────────────────────────────
void HolographicPipeline::stage2_ema() {
    int S = config_.img_size;
    ema_->process(d_frame_float_, d_enhanced_, S, S, stream_);
}

// ─── Stage 3: Holographic Reconstruction ────────────────────────────────────
void HolographicPipeline::stage3_reconstruct() {
    int S = config_.img_size;

    // Copy enhanced frame (which is in [0,1]) scaled to [0,255] into
    // reconstructor's hologram buffer. The reconstructor expects [0,255] range
    // (it computes median for padding).
    // Actually, looking at the Python pipeline: the enhanced output [0,1] is
    // passed directly as the hologram input to reconstruction. The C++ reconstructor
    // also works with any float range (median padding adapts).
    // So we copy d_enhanced_ → reconstructor's d_hologram_ directly.
    reconstructor_->prepare(S, S);
    float* d_holo = reconstructor_->device_hologram_ptr();
    CUDA_CHECK(cudaMemcpyAsync(d_holo, d_enhanced_,
                                S * S * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    reconstructor_->reconstruct_intensity_device(S, S);
    // Result: d_intensity_ is [3][448][448] in [0,1] — NCHW layout
}

// ─── Stage 4: Heatmap + Peak Detection ──────────────────────────────────────
void HolographicPipeline::stage4_detect() {
    int S = config_.img_size;

    // Get intensity buffer from reconstructor (NCHW: [3][H][W])
    const float* d_intensity = reconstructor_->device_intensity_nchw();

    // ImageNet normalize: [3][H][W] → [3][H][W] (in-place to d_normalized_)
    imagenet_normalize(d_intensity, d_normalized_, 3, S, S, stream_);

    // TRT heatmap inference: [1,3,448,448] → [1,1,112,112]
    heatmap_engine_->infer(d_normalized_, d_heatmap_, 1, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Peak detection on heatmap
    int hm_h = S / 4;  // 112
    int hm_w = S / 4;
    last_peaks_ = detect_peaks(d_heatmap_, hm_h, hm_w,
                               config_.heatmap_threshold,
                               config_.heatmap_nms_distance,
                               config_.heatmap_topk,
                               config_.output_scale,
                               stream_);
}

// ─── Stage 5: Classification ────────────────────────────────────────────────
void HolographicPipeline::stage5_classify() {
    int S = config_.img_size;
    int crop_sz = config_.crop_size;
    float det_scale = config_.output_scale;
    float img_scale = (float)S;

    last_detections_.clear();

    if (last_peaks_.empty()) return;

    // Extract crops on CPU
    float scale = img_scale / det_scale;  // 448/640
    int half = crop_sz / 2;

    std::vector<cv::Mat> crops;
    std::vector<int> valid_peak_indices;

    for (size_t i = 0; i < last_peaks_.size(); ++i) {
        int cx = (int)(last_peaks_[i].x * scale);
        int cy = (int)(last_peaks_[i].y * scale);

        int x1 = std::max(cx - half, 0);
        int y1 = std::max(cy - half, 0);
        int x2 = std::min(cx + half, S);
        int y2 = std::min(cy + half, S);

        if (x2 <= x1 || y2 <= y1) continue;

        cv::Mat crop = cpu_frame_(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // Letterbox to crop_sz x crop_sz with gray=114
        cv::Mat canvas(crop_sz, crop_sz, CV_8UC1, cv::Scalar(114));
        int pad_y = (crop_sz - crop.rows) / 2;
        int pad_x = (crop_sz - crop.cols) / 2;
        crop.copyTo(canvas(cv::Rect(pad_x, pad_y, crop.cols, crop.rows)));

        crops.push_back(canvas);
        valid_peak_indices.push_back((int)i);
    }

    if (crops.empty()) return;

    // Process in batches
    int num_crops = (int)crops.size();
    int max_batch = config_.max_batch;

    for (int start = 0; start < num_crops; start += max_batch) {
        int batch_size = std::min(max_batch, num_crops - start);
        int crop_hw = crop_sz * crop_sz;

        // Build batch: convert uint8 grayscale → float32 RGB [B,3,H,W]
        // normalized to [0,1] (classifier expects this from YOLO preprocessing)
        for (int b = 0; b < batch_size; ++b) {
            const uint8_t* src = crops[start + b].data;
            float* dst_base = h_frame_pinned_;  // reuse pinned buffer
            // Duplicate grayscale across 3 channels in NCHW layout
            for (int c = 0; c < 3; ++c) {
                for (int j = 0; j < crop_hw; ++j) {
                    dst_base[(b * 3 + c) * crop_hw + j] =
                        static_cast<float>(src[j]) / 255.0f;
                }
            }
        }

        // H2D
        int batch_elems = batch_size * 3 * crop_hw;
        CUDA_CHECK(cudaMemcpyAsync(d_crops_, h_frame_pinned_,
                                    batch_elems * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_));

        // TRT classify
        classifier_engine_->infer(d_crops_, d_class_out_, batch_size, stream_);

        // D2H class outputs
        int out_elems = batch_size * 2;
        CUDA_CHECK(cudaMemcpyAsync(h_class_out_, d_class_out_,
                                    out_elems * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Build detection results
        for (int b = 0; b < batch_size; ++b) {
            int peak_idx = valid_peak_indices[start + b];
            float prob0 = h_class_out_[b * 2 + 0];  // EC
            float prob1 = h_class_out_[b * 2 + 1];  // PD

            Detection det;
            det.x = last_peaks_[peak_idx].x;
            det.y = last_peaks_[peak_idx].y;
            det.confidence = last_peaks_[peak_idx].confidence;

            if (prob1 > prob0) {
                det.class_name = "PD";
                det.class_confidence = prob1;
            } else {
                det.class_name = "EC";
                det.class_confidence = prob0;
            }

            last_detections_.push_back(det);
        }
    }
}

// ─── Stage 6: Post-processing ───────────────────────────────────────────────
std::vector<Detection> HolographicPipeline::stage6_postprocess() {
    if (last_detections_.empty()) return {};

    // Filter by detection threshold
    std::vector<Detection> filtered;
    for (auto& d : last_detections_) {
        if (d.confidence >= config_.heatmap_threshold) {
            filtered.push_back(d);
        }
    }

    if (filtered.empty()) return {};

    // Static centerpoint filter
    auto mask = filter_.update_and_filter(filtered);

    // Apply mask + classification threshold
    std::vector<Detection> result;
    for (size_t i = 0; i < filtered.size(); ++i) {
        if (mask[i] && filtered[i].class_confidence >= config_.class_threshold) {
            result.push_back(filtered[i]);
        }
    }

    return result;
}

// ─── Full pipeline ──────────────────────────────────────────────────────────
std::vector<Detection> HolographicPipeline::process_frame(const std::string& path) {
    stage1_load_resize(path);
    stage2_ema();
    stage3_reconstruct();
    stage4_detect();
    stage5_classify();
    return stage6_postprocess();
}
