#include "pipeline_stages.h"
#include "holographic_reconstruction.h"  // for CUDA_CHECK
#include <stdexcept>

static constexpr int BLOCK = 256;

// ─── Fused EMA kernel: background update + enhancement ──────────────────────
// bg[i]  = alpha * frame[i] + (1 - alpha) * bg[i]
// out[i] = clamp(|frame[i] - bg[i]| * scale + mean, 0, 255) / 255
__global__ void ema_enhance_kernel(
    const float* __restrict__ frame,    // [H*W] float [0,255]
    float* __restrict__ background,     // [H*W] float [0,255] (in/out)
    float* __restrict__ output,         // [H*W] float [0,1]
    int total,
    float alpha,
    float scale,
    float mean_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float f = frame[idx];
    float bg = background[idx];

    // Update background EMA
    bg = alpha * f + (1.0f - alpha) * bg;
    background[idx] = bg;

    // Enhancement
    float diff = fabsf(f - bg);
    float enhanced = diff * scale + mean_val;
    enhanced = fminf(fmaxf(enhanced, 0.0f), 255.0f) / 255.0f;
    output[idx] = enhanced;
}

// ─── Zero-fill kernel ────────────────────────────────────────────────────────
__global__ void zero_fill_kernel(float* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMAStage implementation
// ═══════════════════════════════════════════════════════════════════════════════

EMAStage::EMAStage(float alpha, float scale, float mean, int warmup, int gpu_id)
    : alpha_(alpha), scale_(scale), mean_(mean),
      warmup_target_(warmup), gpu_id_(gpu_id) {}

EMAStage::~EMAStage() {
    if (d_background_) cudaFree(d_background_);
}

void EMAStage::reset() {
    frame_count_ = 0;
    if (d_background_) {
        cudaFree(d_background_);
        d_background_ = nullptr;
    }
    buf_size_ = 0;
}

void EMAStage::process(const float* d_frame, float* d_output, int H, int W,
                       cudaStream_t stream)
{
    CUDA_CHECK(cudaSetDevice(gpu_id_));
    int total = H * W;

    // Allocate background buffer on first use (or if size changed)
    if (buf_size_ != total) {
        if (d_background_) cudaFree(d_background_);
        CUDA_CHECK(cudaMalloc(&d_background_, total * sizeof(float)));
        buf_size_ = total;
        frame_count_ = 0;
    }

    int grid = (total + BLOCK - 1) / BLOCK;

    if (frame_count_ == 0) {
        // First frame: copy frame → background, output zeros
        CUDA_CHECK(cudaMemcpyAsync(d_background_, d_frame,
                                    total * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
        zero_fill_kernel<<<grid, BLOCK, 0, stream>>>(d_output, total);
        CUDA_CHECK(cudaGetLastError());
        frame_count_ = 1;
        return;
    }

    frame_count_++;

    if (frame_count_ < warmup_target_) {
        // During warmup: still update background, output zeros
        ema_enhance_kernel<<<grid, BLOCK, 0, stream>>>(
            d_frame, d_background_, d_output, total,
            alpha_, scale_, mean_);
        CUDA_CHECK(cudaGetLastError());
        // Overwrite output with zeros during warmup
        zero_fill_kernel<<<grid, BLOCK, 0, stream>>>(d_output, total);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Normal processing
    ema_enhance_kernel<<<grid, BLOCK, 0, stream>>>(
        d_frame, d_background_, d_output, total,
        alpha_, scale_, mean_);
    CUDA_CHECK(cudaGetLastError());
}
