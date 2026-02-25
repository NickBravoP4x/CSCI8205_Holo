#include "holographic_reconstruction.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cfloat>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// ─── Constants ───────────────────────────────────────────────────────────────
static constexpr float PI = 3.14159265358979323846f;
static constexpr int BLOCK_SIZE = 256;

// ─── Kernel 1: Pad hologram with median ─────────────────────────────────────
// Pads the hologram into a complex buffer (real part only).
__global__ void pad_hologram_kernel(
    const float* __restrict__ input,     // H_in x W_in
    cufftComplex* __restrict__ output,   // Hp x Wp complex
    int H_in, int W_in,
    int Hp, int Wp,
    int padding,
    float median_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Hp * Wp;
    if (idx >= total) return;

    int row = idx / Wp;
    int col = idx % Wp;

    float val;
    int src_row = row - padding;
    int src_col = col - padding;
    if (src_row >= 0 && src_row < H_in && src_col >= 0 && src_col < W_in) {
        val = input[src_row * W_in + src_col];
    } else {
        val = median_val;
    }

    output[idx].x = val;
    output[idx].y = 0.0f;
}

// ─── Kernel 2: Build propagation filter (f2_new) ────────────────────────────
// Precomputes f2_new in DFT order (DC at corner), matching standard FFT layout.
// Exactly replicates Python: f2 = fx²+fy², f2_new = fftshift(f2),
// f2_new = sqrt(1 - (λ/r)² * f2_new)
__global__ void build_propagation_filter_kernel(
    float* __restrict__ f2_new,
    int Hp, int Wp,
    float lambda_over_res_sq)  // (wavelength/resolution)^2
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Hp * Wp;
    if (idx >= total) return;

    int row = idx / Wp;
    int col = idx % Wp;

    // Centered frequency coordinates
    float fx = (float)col / (float)Wp - 0.5f;
    float fy = (float)row / (float)Hp - 0.5f;
    float f2 = fx * fx + fy * fy;

    // Apply fftshift: swap quadrants to get DFT order (DC at corner)
    int shifted_row = (row + Hp / 2) % Hp;
    int shifted_col = (col + Wp / 2) % Wp;
    int shifted_idx = shifted_row * Wp + shifted_col;

    float val = lambda_over_res_sq * f2;
    val = fmaxf(1.0f - val, 0.0f);
    f2_new[shifted_idx] = sqrtf(val);
}

// ─── Kernel 3: Apply propagation and broadcast ──────────────────────────────
// For each depth plane z_k, computes:
//   Fplane[k] = Fholo * exp(-2j*pi*z_k*f2_new / wavelength)
__global__ void apply_propagation_and_broadcast_kernel(
    const cufftComplex* __restrict__ Fholo,  // Hp x Wp
    const float* __restrict__ f2_new,         // Hp x Wp
    const float* __restrict__ z_list,         // num_planes
    cufftComplex* __restrict__ Fplanes,       // num_planes x Hp x Wp
    int Hp, int Wp,
    int num_planes,
    float wavelength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int plane_size = Hp * Wp;
    int total = num_planes * plane_size;
    if (idx >= total) return;

    int plane = idx / plane_size;
    int pixel = idx % plane_size;

    float z = z_list[plane];
    float f2 = f2_new[pixel];

    // Phase: -2*pi*z*f2_new / wavelength
    float phase = -2.0f * PI * z * f2 / wavelength;
    float cos_p, sin_p;
    sincosf(phase, &sin_p, &cos_p);

    // Complex multiply: Fholo * exp(j*phase)
    cufftComplex fh = Fholo[pixel];
    cufftComplex result;
    result.x = fh.x * cos_p - fh.y * sin_p;
    result.y = fh.x * sin_p + fh.y * cos_p;

    Fplanes[idx] = result;
}

// ─── Kernel 4: Phase correction, intensity, remove padding ──────────────────
// After IFFT (standard, no shift trick), applies:
//   1. Phase correction: rec *= exp(2j*pi*z/lambda)
//   2. Intensity: |rec|^2
//   3. Remove padding (crop to H_in x W_in)
//   4. Store as [plane][row][col] layout
__global__ void phase_correction_and_intensity_kernel(
    const cufftComplex* __restrict__ rec,  // num_planes x Hp x Wp (IFFT output)
    float* __restrict__ intensity,          // num_planes x H_in x W_in
    const float* __restrict__ z_list,
    int Hp, int Wp,
    int H_in, int W_in,
    int padding,
    int num_planes,
    float wavelength,
    float fft_norm)  // 1.0/(Hp*Wp) normalization factor
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_plane_size = H_in * W_in;
    int total = num_planes * out_plane_size;
    if (idx >= total) return;

    int plane = idx / out_plane_size;
    int pixel = idx % out_plane_size;
    int out_row = pixel / W_in;
    int out_col = pixel % W_in;

    // Map to padded coordinates
    int pad_row = out_row + padding;
    int pad_col = out_col + padding;
    int pad_idx = plane * (Hp * Wp) + pad_row * Wp + pad_col;

    // Get IFFT result (cuFFT doesn't normalize, so divide by N)
    cufftComplex r = rec[pad_idx];
    float re = r.x * fft_norm;
    float im = r.y * fft_norm;

    // Phase correction: rec *= exp(2j*pi*z/lambda)
    float z = z_list[plane];
    float phase = 2.0f * PI * z / wavelength;
    float cos_p, sin_p;
    sincosf(phase, &sin_p, &cos_p);
    float re2 = re * cos_p - im * sin_p;
    float im2 = re * sin_p + im * cos_p;

    // Intensity = |rec|^2
    float intens = re2 * re2 + im2 * im2;
    intensity[idx] = intens;
}

// ─── Kernel 5: Normalize intensity ──────────────────────────────────────────
// Min-max normalize to [0,1], then apply background shift scaling.
__global__ void normalize_intensity_kernel(
    float* __restrict__ intensity,  // in-place, num_planes * H * W
    int total_pixels,
    float min_val,
    float max_val,
    float shift_mean)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float range = max_val - min_val + 1e-8f;
    float val = (intensity[idx] - min_val) / range;
    intensity[idx] = val;
}

// Second pass: apply shift scaling after computing the mean
__global__ void apply_shift_scaling_kernel(
    float* __restrict__ intensity,
    int total_pixels,
    float scale_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float val = intensity[idx] * scale_factor;
    intensity[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

// ─── Kernel: Transpose from [plane][row][col] to [row][col][plane] ──────────
__global__ void transpose_planes_kernel(
    const float* __restrict__ src,   // [num_planes][H][W]
    float* __restrict__ dst,          // [H][W][num_planes]
    int H, int W, int num_planes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W * num_planes;
    if (idx >= total) return;

    // dst index: [row][col][plane]
    int plane = idx % num_planes;
    int rc = idx / num_planes;
    int row = rc / W;
    int col = rc % W;

    // src index: [plane][row][col]
    int src_idx = plane * (H * W) + row * W + col;
    dst[idx] = src[src_idx];
}

// ─── Helper: find min and max using thrust ──────────────────────────────────
static void find_min_max(const float* d_data, int n, float& min_val, float& max_val) {
    thrust::device_ptr<const float> ptr(d_data);
    auto minmax = thrust::minmax_element(ptr, ptr + n);
    CUDA_CHECK(cudaMemcpy(&min_val, minmax.first.get(), sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&max_val, minmax.second.get(), sizeof(float), cudaMemcpyDeviceToHost));
}

// ─── Helper: compute sum using thrust ───────────────────────────────────────
static float compute_sum(const float* d_data, int n) {
    thrust::device_ptr<const float> ptr(d_data);
    return thrust::reduce(ptr, ptr + n, 0.0f, thrust::plus<float>());
}

// ─── Helper: find median of host array using thrust sort on device ──────────
static float compute_median_device(const float* h_data, int n, float* d_buf) {
    CUDA_CHECK(cudaMemcpy(d_buf, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    thrust::device_ptr<float> ptr(d_buf);
    thrust::sort(ptr, ptr + n);
    float median;
    if (n % 2 == 0) {
        float a, b;
        CUDA_CHECK(cudaMemcpy(&a, d_buf + n / 2 - 1, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b, d_buf + n / 2, sizeof(float), cudaMemcpyDeviceToHost));
        median = (a + b) * 0.5f;
    } else {
        CUDA_CHECK(cudaMemcpy(&median, d_buf + n / 2, sizeof(float), cudaMemcpyDeviceToHost));
    }
    return median;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDAHolographicReconstructor implementation
// ═══════════════════════════════════════════════════════════════════════════════

CUDAHolographicReconstructor::CUDAHolographicReconstructor(const Params& params)
    : params_(params)
{
    CUDA_CHECK(cudaSetDevice(params_.gpu_id));

    // Apply resolution scaling (matching Python logic)
    eff_resolution_ = params_.resolution;
    eff_z_start_    = params_.z_start;
    eff_z_step_     = params_.z_step;

    float resolution_threshold = 0.5f;
    if (eff_resolution_ < resolution_threshold) {
        float scale = resolution_threshold / eff_resolution_;
        eff_resolution_ = scale * eff_resolution_;
        eff_z_start_    = scale * scale * eff_z_start_;
        eff_z_step_     = scale * scale * eff_z_step_;
    }
}

CUDAHolographicReconstructor::~CUDAHolographicReconstructor() {
    free_buffers();
}

void CUDAHolographicReconstructor::free_buffers() {
    if (plans_valid_) {
        cufftDestroy(plan_fwd_);
        cufftDestroy(plan_inv_bat_);
        plans_valid_ = false;
    }
    auto safe_free = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(d_hologram_);
    safe_free(d_padded_real_);
    safe_free(d_padded_);
    safe_free(d_Fholo_);
    safe_free(d_f2_new_);
    safe_free(d_z_list_);
    safe_free(d_Fplanes_);
    safe_free(d_rec_);
    safe_free(d_intensity_);
    safe_free(d_output_);
    safe_free(d_reduce_buf_);
}

void CUDAHolographicReconstructor::ensure_initialized(int height, int width) {
    if (height == last_height_ && width == last_width_ && plans_valid_) {
        return;
    }

    free_buffers();

    last_height_ = height;
    last_width_  = width;
    padded_h_    = height + 2 * params_.padding;
    padded_w_    = width  + 2 * params_.padding;

    int pad_total      = padded_h_ * padded_w_;
    int out_total      = height * width;
    int planes         = params_.num_planes;
    int plane_pad_total = planes * pad_total;
    int plane_out_total = planes * out_total;

    // Allocate device buffers
    CUDA_CHECK(cudaMalloc(&d_hologram_,    out_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_padded_,      pad_total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_Fholo_,       pad_total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_f2_new_,      pad_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z_list_,      planes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Fplanes_,     plane_pad_total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_rec_,         plane_pad_total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_intensity_,   plane_out_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_,      plane_out_total * sizeof(float)));
    // Reduction buffer: large enough for input median computation
    int reduce_sz = std::max(out_total, plane_out_total);
    CUDA_CHECK(cudaMalloc(&d_reduce_buf_,  reduce_sz * sizeof(float)));

    // Build z_list on host and upload
    std::vector<float> h_z(planes);
    float z_end = planes * eff_z_step_ + eff_z_start_;
    for (int i = 0; i < planes; ++i) {
        h_z[i] = eff_z_start_ + i * eff_z_step_;
    }
    CUDA_CHECK(cudaMemcpy(d_z_list_, h_z.data(), planes * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Build propagation filter
    float lambda_over_res_sq = (params_.wavelength / eff_resolution_)
                             * (params_.wavelength / eff_resolution_);
    int grid = (pad_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    build_propagation_filter_kernel<<<grid, BLOCK_SIZE>>>(
        d_f2_new_, padded_h_, padded_w_, lambda_over_res_sq);
    CUDA_CHECK(cudaGetLastError());

    // Create cuFFT plans
    // Forward: single 2D C2C
    CUFFT_CHECK(cufftPlan2d(&plan_fwd_, padded_h_, padded_w_, CUFFT_C2C));

    // Inverse: batched 2D C2C (num_planes batches)
    int n[2] = {padded_h_, padded_w_};
    int inembed[2] = {padded_h_, padded_w_};
    int onembed[2] = {padded_h_, padded_w_};
    CUFFT_CHECK(cufftPlanMany(&plan_inv_bat_, 2, n,
                               inembed, 1, pad_total,
                               onembed, 1, pad_total,
                               CUFFT_C2C, planes));

    plans_valid_ = true;
}

void CUDAHolographicReconstructor::reconstruct_intensity(
    const float* h_hologram, int height, int width, float* h_output)
{
    CUDA_CHECK(cudaSetDevice(params_.gpu_id));
    ensure_initialized(height, width);

    int out_total = height * width;
    int pad_total = padded_h_ * padded_w_;
    int planes    = params_.num_planes;

    // Upload hologram to device
    CUDA_CHECK(cudaMemcpy(d_hologram_, h_hologram, out_total * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Compute median on device
    float median_val = compute_median_device(h_hologram, out_total, d_reduce_buf_);

    // Kernel 1: Pad + fftshift trick
    int grid1 = (pad_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pad_hologram_kernel<<<grid1, BLOCK_SIZE>>>(
        d_hologram_, d_padded_, height, width,
        padded_h_, padded_w_, params_.padding, median_val);
    CUDA_CHECK(cudaGetLastError());

    // Forward FFT
    CUFFT_CHECK(cufftExecC2C(plan_fwd_, d_padded_, d_Fholo_, CUFFT_FORWARD));

    // Kernel 3: Apply propagation for all planes
    int total_prop = planes * pad_total;
    int grid3 = (total_prop + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_propagation_and_broadcast_kernel<<<grid3, BLOCK_SIZE>>>(
        d_Fholo_, d_f2_new_, d_z_list_, d_Fplanes_,
        padded_h_, padded_w_, planes, params_.wavelength);
    CUDA_CHECK(cudaGetLastError());

    // Batched inverse FFT
    CUFFT_CHECK(cufftExecC2C(plan_inv_bat_, d_Fplanes_, d_rec_, CUFFT_INVERSE));

    // Kernel 4: Phase correction + intensity + remove padding
    int total_out = planes * out_total;
    int grid4 = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float fft_norm = 1.0f / (float)(padded_h_ * padded_w_);
    phase_correction_and_intensity_kernel<<<grid4, BLOCK_SIZE>>>(
        d_rec_, d_intensity_, d_z_list_,
        padded_h_, padded_w_, height, width,
        params_.padding, planes, params_.wavelength, fft_norm);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 5a: Find min/max for normalization
    float min_val, max_val;
    find_min_max(d_intensity_, total_out, min_val, max_val);

    // Kernel 5b: Min-max normalize
    int grid5 = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    normalize_intensity_kernel<<<grid5, BLOCK_SIZE>>>(
        d_intensity_, total_out, min_val, max_val, params_.shift_mean);
    CUDA_CHECK(cudaGetLastError());

    // Compute mean for shift scaling
    float sum = compute_sum(d_intensity_, total_out);
    float mean_bg = sum / (float)total_out;
    float scale_factor = (params_.shift_mean / 255.0f) / (mean_bg + 1e-8f);

    // Kernel 5c: Apply shift scaling and clamp
    apply_shift_scaling_kernel<<<grid5, BLOCK_SIZE>>>(
        d_intensity_, total_out, scale_factor);
    CUDA_CHECK(cudaGetLastError());

    // Transpose from [plane][H][W] to [H][W][plane]
    int grid_t = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_planes_kernel<<<grid_t, BLOCK_SIZE>>>(
        d_intensity_, d_output_, height, width, planes);
    CUDA_CHECK(cudaGetLastError());

    // Download result
    CUDA_CHECK(cudaMemcpy(h_output, d_output_, total_out * sizeof(float),
                           cudaMemcpyDeviceToHost));
}
