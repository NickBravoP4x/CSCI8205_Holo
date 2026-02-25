#include "cpu_baseline.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <omp.h>

static constexpr float PI = 3.14159265358979323846f;

// ─── FFTW thread init (once) ────────────────────────────────────────────────
static struct FFTWThreadInit {
    FFTWThreadInit() {
        fftwf_init_threads();
    }
    ~FFTWThreadInit() {
        fftwf_cleanup_threads();
    }
} fftw_thread_init;

// ─── CPUHolographicReconstructor ────────────────────────────────────────────

CPUHolographicReconstructor::CPUHolographicReconstructor(const Params& params)
    : params_(params)
{
    // Apply resolution scaling
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

CPUHolographicReconstructor::~CPUHolographicReconstructor() {
    free_plans();
}

void CPUHolographicReconstructor::free_plans() {
    if (plan_fwd_) { fftwf_destroy_plan(plan_fwd_); plan_fwd_ = nullptr; }
    if (plan_inv_) { fftwf_destroy_plan(plan_inv_); plan_inv_ = nullptr; }
    if (fft_in_)   { fftwf_free(fft_in_);   fft_in_   = nullptr; }
    if (fft_out_)  { fftwf_free(fft_out_);  fft_out_  = nullptr; }
    if (plane_buf_){ fftwf_free(plane_buf_);plane_buf_ = nullptr; }
    if (ifft_out_) { fftwf_free(ifft_out_); ifft_out_ = nullptr; }
    plans_valid_ = false;
}

void CPUHolographicReconstructor::set_num_threads(int n) {
    if (n == params_.num_threads && plans_valid_) return;
    params_.num_threads = n;
    free_plans();
    last_height_ = 0;
    last_width_  = 0;
}

void CPUHolographicReconstructor::ensure_initialized(int height, int width) {
    if (height == last_height_ && width == last_width_ && plans_valid_) {
        return;
    }

    free_plans();

    last_height_ = height;
    last_width_  = width;
    padded_h_    = height + 2 * params_.padding;
    padded_w_    = width  + 2 * params_.padding;

    int pad_total = padded_h_ * padded_w_;

    // Allocate FFTW-aligned scratch buffers
    fft_in_    = fftwf_alloc_complex(pad_total);
    fft_out_   = fftwf_alloc_complex(pad_total);
    plane_buf_ = fftwf_alloc_complex(pad_total);
    ifft_out_  = fftwf_alloc_complex(pad_total);

    // Create FFTW plans with threading
    fftwf_plan_with_nthreads(params_.num_threads);

    plan_fwd_ = fftwf_plan_dft_2d(padded_h_, padded_w_,
                                   fft_in_, fft_out_,
                                   FFTW_FORWARD, FFTW_MEASURE);

    plan_inv_ = fftwf_plan_dft_2d(padded_h_, padded_w_,
                                   plane_buf_, ifft_out_,
                                   FFTW_BACKWARD, FFTW_MEASURE);

    // Build propagation filter (f2_new)
    f2_new_.resize(pad_total);
    float lambda_over_res = params_.wavelength / eff_resolution_;
    float lambda_over_res_sq = lambda_over_res * lambda_over_res;

    // Build f2_new in DFT order (DC at corner), matching Python's fftshift(f2)
    #pragma omp parallel for num_threads(params_.num_threads)
    for (int idx = 0; idx < pad_total; ++idx) {
        int row = idx / padded_w_;
        int col = idx % padded_w_;

        // Centered frequency coordinates
        float fx = (float)col / (float)padded_w_ - 0.5f;
        float fy = (float)row / (float)padded_h_ - 0.5f;
        float f2 = fx * fx + fy * fy;

        // fftshift: map centered coords to DFT order
        int shifted_row = (row + padded_h_ / 2) % padded_h_;
        int shifted_col = (col + padded_w_ / 2) % padded_w_;
        int shifted_idx = shifted_row * padded_w_ + shifted_col;

        float val = lambda_over_res_sq * f2;
        val = std::max(1.0f - val, 0.0f);
        f2_new_[shifted_idx] = std::sqrt(val);
    }

    // Build z_list
    int planes = params_.num_planes;
    z_list_.resize(planes);
    for (int i = 0; i < planes; ++i) {
        z_list_[i] = eff_z_start_ + i * eff_z_step_;
    }

    plans_valid_ = true;
}

void CPUHolographicReconstructor::reconstruct_intensity(
    const float* hologram, int height, int width, float* output)
{
    omp_set_num_threads(params_.num_threads);
    ensure_initialized(height, width);

    int pad_total = padded_h_ * padded_w_;
    int out_total = height * width;
    int planes    = params_.num_planes;

    // Compute median
    std::vector<float> sorted_input(hologram, hologram + out_total);
    std::nth_element(sorted_input.begin(),
                     sorted_input.begin() + out_total / 2,
                     sorted_input.end());
    float median_val = sorted_input[out_total / 2];

    // Pad hologram (standard, no shift trick)
    #pragma omp parallel for num_threads(params_.num_threads)
    for (int idx = 0; idx < pad_total; ++idx) {
        int row = idx / padded_w_;
        int col = idx % padded_w_;

        float val;
        int src_row = row - params_.padding;
        int src_col = col - params_.padding;
        if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width) {
            val = hologram[src_row * width + src_col];
        } else {
            val = median_val;
        }

        fft_in_[idx][0] = val;
        fft_in_[idx][1] = 0.0f;
    }

    // Forward FFT
    fftwf_execute(plan_fwd_);

    // Intensity buffer: [plane][H][W]
    std::vector<float> intensity(planes * out_total);
    float fft_norm = 1.0f / (float)pad_total;

    // Process each plane
    for (int p = 0; p < planes; ++p) {
        float z = z_list_[p];

        // Apply propagation: Fplane = Fholo * exp(-2j*pi*z*f2_new/wavelength)
        #pragma omp parallel for num_threads(params_.num_threads)
        for (int idx = 0; idx < pad_total; ++idx) {
            float phase = -2.0f * PI * z * f2_new_[idx] / params_.wavelength;
            float cos_p = std::cos(phase);
            float sin_p = std::sin(phase);

            float re = fft_out_[idx][0];
            float im = fft_out_[idx][1];

            plane_buf_[idx][0] = re * cos_p - im * sin_p;
            plane_buf_[idx][1] = re * sin_p + im * cos_p;
        }

        // Inverse FFT
        fftwf_execute(plan_inv_);

        // Phase correction + intensity + remove padding
        float phase_corr = 2.0f * PI * z / params_.wavelength;
        float pc_cos = std::cos(phase_corr);
        float pc_sin = std::sin(phase_corr);

        #pragma omp parallel for num_threads(params_.num_threads)
        for (int idx = 0; idx < out_total; ++idx) {
            int out_row = idx / width;
            int out_col = idx % width;
            int pad_row = out_row + params_.padding;
            int pad_col = out_col + params_.padding;
            int pad_idx = pad_row * padded_w_ + pad_col;

            // Normalize IFFT (FFTW doesn't normalize)
            float re = ifft_out_[pad_idx][0] * fft_norm;
            float im = ifft_out_[pad_idx][1] * fft_norm;

            // Phase correction
            float re2 = re * pc_cos - im * pc_sin;
            float im2 = re * pc_sin + im * pc_cos;

            // Intensity
            intensity[p * out_total + idx] = re2 * re2 + im2 * im2;
        }
    }

    // Find min/max
    float min_val = *std::min_element(intensity.begin(), intensity.end());
    float max_val = *std::max_element(intensity.begin(), intensity.end());
    float range = max_val - min_val + 1e-8f;

    // Normalize to [0,1]
    int total_out = planes * out_total;
    #pragma omp parallel for num_threads(params_.num_threads)
    for (int i = 0; i < total_out; ++i) {
        intensity[i] = (intensity[i] - min_val) / range;
    }

    // Compute mean for shift scaling
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(params_.num_threads)
    for (int i = 0; i < total_out; ++i) {
        sum += intensity[i];
    }
    float mean_bg = (float)(sum / total_out);
    float scale_factor = (params_.shift_mean / 255.0f) / (mean_bg + 1e-8f);

    // Apply shift scaling and clamp
    #pragma omp parallel for num_threads(params_.num_threads)
    for (int i = 0; i < total_out; ++i) {
        float val = intensity[i] * scale_factor;
        intensity[i] = std::min(std::max(val, 0.0f), 1.0f);
    }

    // Transpose from [plane][H][W] to [H][W][plane]
    #pragma omp parallel for num_threads(params_.num_threads)
    for (int idx = 0; idx < total_out; ++idx) {
        int plane = idx % planes;
        int rc = idx / planes;
        int row = rc / width;
        int col = rc % width;
        output[idx] = intensity[plane * out_total + row * width + col];
    }
}
