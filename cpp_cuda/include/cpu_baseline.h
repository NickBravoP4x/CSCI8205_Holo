#pragma once

#include <fftw3.h>
#include <vector>
#include <complex>
#include <cstdint>

struct CPUReconParams {
    float resolution   = 0.087f;
    float wavelength   = 405.0f / 1.33f / 1000.0f;
    float z_start      = 0.0f;
    float z_step       = 6.0f;
    int   num_planes   = 3;
    float shift_mean   = 105.0f;
    int   padding      = 32;
    int   num_threads  = 1;   // OpenMP + FFTW threads
};

class CPUHolographicReconstructor {
public:
    using Params = CPUReconParams;

    CPUHolographicReconstructor(const Params& params = Params{});
    ~CPUHolographicReconstructor();

    CPUHolographicReconstructor(const CPUHolographicReconstructor&) = delete;
    CPUHolographicReconstructor& operator=(const CPUHolographicReconstructor&) = delete;

    /// Reconstruct intensity from a float32 hologram (H x W, row-major).
    /// Output: H x W x num_planes float32 array, normalized [0,1].
    void reconstruct_intensity(const float* hologram, int height, int width,
                               float* output);

    /// Change thread count (recreates FFTW plans)
    void set_num_threads(int n);

    int num_planes() const { return params_.num_planes; }
    int num_threads() const { return params_.num_threads; }

private:
    void ensure_initialized(int height, int width);
    void free_plans();

    Params params_;

    // Cached dims
    int last_height_ = 0;
    int last_width_  = 0;
    int padded_h_    = 0;
    int padded_w_    = 0;

    // FFTW plans (single-precision: fftwf)
    fftwf_plan plan_fwd_     = nullptr;
    fftwf_plan plan_inv_     = nullptr;
    bool plans_valid_ = false;

    // Pre-computed filter
    std::vector<float> f2_new_;  // padded_h * padded_w

    // Effective parameters
    float eff_resolution_ = 0;
    float eff_z_start_    = 0;
    float eff_z_step_     = 0;

    // z list
    std::vector<float> z_list_;

    // Scratch buffers (FFTW-aligned, manually managed)
    fftwf_complex* fft_in_   = nullptr;
    fftwf_complex* fft_out_  = nullptr;
    fftwf_complex* plane_buf_= nullptr;
    fftwf_complex* ifft_out_ = nullptr;
};
