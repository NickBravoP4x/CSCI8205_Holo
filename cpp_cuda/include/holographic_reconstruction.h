#pragma once

#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// Check CUDA errors
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#define CUFFT_CHECK(call)                                                     \
    do {                                                                      \
        cufftResult err = (call);                                             \
        if (err != CUFFT_SUCCESS) {                                           \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)err);                                      \
            throw std::runtime_error("cuFFT error");                          \
        }                                                                     \
    } while (0)

// Construction parameters matching Python defaults
struct CUDAReconParams {
    float resolution   = 0.087f;               // microns/pixel
    float wavelength   = 405.0f / 1.33f / 1000.0f; // wavelength in microns
    float z_start      = 0.0f;                 // starting depth (microns)
    float z_step       = 6.0f;                 // depth step (microns)
    int   num_planes   = 3;                    // reconstruction planes
    float shift_mean   = 105.0f;               // background shift
    int   padding      = 32;                   // edge padding pixels
    int   gpu_id       = 0;                    // GPU device
};

class CUDAHolographicReconstructor {
public:
    using Params = CUDAReconParams;

    CUDAHolographicReconstructor(const Params& params = Params{});
    ~CUDAHolographicReconstructor();

    // Non-copyable
    CUDAHolographicReconstructor(const CUDAHolographicReconstructor&) = delete;
    CUDAHolographicReconstructor& operator=(const CUDAHolographicReconstructor&) = delete;

    /// Reconstruct intensity from a float32 hologram (H x W, row-major).
    /// Output: H x W x num_planes float32 array, normalized [0,1].
    void reconstruct_intensity(const float* h_hologram, int height, int width,
                               float* h_output);

    /// GPU-only reconstruction (no H2D/D2H transfers).
    /// d_hologram_ must already contain data (via upload_hologram).
    /// Result is left in d_output_ (retrieve via download_output).
    void reconstruct_intensity_device(int height, int width);

    /// Upload hologram from host to internal device buffer.
    void upload_hologram(const float* h_hologram, int height, int width);

    /// Download output from internal device buffer to host.
    void download_output(float* h_output, int height, int width);

    /// Returns the number of depth planes
    int num_planes() const { return params_.num_planes; }

    /// Access internal device intensity buffer [num_planes][H][W] (NCHW layout)
    const float* device_intensity_nchw() const { return d_intensity_; }

    /// Access internal device hologram input buffer [H][W]
    /// Must call prepare(H, W) first to ensure buffer is allocated.
    float* device_hologram_ptr() { return d_hologram_; }

    /// Pre-allocate buffers for given dimensions (call before device_hologram_ptr)
    void prepare(int height, int width) { ensure_initialized(height, width); }

private:
    void ensure_initialized(int height, int width);
    void free_buffers();

    Params params_;

    // Cached dimensions
    int last_height_ = 0;
    int last_width_  = 0;
    int padded_h_    = 0;
    int padded_w_    = 0;
    int actual_pad_h_ = 0;  // (padded_h - height) / 2
    int actual_pad_w_ = 0;  // (padded_w - width)  / 2

    // cuFFT plans
    cufftHandle plan_fwd_     = 0;  // single 2D C2C forward
    cufftHandle plan_inv_bat_ = 0;  // batched 2D C2C inverse (num_planes batched)
    bool plans_valid_ = false;

    // Device buffers
    float*       d_hologram_    = nullptr;  // H x W input
    float*       d_padded_real_ = nullptr;  // Hp x Wp (after pad, before fftshift trick)
    cufftComplex* d_padded_     = nullptr;  // Hp x Wp complex (input to fwd FFT)
    cufftComplex* d_Fholo_      = nullptr;  // Hp x Wp complex (fwd FFT output)
    float*       d_f2_new_      = nullptr;  // Hp x Wp propagation filter (precomputed)
    float*       d_z_list_      = nullptr;  // num_planes z values
    cufftComplex* d_Hz_stack_   = nullptr;  // num_planes x Hp x Wp (cached propagation)
    cufftComplex* d_Fplanes_    = nullptr;  // num_planes x Hp x Wp (propagated spectra)
    cufftComplex* d_rec_        = nullptr;  // num_planes x Hp x Wp (IFFT output)
    float*       d_intensity_   = nullptr;  // num_planes x H x W (output intensity)
    float*       d_output_      = nullptr;  // H x W x num_planes (transposed output)

    // Reduction workspace
    float*       d_reduce_buf_  = nullptr;  // for min/max reduction

    // Effective parameters (after resolution scaling)
    float eff_resolution_ = 0;
    float eff_z_start_    = 0;
    float eff_z_step_     = 0;
};
