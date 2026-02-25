#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "holographic_reconstruction.h"
#include "cpu_baseline.h"

namespace py = pybind11;

// ─── CUDA Reconstructor Python wrapper ──────────────────────────────────────
class PyCUDAReconstructor {
public:
    PyCUDAReconstructor(
        float resolution = 0.087f,
        float wavelength = 405.0f / 1.33f / 1000.0f,
        float z_start = 0.0f,
        float z_step = 6.0f,
        int num_planes = 3,
        float shift_mean = 105.0f,
        int padding = 32,
        int gpu_id = 0)
    {
        CUDAHolographicReconstructor::Params p;
        p.resolution = resolution;
        p.wavelength = wavelength;
        p.z_start    = z_start;
        p.z_step     = z_step;
        p.num_planes = num_planes;
        p.shift_mean = shift_mean;
        p.padding    = padding;
        p.gpu_id     = gpu_id;
        impl_ = std::make_unique<CUDAHolographicReconstructor>(p);
    }

    py::array_t<float> reconstruct_intensity(py::array_t<float, py::array::c_style> hologram) {
        auto buf = hologram.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D (H x W)");
        }
        int H = (int)buf.shape[0];
        int W = (int)buf.shape[1];
        int planes = impl_->num_planes();

        // Allocate output
        py::array_t<float> output({H, W, planes});
        auto out_buf = output.request();

        impl_->reconstruct_intensity(
            static_cast<const float*>(buf.ptr), H, W,
            static_cast<float*>(out_buf.ptr));

        return output;
    }

    int num_planes() const { return impl_->num_planes(); }

private:
    std::unique_ptr<CUDAHolographicReconstructor> impl_;
};

// ─── CPU Reconstructor Python wrapper ───────────────────────────────────────
class PyCPUReconstructor {
public:
    PyCPUReconstructor(
        float resolution = 0.087f,
        float wavelength = 405.0f / 1.33f / 1000.0f,
        float z_start = 0.0f,
        float z_step = 6.0f,
        int num_planes = 3,
        float shift_mean = 105.0f,
        int padding = 32,
        int num_threads = 1)
    {
        CPUHolographicReconstructor::Params p;
        p.resolution  = resolution;
        p.wavelength  = wavelength;
        p.z_start     = z_start;
        p.z_step      = z_step;
        p.num_planes  = num_planes;
        p.shift_mean  = shift_mean;
        p.padding     = padding;
        p.num_threads = num_threads;
        impl_ = std::make_unique<CPUHolographicReconstructor>(p);
    }

    py::array_t<float> reconstruct_intensity(py::array_t<float, py::array::c_style> hologram) {
        auto buf = hologram.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D (H x W)");
        }
        int H = (int)buf.shape[0];
        int W = (int)buf.shape[1];
        int planes = impl_->num_planes();

        py::array_t<float> output({H, W, planes});
        auto out_buf = output.request();

        impl_->reconstruct_intensity(
            static_cast<const float*>(buf.ptr), H, W,
            static_cast<float*>(out_buf.ptr));

        return output;
    }

    void set_num_threads(int n) { impl_->set_num_threads(n); }
    int num_planes() const { return impl_->num_planes(); }
    int num_threads() const { return impl_->num_threads(); }

private:
    std::unique_ptr<CPUHolographicReconstructor> impl_;
};

// ─── Module definition ──────────────────────────────────────────────────────
PYBIND11_MODULE(cuda_holographic, m) {
    m.doc() = "C++/CUDA and OpenMP holographic reconstruction";

    py::class_<PyCUDAReconstructor>(m, "CUDAReconstructor",
        "CUDA-accelerated holographic reconstruction using cuFFT")
        .def(py::init<float, float, float, float, int, float, int, int>(),
             py::arg("resolution") = 0.087f,
             py::arg("wavelength") = 405.0f / 1.33f / 1000.0f,
             py::arg("z_start") = 0.0f,
             py::arg("z_step") = 6.0f,
             py::arg("num_planes") = 3,
             py::arg("shift_mean") = 105.0f,
             py::arg("padding") = 32,
             py::arg("gpu_id") = 0)
        .def("reconstruct_intensity", &PyCUDAReconstructor::reconstruct_intensity,
             py::arg("hologram"),
             "Reconstruct intensity volume from 2D hologram. Returns [H,W,num_planes] float32.")
        .def_property_readonly("num_planes", &PyCUDAReconstructor::num_planes);

    py::class_<PyCPUReconstructor>(m, "CPUReconstructor",
        "CPU holographic reconstruction using FFTW + OpenMP")
        .def(py::init<float, float, float, float, int, float, int, int>(),
             py::arg("resolution") = 0.087f,
             py::arg("wavelength") = 405.0f / 1.33f / 1000.0f,
             py::arg("z_start") = 0.0f,
             py::arg("z_step") = 6.0f,
             py::arg("num_planes") = 3,
             py::arg("shift_mean") = 105.0f,
             py::arg("padding") = 32,
             py::arg("num_threads") = 1)
        .def("reconstruct_intensity", &PyCPUReconstructor::reconstruct_intensity,
             py::arg("hologram"),
             "Reconstruct intensity volume from 2D hologram. Returns [H,W,num_planes] float32.")
        .def("set_num_threads", &PyCPUReconstructor::set_num_threads,
             py::arg("n"), "Change thread count (recreates FFTW plans)")
        .def_property_readonly("num_planes", &PyCPUReconstructor::num_planes)
        .def_property_readonly("num_threads", &PyCPUReconstructor::num_threads);
}
