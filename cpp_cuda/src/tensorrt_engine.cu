#include "tensorrt_engine.h"
#include "holographic_reconstruction.h"  // for CUDA_CHECK

#include <fstream>
#include <numeric>
#include <functional>

// ─── Load serialized engine from file ────────────────────────────────────────
void TensorRTEngine::load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open engine file: " + path);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine: " + path);
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
}

// ─── Constructor ─────────────────────────────────────────────────────────────
TensorRTEngine::TensorRTEngine(const std::string& engine_path, int gpu_id)
    : gpu_id_(gpu_id)
{
    CUDA_CHECK(cudaSetDevice(gpu_id_));
    load_engine(engine_path);

    // Discover I/O tensor names and shapes
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);

        // Compute element count (skip batch dim if dynamic, i.e. -1)
        int64_t elem = 1;
        std::vector<int64_t> shape;
        for (int d = 0; d < dims.nbDims; ++d) {
            shape.push_back(dims.d[d]);
            if (dims.d[d] > 0) elem *= dims.d[d];
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            input_dims_ = shape;
            input_size_ = elem;
        } else {
            output_name_ = name;
            output_dims_ = shape;
            output_size_ = elem;
        }
    }

    if (input_name_.empty() || output_name_.empty()) {
        throw std::runtime_error("Engine must have exactly 1 input and 1 output");
    }
}

// ─── Destructor ──────────────────────────────────────────────────────────────
TensorRTEngine::~TensorRTEngine() {
    if (d_warmup_in_)  cudaFree(d_warmup_in_);
    if (d_warmup_out_) cudaFree(d_warmup_out_);
}

// ─── Inference ───────────────────────────────────────────────────────────────
void TensorRTEngine::infer(const float* d_input, float* d_output, int batch,
                           cudaStream_t stream)
{
    CUDA_CHECK(cudaSetDevice(gpu_id_));

    // Set input shape — resolve any dynamic dimensions
    auto in_dims = engine_->getTensorShape(input_name_.c_str());
    bool has_dynamic = false;
    for (int d = 0; d < in_dims.nbDims; ++d) {
        if (in_dims.d[d] == -1) has_dynamic = true;
    }
    if (has_dynamic) {
        // Set batch dimension
        if (in_dims.d[0] == -1) in_dims.d[0] = batch;
        // For dynamic spatial dims, use the opt profile values
        // (they were set during engine build)
        auto profile = engine_->getProfileShape(
            input_name_.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
        for (int d = 1; d < in_dims.nbDims; ++d) {
            if (in_dims.d[d] == -1) in_dims.d[d] = profile.d[d];
        }
        context_->setInputShape(input_name_.c_str(), in_dims);
    }

    // Set tensor addresses (TRT 10.x API)
    context_->setTensorAddress(input_name_.c_str(), const_cast<float*>(d_input));
    context_->setTensorAddress(output_name_.c_str(), d_output);

    // Execute
    bool ok = context_->enqueueV3(stream);
    if (!ok) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
}

// ─── Warmup ──────────────────────────────────────────────────────────────────
void TensorRTEngine::warmup(int n) {
    CUDA_CHECK(cudaSetDevice(gpu_id_));

    // Allocate scratch buffers if needed
    if (!d_warmup_in_) {
        CUDA_CHECK(cudaMalloc(&d_warmup_in_, input_size_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_warmup_in_, 0, input_size_ * sizeof(float)));
    }
    if (!d_warmup_out_) {
        CUDA_CHECK(cudaMalloc(&d_warmup_out_, output_size_ * sizeof(float)));
    }

    for (int i = 0; i < n; ++i) {
        infer(d_warmup_in_, d_warmup_out_, 1, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
