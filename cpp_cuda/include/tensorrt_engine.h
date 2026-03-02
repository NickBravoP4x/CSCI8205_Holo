#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdio>

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& engine_path, int gpu_id = 0);
    ~TensorRTEngine();

    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    /// Run inference: d_input and d_output must be device pointers.
    void infer(const float* d_input, float* d_output, int batch = 1,
               cudaStream_t stream = nullptr);

    /// Warmup N iterations to stabilize GPU clocks and trigger JIT
    void warmup(int n = 20);

    /// Get binding dimensions (excludes batch for dynamic batch engines)
    std::vector<int64_t> input_dims() const { return input_dims_; }
    std::vector<int64_t> output_dims() const { return output_dims_; }

    /// Total element counts (per-sample, excluding batch)
    int64_t input_size() const { return input_size_; }
    int64_t output_size() const { return output_size_; }

    /// Input/output tensor names
    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }

private:
    void load_engine(const std::string& path);

    int gpu_id_;
    TRTLogger logger_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_dims_;
    std::vector<int64_t> output_dims_;
    int64_t input_size_ = 0;   // elements per sample
    int64_t output_size_ = 0;

    // Warmup buffers (allocated once)
    float* d_warmup_in_ = nullptr;
    float* d_warmup_out_ = nullptr;
};
