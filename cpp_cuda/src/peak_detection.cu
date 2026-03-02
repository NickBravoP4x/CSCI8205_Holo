#include "pipeline_stages.h"
#include "holographic_reconstruction.h"  // for CUDA_CHECK

#include <stdexcept>
#include <algorithm>

static constexpr int BLOCK = 256;

// ─── ImageNet normalization constants ────────────────────────────────────────
// scale[c] = 1.0 / std[c],  offset[c] = -mean[c] / std[c]
// mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225]
__constant__ float c_imagenet_scale[3]  = { 1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f };
__constant__ float c_imagenet_offset[3] = { -0.485f/0.229f, -0.456f/0.224f, -0.406f/0.225f };

// ─── Kernel: ImageNet normalize [C][H][W] → [1][C][H][W] ───────────────────
__global__ void imagenet_normalize_kernel(
    const float* __restrict__ input,   // [C][H][W] in [0,1]
    float* __restrict__ output,        // [C][H][W] ImageNet-normalized
    int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * HW;
    if (idx >= total) return;

    int c = idx / HW;
    output[idx] = input[idx] * c_imagenet_scale[c] + c_imagenet_offset[c];
}

void imagenet_normalize(const float* d_input, float* d_output,
                        int C, int H, int W, cudaStream_t stream)
{
    int total = C * H * W;
    int grid = (total + BLOCK - 1) / BLOCK;
    imagenet_normalize_kernel<<<grid, BLOCK, 0, stream>>>(
        d_input, d_output, C, H * W);
    CUDA_CHECK(cudaGetLastError());
}

// ─── Kernel: Max-pool NMS on 2D heatmap ─────────────────────────────────────
// Replicate-padded max pool with kernel_size, stride=1
// Output: peak mask (1.0 where pixel == local max AND > threshold, else 0.0)
__global__ void maxpool_nms_kernel(
    const float* __restrict__ heatmap,  // [H][W]
    float* __restrict__ peaks_val,      // [H][W] peak values (0 if not peak)
    int H, int W,
    int half_k,     // kernel_size / 2
    float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int row = idx / W;
    int col = idx % W;
    float val = heatmap[idx];

    if (val <= threshold) {
        peaks_val[idx] = 0.0f;
        return;
    }

    // Find local max in kernel window with replicate padding
    float local_max = -1e30f;
    for (int dy = -half_k; dy <= half_k; ++dy) {
        for (int dx = -half_k; dx <= half_k; ++dx) {
            int ny = min(max(row + dy, 0), H - 1);  // replicate pad
            int nx = min(max(col + dx, 0), W - 1);
            float neighbor = heatmap[ny * W + nx];
            local_max = fmaxf(local_max, neighbor);
        }
    }

    // Peak if we're the local maximum
    peaks_val[idx] = (val >= local_max) ? val : 0.0f;
}

// ─── Kernel: Extract non-zero peaks with atomic counter ─────────────────────
struct RawPeak {
    float val;
    int idx;
};

__global__ void extract_peaks_kernel(
    const float* __restrict__ peaks_val, // [H*W]
    RawPeak* __restrict__ out_peaks,     // [max_peaks]
    int* __restrict__ counter,           // atomic counter
    int total,
    int max_peaks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float val = peaks_val[idx];
    if (val > 0.0f) {
        int pos = atomicAdd(counter, 1);
        if (pos < max_peaks) {
            out_peaks[pos].val = val;
            out_peaks[pos].idx = idx;
        }
    }
}

// ─── Host function: detect peaks ─────────────────────────────────────────────
std::vector<Peak> detect_peaks(const float* d_heatmap, int H, int W,
                               float threshold, int min_distance, int topk,
                               float output_scale, cudaStream_t stream)
{
    int total = H * W;
    int half_k = min_distance;  // kernel half-size (full = 2*min_distance+1)
    int max_peaks = topk * 4;   // over-allocate for atomics

    // Allocate temporary buffers
    float* d_peaks_val;
    RawPeak* d_raw_peaks;
    int* d_counter;

    CUDA_CHECK(cudaMalloc(&d_peaks_val, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_raw_peaks, max_peaks * sizeof(RawPeak)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));

    // Max-pool NMS
    int grid1 = (total + BLOCK - 1) / BLOCK;
    maxpool_nms_kernel<<<grid1, BLOCK, 0, stream>>>(
        d_heatmap, d_peaks_val, H, W, half_k, threshold);
    CUDA_CHECK(cudaGetLastError());

    // Extract peaks
    extract_peaks_kernel<<<grid1, BLOCK, 0, stream>>>(
        d_peaks_val, d_raw_peaks, d_counter, total, max_peaks);
    CUDA_CHECK(cudaGetLastError());

    // Copy counter and peaks to host
    int num_peaks;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&num_peaks, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    num_peaks = std::min(num_peaks, max_peaks);

    std::vector<RawPeak> h_peaks(num_peaks);
    if (num_peaks > 0) {
        CUDA_CHECK(cudaMemcpy(h_peaks.data(), d_raw_peaks,
                               num_peaks * sizeof(RawPeak), cudaMemcpyDeviceToHost));
    }

    // Sort descending by value
    std::sort(h_peaks.begin(), h_peaks.end(),
              [](const RawPeak& a, const RawPeak& b) { return a.val > b.val; });

    // Convert to output format, take top-k
    int k = std::min(num_peaks, topk);
    float scale_x = output_scale / (float)W;
    float scale_y = output_scale / (float)H;

    std::vector<Peak> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i) {
        if (h_peaks[i].val <= 0.0f) break;
        int row = h_peaks[i].idx / W;
        int col = h_peaks[i].idx % W;
        result.push_back({
            col * scale_x,
            row * scale_y,
            h_peaks[i].val
        });
    }

    // Cleanup
    cudaFree(d_peaks_val);
    cudaFree(d_raw_peaks);
    cudaFree(d_counter);

    return result;
}
