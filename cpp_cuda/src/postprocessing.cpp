#include "pipeline_stages.h"

#include <cmath>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// StaticCenterpointFilter implementation
// ═══════════════════════════════════════════════════════════════════════════════

StaticCenterpointFilter::StaticCenterpointFilter(
    int max_history, float distance_threshold, int min_hits)
    : max_history_(max_history),
      distance_threshold_(distance_threshold),
      min_hits_(min_hits)
{
    history_x_.resize(max_history);
    history_y_.resize(max_history);
}

void StaticCenterpointFilter::reset() {
    history_size_ = 0;
    history_start_ = 0;
}

std::vector<bool> StaticCenterpointFilter::update_and_filter(
    const std::vector<Detection>& detections)
{
    std::vector<bool> mask(detections.size(), true);

    if (detections.empty()) {
        return mask;
    }

    // Compare each detection against history
    float dist_sq_thresh = distance_threshold_ * distance_threshold_;

    for (size_t i = 0; i < detections.size(); ++i) {
        float px = detections[i].x;
        float py = detections[i].y;
        int hits = 0;

        // Check against all history points
        for (int j = 0; j < history_size_; ++j) {
            int idx = (history_start_ + j) % max_history_;
            float dx = px - history_x_[idx];
            float dy = py - history_y_[idx];
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < dist_sq_thresh) {
                hits++;
                if (hits >= min_hits_) break;  // early exit
            }
        }

        mask[i] = (hits < min_hits_);
    }

    // Add current detections to history ring buffer
    for (size_t i = 0; i < detections.size(); ++i) {
        int write_idx;
        if (history_size_ < max_history_) {
            write_idx = history_size_;
            history_size_++;
        } else {
            write_idx = history_start_;
            history_start_ = (history_start_ + 1) % max_history_;
        }
        history_x_[write_idx] = detections[i].x;
        history_y_[write_idx] = detections[i].y;
    }

    return mask;
}
