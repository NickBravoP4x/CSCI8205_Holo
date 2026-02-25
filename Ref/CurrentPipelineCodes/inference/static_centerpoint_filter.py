"""
Static Centerpoint Filter - Removes persistent false positive detections for heatmap detection.

Uses Euclidean distance tracking to identify centerpoints that appear repeatedly
in the same location across multiple frames.
"""
from collections import deque
from typing import Tuple
import numpy as np


class StaticCenterpointFilter:
    """
    Filters out static points (dust, sensor artifacts) from heatmap detections.

    Uses Euclidean distance tracking to identify centerpoints that appear
    repeatedly in the same location across multiple frames.
    """

    def __init__(
        self,
        max_history: int = 200,
        distance_threshold: float = 50.0,
        min_hits: int = 5
    ):
        """
        Args:
            max_history: Maximum number of points to track in history
            distance_threshold: Distance threshold for matching points (pixels)
            min_hits: Minimum hits before filtering (points with >= min_hits are removed)
        """
        self.history = deque(maxlen=max_history)
        self.distance_threshold = distance_threshold
        self.min_hits = min_hits

    def update_and_filter(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update history and filter out persistent detections.

        Args:
            points: Array of shape (N, 2) containing (x, y) centerpoints

        Returns:
            Tuple of:
                - filtered_points: Array of shape (M, 2) with static points removed
                - mask: Boolean array of shape (N,) where True = kept, False = filtered
        """
        if len(points) == 0:
            return points, np.array([], dtype=bool)

        # Ensure points is a 2D numpy array
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Build history array for vectorized distance computation
        if len(self.history) > 0:
            history_array = np.array(list(self.history))
        else:
            history_array = np.empty((0, 2))

        # Add current points to history
        for point in points:
            self.history.append(point.copy())

        # If no history to compare against, keep all points
        if len(history_array) == 0:
            mask = np.ones(len(points), dtype=bool)
            return points, mask

        # Compute distances from each point to all history points
        # points: (N, 2), history: (H, 2)
        # distances: (N, H)
        diff = points[:, np.newaxis, :] - history_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        # Count hits for each point (number of history points within threshold)
        hits = np.sum(distances < self.distance_threshold, axis=1)

        # Keep points with hits below threshold
        mask = hits < self.min_hits

        return points[mask], mask

    def reset(self):
        """Clear the history buffer."""
        self.history.clear()
