"""
Laser Spot Filter - Removes persistent false positive detections
"""
from collections import deque
from typing import List, Tuple


class LaserSpotFilter:
    """
    Filters out laser spots and other persistent false positives.

    Uses IoU (Intersection over Union) tracking to identify bounding boxes
    that appear repeatedly in the same location across multiple frames.
    """

    def __init__(
        self,
        max_history: int = 100,
        iou_threshold: float = 0.8,
        min_hits: int = 10
    ):
        """
        Args:
            max_history: Maximum number of frames to track
            iou_threshold: IoU threshold for matching boxes (0.8 = 80% overlap)
            min_hits: Minimum hits before filtering (boxes seen >= min_hits are removed)
        """
        self.history = deque(maxlen=max_history)
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    def iou(
        self,
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate Intersection over Union between two boxes.

        Args:
            boxA: [x1, y1, x2, y2]
            boxB: [x1, y1, x2, y2]

        Returns:
            IoU value in range [0, 1]
        """
        # Intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Intersection area
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        # Box areas
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Union area
        union_area = boxA_area + boxB_area - inter_area + 1e-6

        return inter_area / union_area

    def update_and_filter(
        self,
        boxes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Update history and filter out persistent detections.

        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]

        Returns:
            Filtered list of boxes (persistent ones removed)
        """
        # Add all boxes to history
        self.history.extend(boxes)

        # Filter boxes based on hit count
        valid_boxes = []
        for box in boxes:
            # Count how many times this box appears in history
            hits = sum(
                1 for prev_box in self.history
                if self.iou(box, prev_box) > self.iou_threshold
            )

            # Keep only if hit count is below threshold
            if hits < self.min_hits:
                valid_boxes.append(box)

        return valid_boxes

    def reset(self):
        """Clear the history buffer."""
        self.history.clear()
