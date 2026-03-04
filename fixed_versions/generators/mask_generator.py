import numpy as np
import cv2


class MaskGenerator:
    def __init__(self,
                 min_strokes=1, max_strokes=12,
                 min_thickness=10, max_thickness=40,
                 min_box_side=20,
                 max_boxes=3):
        """
        Configures the boundaries for the random SOTA mask generation.
        """
        self.min_strokes = min_strokes
        self.max_strokes = max_strokes
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.min_box_side = min_box_side
        self.max_boxes = max_boxes

    def __call__(self, image, image_id, iteration):
        """
        Generates the mask and applies Gaussian noise to the masked regions.
        Expects `image` as a NumPy array of shape (H, W, C).
        """
        height, width = image.shape[:2]

        # 1. Deterministic Seeding
        seed = int(image_id) * 100 + int(iteration)
        rng = np.random.default_rng(seed)

        # Initialize an empty mask (1 channel)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 2. Generate Random Polygonal Strokes
        num_strokes = rng.integers(self.min_strokes, self.max_strokes + 1)
        for _ in range(num_strokes):
            # A stroke is a sequence of random points (polygonal chain)
            num_points = rng.integers(3, 8)
            points = np.array([[rng.integers(0, width), rng.integers(0, height)]
                               for _ in range(num_points)])

            thickness = rng.integers(self.min_thickness, self.max_thickness)

            # Draw the connected lines
            for i in range(num_points - 1):
                cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), 255, thickness)
                # Draw a circle at the joints to make the stroke perfectly smooth
                cv2.circle(mask, tuple(points[i]), thickness // 2, 255, -1)
                cv2.circle(mask, tuple(points[i + 1]), thickness // 2, 255, -1)

        # 3. Generate Random Rectangular Boxes (Standard SOTA mix)
        num_boxes = rng.integers(0, self.max_boxes + 1)
        for _ in range(num_boxes):
            w = rng.integers(self.min_box_side, width // 2)
            h = rng.integers(self.min_box_side, height // 2)
            x = rng.integers(0, width - w)
            y = rng.integers(0, height - h)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # Normalize mask to [0, 1] for mathematical operations
        mask_normalized = (mask / 255.0).astype(np.float32)
        mask_expanded = np.expand_dims(mask, axis=-1)

        # Calculate Coverage Ratio (percentage of image masked)
        coverage_ratio = np.mean(mask_normalized)
        return mask_expanded, coverage_ratio
