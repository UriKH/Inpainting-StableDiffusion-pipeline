import numpy as np
import cv2
import math

class MaskGenerator:
    def __init__(self,
                 # Line configuration
                 min_lines=1, max_lines=4,
                 min_thickness=5, max_thickness=20,
                 min_line_length=30, max_line_length=150,
                 # Rectangle configuration
                 min_rectangles=0, max_rectangles=2,
                 min_rect_side=20, max_rect_side=60,
                 # Circle configuration
                 min_circles=0, max_circles=2,
                 min_radius=15, max_radius=40):
        """
        Configures the boundaries for a controlled, randomized mask generation.
        """
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        
        self.min_rectangles = min_rectangles
        self.max_rectangles = max_rectangles
        self.min_rect_side = min_rect_side
        self.max_rect_side = max_rect_side
        
        self.min_circles = min_circles
        self.max_circles = max_circles
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image, image_id):
        """
        Generates the mask.
        Expects `image` as a NumPy array of shape (H, W, C).
        """
        height, width = image.shape[:2]
        d = (width + height) // 2
        
        min_thickness = self.min_thickness * d
        max_thickness = self.max_thickness * d
        min_line_length = self.min_line_length * d
        max_line_length = self.max_line_length * d
        min_rect_side = self.min_rect_side * d
        max_rect_side = self.max_rect_side * d
        min_radius = self.min_radius * d
        max_radius = self.max_radius * d

        # 1. Deterministic Seeding
        seed = int(image_id)
        rng = np.random.default_rng(seed)

        # Initialize an empty mask (1 channel)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 2. Generate Lines (At most 1 joint means 1 or 2 line segments)
        num_lines = rng.integers(self.min_lines, self.max_lines + 1)
        for _ in range(num_lines):
            # 1 segment = straight line (0 joints), 2 segments = angled line (1 joint)
            num_segments = rng.integers(1, 3) 
            thickness = rng.integers(min_thickness, max_thickness + 1)
            
            # Pick a random starting point
            x0 = rng.integers(0, width)
            y0 = rng.integers(0, height)
            
            for _ in range(num_segments):
                # Generate a line using an angle and restricted length
                length = rng.uniform(min_line_length, max_line_length)

                while True:                    
                    # Calculate the end point of this segment
                    angle = rng.uniform(0, 2 * math.pi)
                    if 2 * math.pi - 0.175 < angle < 2 * math.pi + 0.175:
                        continue
                        
                    x1 = int(x0 + length * math.cos(angle))
                    y1 = int(y0 + length * math.sin(angle))

                    if 0 <= x1 <= width and 0 <= y1 <= height:
                        break
                    
                # Draw the line and round the joints for smoothness
                cv2.line(mask, (x0, y0), (x1, y1), 255, thickness)
                cv2.circle(mask, (x0, y0), thickness // 2, 255, -1)
                cv2.circle(mask, (x1, y1), thickness // 2, 255, -1)
                
                # The next segment (if any) starts where this one ended
                x0, y0 = x1, y1

        # 3. Generate Random Rectangular Boxes
        num_rects = rng.integers(self.min_rectangles, self.max_rectangles + 1)
        for _ in range(num_rects):
            # Safe bounds to prevent errors on very small images
            w = rng.integers(min_rect_side, min(max_rect_side, max(min_rect_side + 1, width // 2)))
            h = rng.integers(min_rect_side, min(max_rect_side, max(min_rect_side + 1, height // 2)))
            x = rng.integers(0, max(1, width - w))
            y = rng.integers(0, max(1, height - h))
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # 4. Generate Random Circles
        num_circles = rng.integers(self.min_circles, self.max_circles + 1)
        for _ in range(num_circles):
            r = rng.integers(min_radius, max_radius + 1)
            x = rng.integers(0, width)
            y = rng.integers(0, height)
            cv2.circle(mask, (x, y), r, 255, -1)
        
        coverage_ratio = cv2.countNonZero(mask) / mask.size
        return mask, coverage_ratio
        