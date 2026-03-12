import numpy as np
import cv2
import math


class MaskGenerator:
    """
    Generates masks for images based on specified parameters reproducibly using image-specific seeds.
    """

    def __init__(self,
                 min_lines=1, max_lines=4,
                 min_thickness=5, max_thickness=20,
                 min_line_length=30, max_line_length=150,
                 min_rectangles=0, max_rectangles=2,
                 min_rect_side=20, max_rect_side=60,
                 min_circles=0, max_circles=2,
                 min_radius=15, max_radius=40):
        """
        Configures the boundaries for a controlled, randomized mask generation.
        :param min_lines: Minimum number of lines to generate.
        :param max_lines: Maximum number of lines to generate.
        :param min_thickness: Minimum thickness of lines in pixels.
        :param max_thickness: Maximum thickness of lines in pixels.
        :param min_line_length: Minimum length of lines in pixels.
        :param max_line_length: Maximum length of lines in pixels.
        :param min_rectangles: Minimum number of rectangles to generate.
        :param max_rectangles: Maximum number of rectangles to generate.
        :param min_rect_side: Minimum side length of rectangles in pixels.
        :param max_rect_side: Maximum side length of rectangles in pixels.
        :param min_circles: Minimum number of circles to generate.
        :param max_circles: Maximum number of circles to generate.
        :param min_radius: Minimum radius of circles in pixels.
        :param max_radius: Maximum radius of circles in pixels.
        """
        # line configurations
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length

        # rectangle configurations
        self.min_rectangles = min_rectangles
        self.max_rectangles = max_rectangles
        self.min_rect_side = min_rect_side
        self.max_rect_side = max_rect_side

        # circle configurations
        self.min_circles = min_circles
        self.max_circles = max_circles
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image, image_id: int):
        """
        Generates the mask of a specific image.
        :param image: The image to generate the mask for.
        :param image_id: The ID of the image.
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

        rng = np.random.default_rng(int(image_id))

        mask = np.zeros((height, width), dtype=np.uint8)

        # generate random lines
        num_lines = rng.integers(self.min_lines, self.max_lines + 1)
        for _ in range(num_lines):
            num_segments = rng.integers(1, 3) 
            thickness = rng.integers(min_thickness, max_thickness + 1)
            
            x0 = rng.integers(0, width)
            y0 = rng.integers(0, height)
            for _ in range(num_segments):
                length = rng.uniform(min_line_length, max_line_length)

                while True:                    
                    angle = rng.uniform(0, 2 * math.pi)
                    # no lines in 10 degrees from the original line
                    if 2 * math.pi - math.pi / 18 < angle < 2 * math.pi + math.pi / 18:
                        continue
                    x1 = int(x0 + length * math.cos(angle))
                    y1 = int(y0 + length * math.sin(angle))
                    # prevent lines from going outside the image
                    if 0 <= x1 <= width and 0 <= y1 <= height:
                        break

                cv2.line(mask, (x0, y0), (x1, y1), 255, thickness)
                cv2.circle(mask, (x0, y0), thickness // 2, 255, -1)
                cv2.circle(mask, (x1, y1), thickness // 2, 255, -1)
                x0, y0 = x1, y1

        # generate random rectangles
        num_rects = rng.integers(self.min_rectangles, self.max_rectangles + 1)
        for _ in range(num_rects):
            w = rng.integers(min_rect_side, min(max_rect_side, max(min_rect_side + 1, width // 2)))
            h = rng.integers(min_rect_side, min(max_rect_side, max(min_rect_side + 1, height // 2)))
            x = rng.integers(0, max(1, width - w))
            y = rng.integers(0, max(1, height - h))
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # generate random circles
        num_circles = rng.integers(self.min_circles, self.max_circles + 1)
        for _ in range(num_circles):
            r = rng.integers(min_radius, max_radius + 1)
            x = rng.integers(0, width)
            y = rng.integers(0, height)
            cv2.circle(mask, (x, y), r, 255, -1)
        
        coverage_ratio = cv2.countNonZero(mask) / mask.size
        return mask, coverage_ratio
        