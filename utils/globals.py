import os

wd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(wd, 'data')
COCO_PATH = os.path.join(DATA_PATH, 'coco')
COCO_ANNOTATIONS_PATH = os.path.join(COCO_PATH, 'annotations')
COCO_CAPTIONS_PATH = os.path.join(COCO_ANNOTATIONS_PATH, 'captions_val2017.json')
COCO_INSTANCES_PATH = os.path.join(COCO_ANNOTATIONS_PATH, 'instances_val2017.json')
COCO_IMAGES_PATH = os.path.join(COCO_PATH, 'val2017_subset')
EVAL_IMAGES_PATH = os.path.join(COCO_PATH, 'evaluation')
VALIDATION_IMAGES_PATH = os.path.join(COCO_PATH, 'validation')
OUR_DATASET_PATH = os.path.join(DATA_PATH, 'our_dataset')
OUR_DATASET_CAPTIONS_PATH = os.path.join(OUR_DATASET_PATH, 'captions.json')

TO_COVERAGE_RATIO = {i: f'{i}-{i + 10}' for i in range(0, 91, 10)}
FROM_COVERAGE_RATIO = {f'{i}-{i + 10}': i for i in range(0, 91, 10)}

MASKING_CONFIGS = {
    "min_lines": 2, "max_lines": 5,
    "min_thickness": 0.03, "max_thickness": 0.05,
    "min_line_length": 0.1, "max_line_length": 0.25,
    "min_rectangles": 2, "max_rectangles": 4,
    "min_rect_side": 0.1, "max_rect_side": 0.25,
    "min_circles": 1, "max_circles": 4,
    "min_radius": 0.05, "max_radius": 0.125
}
