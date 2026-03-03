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
