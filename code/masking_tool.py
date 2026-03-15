"""
Some of the code here was written with the assistance of AI
"""

import sys
import os.path
import cv2 as cv
import numpy as np
import argparse

MAX_SCREEN_SIZE = 800
BRUSH_SIZE = 20
drawing = False
mask = None
display_img = None


def draw_mask(event, x, y, flags, param):
    """
    Draws a circle on the mask image according to the mouse event.
    """
    global drawing, mask, display_img
    brush_size = BRUSH_SIZE

    match event:
        case cv.EVENT_LBUTTONDOWN:
            drawing = True
            cv.circle(mask, (x, y), brush_size, (255, 255, 255), -1)
            cv.circle(display_img, (x, y), brush_size, (0, 255, 0), -1)
        case cv.EVENT_MOUSEMOVE:
            if drawing:
                cv.circle(mask, (x, y), brush_size, (255, 255, 255), -1)
                cv.circle(display_img, (x, y), brush_size, (0, 255, 0), -1)
        case cv.EVENT_LBUTTONUP:
            drawing = False
            cv.circle(mask, (x, y), brush_size, (255, 255, 255), -1)
            cv.circle(display_img, (x, y), brush_size, (0, 255, 0), -1)
        case _: pass


def crop_image(img):
    """
    Crop the image to a square if it's not already.
    :param img: The image to crop.
    :return: The cropped image.
    """
    h, w = img.shape[:2]

    if max(h, w) > MAX_SCREEN_SIZE:
        scale = MAX_SCREEN_SIZE / max(h, w)
        img = cv.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]

    if h == w:
        return img

    print(
        """\n--- CROP INSTRUCTIONS ---
        * Image is not a square. Click and drag to select a region :)
        * Press ENTER or SPACE to confirm your crop.
        * Press 'c' to cancel and let it auto-center-crop.
        """
    )

    roi = cv.selectROI("Crop Image", img, fromCenter=False, showCrosshair=True)
    cv.destroyWindow("Crop Image")
    x, y, roi_w, roi_h = roi

    if roi_w > 0 and roi_h > 0:
        side = min(roi_w, roi_h)
        img = img[y:y + side, x:x + side]
        print("Cropped successfully!")
        return img

    print("Auto-center-cropping to a square...")
    side = min(h, w)
    start_x = w // 2 - side // 2
    start_y = h // 2 - side // 2
    return img[start_y:start_y + side, start_x:start_x + side]


def run_masking_tool(image_path: str, output_dir: str):
    """
    Run the masking tool interactively.
    :param image_path: Path to the input image.
    :param output_dir: Path to the output directory.
    """
    global mask, display_img

    img = cv.imread(image_path)
    if img is None:
        raise Exception(f"Error: Could not load {image_path}. Check the filename!")

    img = crop_image(img)
    img = cv.resize(img, (512, 512))
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    display_img = img.copy()
    image_name = image_path.split(os.path.sep)[-1].split(".")[0]

    cv.namedWindow('Draw Your Mask')
    cv.setMouseCallback('Draw Your Mask', draw_mask)
    print(
        """
        \n--- INSTRUCTIONS ---
        * Press 'c' to clear canvas.
        * Press 's' to save the final image and mask, then exit.
        * Press 'q' to quit without saving.
        """
    )

    while True:
        blended = cv.addWeighted(img, 0.7, display_img, 0.3, 0)
        cv.imshow('Draw Your Mask', blended)
    
        key = cv.waitKey(1) & 0xFF
        mask_name = os.path.join(output_dir, f"{image_name}.mask.png")
        crop_name = os.path.join(output_dir, f"{image_name}.png")

        match key:
            case ord('s'):
                cv.imwrite(crop_name, img)
                cv.imwrite(mask_name, mask)
                print("Images saved :)")
                break
            case ord('c'):
                display_img = img.copy()
                mask = np.zeros((512, 512, 3), dtype=np.uint8)
                cv.setMouseCallback('Draw Your Mask', draw_mask)
            case ord('q'):
                break
            case ord('r'):
                display_img = img.copy()
                mask = np.zeros((512, 512, 3), dtype=np.uint8)
                while True:
                    roi = cv.selectROI("Select rectangular mask", img,
                                       fromCenter=False,
                                       showCrosshair=True)
                    cv.destroyWindow("Select rectangular mask")
                    x, y, roi_w, roi_h = roi

                    if roi_w > 0 and roi_h > 0:
                        mask[y:y + roi_h, x:x + roi_w] = 255
                        print("Selected successfully!")
                        break
                cv.imwrite(crop_name, img)
                cv.imwrite(mask_name, mask)
                break
    cv.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise IOError("No image path provided. Please provide a path to an image.")

    parser = argparse.ArgumentParser(description="Interactive tool to crop and mask images for Stable Diffusion.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("-o", "--output", default=".", help="Folder path to save the resulting images.")

    args = parser.parse_args()
    run_masking_tool(args.image_path, args.output)