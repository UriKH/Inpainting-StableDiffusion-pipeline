import cv2 as cv
import numpy as np
import sys

BRUSH_SIZE = 20
drawing = False
mask = None
display_img = None


def draw_mask(event, x, y, flags, param):
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
    h, w = img.shape[:2]
    if h != w:
        print("\n--- CROP INSTRUCTIONS ---")
        print("1. Your image is not a square. Click and drag to select a region.")
        print("2. Press ENTER or SPACE to confirm your crop.")
        print("3. Press 'c' to cancel and let it auto-center-crop.\n")

        # Select ROI
        roi = cv.selectROI("Crop Image (Press Enter when done)", img, fromCenter=False, showCrosshair=True)
        cv.destroyWindow("Crop Image (Press Enter when done)")
        x, y, roi_w, roi_h = roi

        if roi_w > 0 and roi_h > 0:
            side = min(roi_w, roi_h)
            img = img[y:y + side, x:x + side]
            print("Cropped successfully!")
        else:
            print("No box drawn. Auto-center-cropping to a square...")
            side = min(h, w)
            start_x = w // 2 - side // 2
            start_y = h // 2 - side // 2
            img = img[start_y:start_y + side, start_x:start_x + side]
    return img


def run_masking_tool(image_path):
    global mask, display_img

    img = cv.imread(image_path)
    if img is None:
        raise Exception(f"Error: Could not load {image_path}. Check the filename!")

    img = crop_image(img)
    img = cv.resize(img, (512, 512))
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    display_img = img.copy()
    
    # Create mask
    cv.namedWindow('Draw Your Mask')
    cv.setMouseCallback('Draw Your Mask', draw_mask)
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Click and drag on the image to paint your mask.")
    print("2. Press 'c' to clear canvas.")
    print("3. Press 's' to save the final image and mask, then exit.")
    print("4. Press 'q' to quit without saving.\n")
    
    while True:
        blended = cv.addWeighted(img, 0.7, display_img, 0.3, 0)
        cv.imshow('Draw Your Mask', blended)
    
        key = cv.waitKey(1) & 0xFF
    
        if key == ord('s'):
            cv.imwrite("init_image.png", img)
            cv.imwrite("mask_image.mask.png", mask)
            print("Success! Saved 'init_image.png' and 'mask_image.png'.")
            break
        elif key == ord('c'):
            display_img = img.copy()
            mask = np.zeros((512, 512, 3), dtype=np.uint8)
            print("Cleared mask.")
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise IOError("No image path provided. Please provide a path to an image.")
    run_masking_tool(sys.argv[1])
