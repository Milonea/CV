from datasets import load_dataset
import cv2
import numpy as np
from PIL import Image
dataset = load_dataset("ethz/food101", split="train")
def is_bright_enough(image: Image.Image, threshold=50):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness >= threshold
def get_contour_mask(image: Image.Image):
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_bgr)
    cv2.drawContours(mask, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
    return mask, len(contours) > 0
def has_large_enough_object(image: Image.Image, min_ratio=0.3):
    mask, exists = get_contour_mask(image)
    if not exists:
        return False
    object_area = cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    total_area = mask.shape[0] * mask.shape[1]
    ratio = object_area / total_area
    return ratio >= min_ratio
sample_count = 5
selected_images = []
for img_data in dataset:
    img = img_data['image']
    if is_bright_enough(img, threshold=50) and has_large_enough_object(img):
        selected_images.append(img)
    if len(selected_images) >= sample_count:
        break
for i, img in enumerate(selected_images):
    orig_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    contour_mask, _ = get_contour_mask(img)
    overlay_orig = cv2.addWeighted(orig_bgr, 1.0, contour_mask, 0.5, 0)
    processed_bgr = orig_bgr.copy()
    if i == 0:
        gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
        processed_bgr = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        processed_bgr = cv2.cvtColor(processed_bgr, cv2.COLOR_GRAY2BGR)
    elif i == 1:
        processed_bgr = cv2.GaussianBlur(processed_bgr, (5, 5), 0)
    elif i == 2:
        processed_bgr = cv2.flip(processed_bgr, 1)
    elif i == 3:
        processed_bgr = cv2.rotate(processed_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif i == 4:
        processed_bgr = cv2.bitwise_not(processed_bgr)
    processed_bgr = cv2.resize(processed_bgr, (224, 224))
    cv2.imshow(f'Original with Contour {i+1}', overlay_orig)
    cv2.imshow(f'Processed {i+1}', processed_bgr)
    key = cv2.waitKey(0)