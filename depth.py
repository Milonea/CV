import cv2
import numpy as np

def generate_depth_map(image):
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return depth_map
if __name__ == "__main__":
    image = cv2.imread('sample.jpg')
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")
        exit()
    depth_map = generate_depth_map(image)
    cv2.imshow('Original Image', image)
    cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()