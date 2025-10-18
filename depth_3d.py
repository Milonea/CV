import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_point_cloud(image):
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    return depth_map, X, Y, Z

if __name__ == "__main__":
    image = cv2.imread('sample.jpg')
    if image is None:
        raise ValueError("입력 이미지가 없습니다.")
        exit()
    depth_map, X, Y, Z = generate_point_cloud(image)
    step = 4
    X = X[::step, ::step]
    Y = Y[::step, ::step]
    Z = Z[::step, ::step]
    colors = depth_map[::step, ::step, ::-1] / 255.0
    cv2.imshow('Original Image', image)
    cv2.imshow('Depth Map', depth_map)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X.flatten(),
        Y.flatten(),
        Z.flatten(),
        c=colors.reshape(-1, 3),
        s=1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth (Z)")
    ax.set_title("3D Point Cloud from Depth Map")
    plt.show()