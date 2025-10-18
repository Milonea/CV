import numpy as np
import pytest
import cv2
from depth import generate_depth_map
from depth_3d import generate_point_cloud
def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)
    assert isinstance(depth_map, np.ndarray), "출력 타입이 ndarray가 아닙니다."
    assert depth_map.shape == image.shape, "출력 크기가 입력과 다릅니다."
def test_generate_point_cloud():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map, X, Y, Z = generate_point_cloud(image)
    assert isinstance(depth_map, np.ndarray), "출력 타입이 ndarray가 아닙니다."
    assert depth_map.shape == image.shape, "출력 크기가 입력과 다릅니다."
    assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray) and isinstance(Z, np.ndarray)
    assert X.shape == Y.shape == Z.shape == (100, 100)

if __name__ == "__main__":
    pytest.main()
