import cv2
import numpy as np
# 이미지 로드
image = cv2.imread('sample.jpg') # 분석할 이미지 파일
# BGR에서 HSV 색상 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 색상 범위 지정
lower = np.array([10, 10, 70])
upper = np.array([30, 255, 255])
# 마스크 생성
mask = cv2.inRange(hsv, lower, upper)
# 원본 이미지에서 색상 범위 내 부분만 추출
result = cv2.bitwise_and(image, image, mask=mask)
# 결과 이미지 출력
cv2.imshow('Original', image)
cv2.imshow('Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()