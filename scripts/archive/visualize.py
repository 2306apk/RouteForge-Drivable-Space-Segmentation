import cv2

img = cv2.imread("sample.jpg")
mask = cv2.imread("mask.png", 0)

overlay = img.copy()
overlay[mask == 255] = (0, 255, 0)

cv2.imshow("overlay", overlay)
cv2.waitKey(0)