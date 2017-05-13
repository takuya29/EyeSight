# coding: utf-8

import cv2
from IPython import embed

# 画像ファイルの読み込み
img = cv2.imread('002a18_output.png')

# ORB (Oriented FAST and Rotated BRIEF)
detector = cv2.FeatureDetector_create('SIFT')

# 特徴検出
keypoints = detector.detect(img)

descripter = cv2.DescriptorExtractor_create("SIFT")
k1,d1 = descripter.compute(img, keypoints)
embed()
print len(k1),d1

# 画像への特徴点の書き込み
out = cv2.drawKeypoints(img, keypoints, None)

# 表示
cv2.imshow("keypoints", out)
cv2.imwrite("keypoints.png", out)