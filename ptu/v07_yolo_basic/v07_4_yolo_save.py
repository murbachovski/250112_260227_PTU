from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("yolo11n.pt")

# 2. 모델 추론
model("v07_yolo_basic/input_det.mp4", save=True)

# Ultralytics 공식 문서나 구글링, GPT, Gemini 검색하여 찾아보기