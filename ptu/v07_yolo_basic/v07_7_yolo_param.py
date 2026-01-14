from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("yolo11x.pt")

# 모델 클래스 확인
# print(f"모델 클래스 목록 : {model.names}")

model(
    "v07_yolo_basic/class.jpg",
    save=True,
    # max_det=1
    # save_crop=True,
    # save_txt=True,
    # save_conf=True
)
# 결과 이미지 => 
# cell phone, book만 탐지되도록
# Ultralytics 공식 문서, 구글링, GPT
