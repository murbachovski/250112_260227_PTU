from ultralytics import YOLO
import cv2

# 1. 웹캠 연결
cap = cv2.VideoCapture(0)

# 2. 모델 로드
model = YOLO("yolo11n-seg.pt")

# 3. 프레임 처리
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("웹캠 읽기 실패")
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO_SEG", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q키를 눌러서 종료")
        break
    
# 4. 자원해제
cap.release()
cv2.destroyAllWindows()