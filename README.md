# 250000_0000_우리인재개발원
### 강의 주제
```
통합(인파 밀집 위험 예측 경보 시스템 + 교차로 교통 장애물 및 이벤트 감지 시스템)
```

### 강의 내용
```
1. 개발 요구사항 분석 및 기술 분석
교차로 교통 장애물 및 이벤트 감지 시스템의 필요성과 기술적 요구사항을 분석

2. 목표 설정 및 기획서 작성
프로젝트의 목표 설정 및 기획서 작성

3. 데이터 수집 및 전처리 모듈 개발
교차로 데이터 수집 및 전처리 작업 수행

4. 객체 감지 및 분류 모델 모듈 개발
객체 감지를 위한 모델 개발 및 분류 기능 구현

5. 실시간 처리 및 시각화 모듈 개발
실시간 데이터 처리를 위한 모듈 개발 및 결과 시각화

6. 이벤트 감지 및 분류 모듈 개발
이벤트 감지를 위한 모듈 개발 및 분류 기능 구현

7. 모델 성능 평가 및 성능 개선
개발된 모델의 성능 평가 및 개선 작업 수행

8. 교차로 교통 장애물 감지 모듈 개발
교차로 교통 장애물 감지를 위한 모듈 개발

9. Dashboard 연동, 테스트 및 디버깅
Dashboard와의 연동, 시스템 테스트 및 디버깅

10. 프로젝트 결과
프로젝트 PPT 발표
```

### 강의 시간
```
1) 09:30 ~ 10:20(50분)
2) 10:30 ~ 11:20(50분)
3) 11:30 ~ 12:20(50분)
4) 12:30 ~ 13:20(50분)
점심 13:20 ~ 14:10(점심)
5) 14:10 ~ 15:00(50분)
6) 15:10 ~ 16:00(50분)
7) 16:10 ~ 17:00(50분)
8) 17:10 ~ 18:00(50분)
(8교시, 총 400분)
```

### 강의 목차
```
v0_Install anaconda/
→ Anaconda 설치 및 환경 설정

v1_Install vscode/
→ VS Code 설치 및 개발 환경 구성

v2_Basic python/
→ Python 기초 문법 및 실습

v3_Yolo 기초/
→ YOLO 객체 탐지 모델 추론 및 활용

v4_TWilio/
→ Twilio API를 활용한 문자(SMS) 알림 기능 구현

v5_OpenCV2/
→ OpenCV를 이용한 이미지 처리 실습

v6_Data
│
├── v6_1_Data/
│ → 공공데이터 포털 활용 및 교통/환경 데이터 수집
│
├── v6_2_Get Local Data/
│ → Local 이미지 수집 및 자동 이미지 저장 기능 구현
│
└── v6_3_OpenAPI/
     → 공공기관(OpenAPI) 연계 실시간 정보 수집

v7_YOLO 심화/
→ Solution 탐색
├── classify
├── train classify
├── detect
├── params
├── alarm
├── distance
├── sahi
├── heatmap
├── region
├── get region
├── speed
├── blurr
├── crop
├── in and out
├── line
├── YOLOE
├── multi thread
├── model.fuse()
├── Streamlit YOLO
└── OpenVINO int8

v8_Web
└── v8_3_Streamlit/
     → Streamlit을 활용한 YOLO 객체 탐지 실시간 시각화 대시보드 구현

v8_4_Plus/
    → HuggingFace
    → ngrok
    → pip free > requirements.txt
    → pip install pipreqs
    → model.fuse()
    → YOLOE
    → OpenVINO
    → YOLO_Streamlit
    → Export TensorRT
```

---
## 강의 관련 링크 모음

### 2. model.fuse
[model.fuse](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.fuse)

---

### 3. YOLOE
[YOLOE](https://docs.ultralytics.com/ko/models/yoloe/)

---

### 4. OpenVINO
[OpenVINO](https://docs.ultralytics.com/ko/guides/optimizing-openvino-latency-vs-throughput-modes/)

---

### 6. Streamlit
[Streamlit](https://docs.ultralytics.com/ko/guides/streamlit-live-inference/)

---

### 7. Training
[Training](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)

---

### 8. Line, MultiThread 관련 Ultralytics 공식 문서 자료
[Ultralytics](https://docs.ultralytics.com/ko/modes/track/#faq)

---

### 9. TensorRT 관련 Ultralytics 공식 문서 자료
[Ultralytics TensorRT](https://docs.ultralytics.com/ko/integrations/tensorrt/)

---

### 10. Miro
[Miro](https://miro.com/app/dashboard/)

---

### 11. MLOPS
[Google MLOps](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko#devops_versus_mlops)

---

### 12. LabelImg
[LabelImg](https://github.com/HumanSignal/labelImg)<br>

---

### 13. RoboFlow
[Roboflow](https://roboflow.com/)<br>

---

### 14. Pixels
[pixels](https://roboflow.com/)<br>

---

### 15. ITS
[ITS](https://its.go.kr/)<br>

### 16. 성능평가 관련
🚩 [Precision-Recall vs. ROC Curve - CosmicCoding](https://cosmiccoding.com.au/tutorials/pr_vs_roc_curves/) <br>
🚩 [Receiver Operating Characteristic (ROC) - Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) <br> 
🚩 [Confusion Matrix - Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix) <br>
🚩 [ROC Curve & AUC 설명 - Dream2Reality 블로그](https://dream2reality.tistory.com/9) <br>
🚩 [머신러닝 성능 측정 방법 - Meme2515 블로그](https://meme2515.github.io/machine_learning/performance_measurement/) <br>
🚩 [분류 성능 지표 (Precision, Recall) - AI-Com 블로그](https://ai-com.tistory.com/entry/ML-%EB%B6%84%EB%A5%98-%EC%84%B1%EB%8A%A5-%EC%A7%80%ED%91%9C-Precision%EC%A0%95%EB%B0%80%EB%8F%84-Recall%EC%9E%AC%ED%98%84%EC%9C%A8) <br>

---

### 17. 성능 평가 문제
🚩 [Google Machine Learning Crash Course - Precision & Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) <br>
🚩 [Google Machine Learning Crash Course - Classification: ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) <br>

---

## 강의 관련 내용 모음
### 00. OpenAPI 설명
<p align="center">
  <img src="https://github.com/user-attachments/assets/9e80f6a1-f7c2-47ee-b162-a59e9cc888fb" width="1000">
</p>

### 1. APIKEY
```
db5c00dc1fce45c49049bff225a0fea6
```

---

### 2. API 요청 URL 생성
```
url_cctv = f"https://openapi.its.go.kr:9443/cctvInfo?apiKey={key}&type={Type}&cctvType=1&minX={minX}&maxX={maxX}&minY={minY}&maxY={maxY}&getType={getType}"
```

---

### 3. 면적 측정

1. [구글맵](https://www.google.co.kr/maps/?entry=ttu&g_ep=EgoyMDI1MDIwMi4wIKXMDSoASAFQAw%3D%3D)
<p align="center">
  <img src="https://github.com/user-attachments/assets/46a67170-4d6c-4a3b-9f64-1d9f5f8c2a98" width="300">
</p>

2. [네이버지도](https://map.naver.com/p?c=15.00,0,0,0,dh)
<p align="center">
  <img src="https://github.com/user-attachments/assets/02d5db1d-1d19-4c20-b180-25c9655469a7" width="300">
</p>

3. [카카오맵](https://map.kakao.com/?nil_profile=title&nil_src=local)
<p align="center">
  <img src="https://github.com/user-attachments/assets/5aa664df-7d71-4a70-88a1-1ea83d45786f" width="300">
</p>

---

### 4. YOLO custom_datasets 경로 셋팅
```
coco8.yaml => path : coco8 폴더 경로, train : train 폴더 경로, val : val 폴더 경로
model.train(data='coco8.yaml 파일 경로')
```

---

### 5. Background images
<img src="https://github.com/user-attachments/assets/052d795a-8361-4905-b325-8124e7ba729d" width="600">
```
FP => 거짓 탐지 => 오탐을 줄일 수 있다.
```

---

### 6. Data Augmentation
<p align="center">
  <img src="https://github.com/user-attachments/assets/81c866a3-c39d-4cb4-89d6-a7bc818e7a65" width="600">
</p>

---

### 7. 파이썬 경고음 넣기
[더미 경고음 사이트](https://pixabay.com/ko/sound-effects/search/%EA%B2%BD%EA%B3%A0%EC%9D%8C/)

---

```
# MAC
import os
os.system(afplay ./alarm.mp3)

# MAC(비동기)
import subprocess
subprocess.Popen(["afplay", "./alarm.mp3"])

# Winodws
pip install playsound
from playsound import playsound
playsound('./alarm.mp3')
```

---

### 8. requirements.txt 생성 라이브러리 piqres
```
1. 기존 pip freeze 와 비교
2. pip freeze > requirements.txt
3. pip install pipreqs
3-1. pipreqs .
```

---

### 9. ngrok 외부 호스팅
```
1. ngrok 설치 https://ngrok.com/downloads/windows?tab=download
2. 실행 명령어 ngrok http 8051(자신의 포트번호)
3. 회원 가입 후 키 발급 확인
3-1. https://dashboard.ngrok.com/authtokens
4. 키 인증
4-1. ngrok config add-authtoken 32DAhV31Wq2vLJIr5WKWQ9vyN8v_2s9tVHTeD1WdCK23oVjFa
5. ngrok http 8080(자신의 포트번호) 외부 호스팅 2시간 무료
```

---

### 10. requirements.txt 만들기
```
1. pip install pipreqs 설치
2. 프로젝트 폴더 경로 이동
3. pipreqs --savepath ./requirements.txt
4. 저장 경로 확인
```

---

### 11. Twilio 활용하여 Python으로 문자 알림 보내기
[Twilio](https://www.twilio.com/en-us)
```
Twilio 회원가입 후
번호 등록 및 생성 후 코드 변환하여 사용
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd68c8dd-626e-475c-97fc-a50108abdd10" width="1000">
</p>

---

### 12. README.md 파일 작성법 및 소개
```
https://gist.github.com/ihoneymon/652be052a0727ad59601
```

---

### 13. Precision(정밀도)
```
모델의 Positive로 판정한 것 중, 실제 Positive 비율
```

---

### 14. Recall(재현율)
```
실제 Positive 중 모델의 Positive 비율
```

---

### 15. F1-score
<p align="center">
  <img src="https://github.com/user-attachments/assets/4fdffc5c-ae29-4dab-8ec0-5f80e025d268" width="300">
</p>

---

### 16. 조화평균(여러 값의 평균을 구할 때, 작은 값이 상대적으로 더 큰 영향을 주는 평균 방식)
```
1. Precision과 Recall 중 하나라도 낮으면 F1-score도 낮아짐
  예를 들어 Precision = 90, Recall = 10이면 일반 평균은 50이지만, 조화평균을 쓰면 F1-score ≈ 18.2로 낮아짐 → 한쪽이 낮으면 전체 성능도 낮게 반영

2. 둘의 균형을 맞추는 데 효과적
  Precision이 높지만 Recall이 낮거나, 그 반대인 경우를 방지

3. 극단적인 값을 줄여줌
  예를 들어, 일반 평균(산술평균)은 극단적인 값(예: 한쪽이 100, 한쪽이 1)에 영향을 많이 받지만, 조화평균은 이를 보완
```

---

### 17. ROC Curve
<p align="center">
  <img src="https://github.com/user-attachments/assets/91d7948a-ec7e-483a-b8eb-e0e1a53e0f60" width="300">
</p>
```
✅ ROC Curve는 모델의 전체적인 분류 성능을 평가하는 곡선
✅ FPR vs. TPR의 관계를 나타내며, 좌상단에 가까울수록 좋은 모델
✅ AUC 값이 클수록 좋은 성능을 의미 (1에 가까울수록 우수)
```

---

### 18. Precision과 Recall의 경우의 수
| Precision (정밀도) | Recall (재현율) | 의미                                       |
|-------------------|---------------|------------------------------------------|
| 높음               | 높음            | 이상적인 모델 (오탐과 미탐이 적음)              |
| 높음               | 낮음            | 탐지를 신중하게 하지만 많은 객체를 놓침 (미탐 증가) |
| 낮음               | 높음            | 많은 객체를 탐지하지만 오탐이 많음 (오탐 증가)     |
| 낮음               | 낮음            | 모델 성능이 매우 나쁨 (오탐과 미탐이 많음)        |

### 19. Precision-Recall 관련 문제
#### 객체 탐지 모델을 적용했더니 탐지된 객체는 대부분 정확하지만, 많은 실제 객체를 놓치는 경우
```
✅ 정답: Precision ↑, Recall ↓
✅ 해결 방법: Recall을 높이기 위해 Confidence Threshold를 낮추고 탐지 범위를 확대해야 한다.
```

---

#### 탐지 시스템에서 거의 모든 사람을 탐지할 수 있지만, 실제 사람이 아닌 그림자나 마네킹도 사람으로 오탐하는 경우
```
✅ 정답: Precision ↓, Recall ↑
✅ 해결 방법: Confidence Threshold를 높이고, Hard Negative Mining을 적용
```

---

#### 탐지된 객체 중 상당수가 오탐이며, 실제 객체도 잘 탐지되지 않는 경우가 발생
```
✅ 정답: Precision ↓, Recall ↓
✅ 해결 방법: 데이터셋을 개선하고, 모델을 추가 학습해야 한다. 또한, NMS와 Confidence Threshold를 적절히 조정하여 탐지 성능을 개선
```

---

#### 모델이 적용된 후 탐지된 객체는 대부분 정확하고, 실제 객체도 놓치지 않고 탐지
```
✅ 정답: Precision ↑, Recall ↑
✅ 설명: 이상적인 모델이며, 성능이 최적화된 상태
```

---

### 20. Hard Negative Mining
```
객체 탐지 모델에서 오탐(False Positive)이 많은 경우, 특히 배경을 객체로 잘못 인식하는 문제를 해결하는 기법
즉, 모델이 헷갈려하는 "어려운 배경(하드 네거티브)"을 학습 데이터로 추가하여 성능을 개선하는 방법
```

---

### 21. Base64 인코딩
<p align="center">
  <img src="https://github.com/user-attachments/assets/84a0bfe9-ce05-46cc-84f1-ac3343aa3c73" width="600">
</p>

```
문자열을 다시 디코딩하면 이미지로 돌아옵니다.
```

---

### 22. Anaconda 환경 셋팅
가상환경 생성
```
conda create -n "가상환경 이름" python=3.9
$  conda create -n py39 python=3.9
```

가상환경 실행
```
conda activate "가상환경 이름"
$  conda activate py39
```

가상환경 종료
```
conda deactivate 
```

라이브러리 설치
```
pip install "설치할 라이브러리"
pip install ultralytics
```

---

### 23. README.md 파일 작성법 및 소개
```
https://gist.github.com/ihoneymon/652be052a0727ad59601
```

---

### 24. HuggingFace<br>
[허깅페이스](https://huggingface.co/)<br>
```
1. 회원가입
2. 이메일 인증
3. 인증키 발급
4. 원하는 모델 검색
5. 인증키 적용
6. 코드 실행
```

---

**Original Image**<br>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/9a975c01-97eb-46e6-a755-3042c6919213" width="500" height="500" style="object-fit: cover;">
</div>

---

 **Background Image**(based on ChatGPT)<br>
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/1f34235a-210f-45a8-a544-366266fa65a4" width="500" height="500" style="object-fit: cover;">
</div>

---

**tips_for_best_training_results**<br>
[tips_for_best_training_results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results)<br>

---

**Resuming Interrupted Trainings**<br>
[YOLO Train Parameter resume](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings)<br>

---

**Transfer Learning with Frozen Layers**<br>
[YOLO Train Parameter freeze](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/)<br>

---

**Pruning**<br>
[Model Pruning and Sparsity in YOLOv5](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity/)<br>
<p align="left">
  <img src="https://github.com/user-attachments/assets/6257c65e-c700-4e30-83b8-32e9f5e33abd" width="700">
</p>

---

**Quantization**<br>
[Model Quantization](https://docs.ultralytics.com/guides/model-deployment-practices/#model-quantization)<br>
<p align="left">
  <img src="https://github.com/user-attachments/assets/7703ab83-7ccc-48b2-b7cc-79eea977c767" width="700">
</p>

---

## 💡Tensorboard 확인
```
# log 파일 생성 확인 방법
Yolo Train => runs/detect/train/events.out.tfevents.1744182544.2AT.1604

# Tensorboard 설치
pip install tensorboard

# Tensorboard 실행(CMD)
tensorboard --logdir="C:/Users/Administrator/Desktop/ai/runs/detect/train"
```

---

## 🔍 `visualize=True` 시각화 스테이지 분석
<p align="left">
  <img src="https://github.com/user-attachments/assets/927a7a29-d8de-4cca-85d4-d5d348c80ffd" width="700">
</p>

```
visualize=True 옵션은 YOLO 모델의 추론 시 중간 레이어의 Feature Map(특징 맵) 을 이미지로 저장해주는 기능
모델 내부가 어떻게 입력 영상을 해석하고 있는지 시각적으로 확인
```

---

```
밝은 영역: 해당 위치에서 강한 activation (특징 반응) 이 있었음을 의미
어두운 영역: 모델이 관심을 덜 가지는 부분
중간 레이어는 저수준 특징(모서리, 색상 등), 깊은 레이어는 고수준 특징(객체 윤곽 등)을 포착함
주의: 밝은 activation이 있다고 해서 반드시 탐지된 객체가 있는 건 아님 (confidence, 후처리 기준 미달일 수 있음)
```

| Stage 범위         | 위치                      | 의미                           | 시각화 특징                                 |
|-------------------|---------------------------|--------------------------------|---------------------------------------------|
| `stage_0 ~ stage_4`  | 초기 Convolution Layer     | Edge, Corner 탐지               | 밝기 변화, 윤곽선 강조                        |
| `stage_5 ~ stage_10` | 중간 Layer (Backbone)      | 모양, 패턴 인식                 | 윤곽보다 내부 구조 표현                       |
| `stage_11 ~ stage_16`| Neck (FPN, PAN 등)         | Multi-scale Feature 강화        | 복잡하고 의미 있는 부분 강조                 |
| `stage_17 ~ stage_20`| Head (예측 전 단계)        | 객체 존재 위치/크기 판단         | 관심 객체의 중심 부분만 밝게 나옴             |

---

### 초기 셋팅
```
1. Anaconda 설치
https://repo.anaconda.com/archive/
버전 : Anaconda3-2024.10-1-Windows-x86_64.exe ==> 설치

2. 가상환경 생성
conda create -n "Youre_env_name" python=3.9

3. 가상환경 실행
conda activate py39
```

---

### 라이브러리 설치
```
1. pip install pipreqs 설치
2. 프로젝트 폴더 경로 이동
3. pipreqs --savepath ./requirements.txt
4. 저장 경로 확인
5. pip install -r ./requirements.txt
```

---
