# 1. 라이브러리 설치
# pip install transformers

# 2. 라이브러리 불러오기
from transformers import pipeline
# pipeline : 텍스트, 이미지 등 다양한 AI 테스크를 쉽게 실행할 수 있는 유틸

# 3. 감정 분석 파이프라인 생성
classifier = pipeline("sentiment-analysis")
"""
설명:
- sentiment_analysis : 감정 분석을 의미
- 영어 문장을 입력하면 긍정인지 부정인지 판단
"""

# 4. 감정 분석할 문장 입력
# text = "I'm having a hard time today"
text = "I'm feeling really great today"
results = classifier(text)

# 5. 결과 확인
print(f"감정 분석 결과 : {results[0]['label']}")
print(f"감정 분석 점수 : {results[0]['score']:.4f}") # 확률 값 0 ~ 1
