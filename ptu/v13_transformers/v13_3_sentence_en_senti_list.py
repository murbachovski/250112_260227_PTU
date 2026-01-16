# 1. 라이브러리 불러오기
from transformers import pipeline

# 2. 감정 분석 파이프라인 생성
sentiment_analysis = pipeline(
    "sentiment-analysis", # 감정 분석 테스크 지정
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# 3. 감정 분석할 문장 목록(리스트)
sentences = [
    "I don't like a bird",
    "I like a dog",
    "I really hate summer",
    "I like a winter"
]

# 4. 각 문장에 대해 감정 분석 수행
for s in sentences:
    result = sentiment_analysis(s)[0]
    
    # 5. 결과 출력
    print(f"분석할 문장 : {s}")
    print(f"감정 : {result['label']}")
    print(f"점수 : {result['score']:.5f}")
    print("### 출력 완료!!! ###")
