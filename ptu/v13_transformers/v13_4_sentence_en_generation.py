# 1. 라이브러리 불러오기
from transformers import pipeline

# 2. 텍스트 생성 파이프라인
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# 3. 사용자 입력 받기
answer = input("생성 문장을 입력해주세요 : ")

# 4. 텍스트 생성 실행
result = generator(
    answer, # 입력 문장
    max_new_tokens=50,   # 생성할 최대 토큰 수 (단어 단위와는 다름)
    num_return_sequences=1,  # 생성할 문장 수
    truncation=True # 입력이 모델 최대 토큰보다 길면 자르기
)

# 5. 결과 출력
print(result[0]["generated_text"])

