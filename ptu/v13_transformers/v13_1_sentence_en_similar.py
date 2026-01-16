# 1. 라이브러리 설치
    # pip install sentence-transformers
    
# 2. 라이브러리 불러오기
from sentence_transformers import SentenceTransformer, util

# 3. 사전 학습된 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")
"""
all-MiniLM-L6-v2 모델 설명 : 
- 가벼운(경량) 문장 임베딩 모델
- 영어 문장을 벡터 공간에 매핑
- 특징
    1. 빠른 연산 속도
    2. 문장 의미를 벡터로 잘 반영
    3. 검색, 추천, 유사도 계산에 유용
"""

# 4. 비교할 두 문장 정의(다른 무장)
# sen1 = "The cat is sleeping on the sofa"
# sen2 = "Tomorrow, I have a math exam at school"

# 4-1. 비교할 두 문장 정의(비슷한 문장)
sen1 = "He is reading a book in the library"
sen2 = "He is at the library reading a book"

# 의미상 완전히 다른 문장 예시
# => 낮은 유사도 기대
# print(sen1)
# 5. 문장을 벡터로 변환
# 모델이 문장을 이해할 수 있도록 벡터로 변환
emb1 = model.encode(sen1, convert_to_tensor=True)
emb2 = model.encode(sen2, convert_to_tensor=True)
# print(emb1)

# 6. 코사인 유사도 계산
cos_sim = util.pytorch_cos_sim(emb1, emb2)
# -1 : 완전히 반대
# 0 : 무관
# 1 : 완전히 동일

# 7. 결과 출력
print(f"두 문장의 유사도 : {cos_sim.item():.4f}")
