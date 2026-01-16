from transformers import pipeline

# 1. 요약 파이프라인 생성
summarizer = pipeline(
    "summarization", # 요약 테스크 지정
    model="t5-small"
)

# 2. 요약할 긴 문장 입력
text = """
A special 25th anniversary edition of the extraordinary international bestseller, including a new Foreword by Paulo Coelho.
Combining magic, mysticism, wisdom and wonder into an inspiring tale of self-discovery, The Alchemist has become a modern classic, selling millions of copies around the world and transforming the lives of countless readers across generations.
Paulo Coelho's masterpiece tells the mystical story of Santiago, an Andalusian shepherd boy who yearns to travel in search of a worldly treasure. His quest will lead him to riches far different-and far more satisfying-than he ever imagined. Santiago's journey teaches us about the essential wisdom of listening to our hearts, of recognizing opportunity and learning to read the omens strewn along life's path, and, most importantly, to follow our dreams.
"""

# 3. 요약문 생성
summary = summarizer(text)

# 4. 요약문 가져오기
sum_text = summary[0]['summary_text']

# 5. 요약문 출력
print("########################")
print(f"요약된 문장 : {sum_text}")