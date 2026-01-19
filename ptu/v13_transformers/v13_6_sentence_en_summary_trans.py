# 1. 라이브러리 불러오기
from transformers import pipeline
from deep_translator import GoogleTranslator

def trans_en_to_ko(sentence):
    """
    주어진 영어 문장을 한국어로 번역하는 함수
    """
    translated_sen = GoogleTranslator(source='en', target='ko').translate(sentence)
    return translated_sen

# 2. 요약 파이프라인 생성
summarizer = pipeline(
    "summarization",
    model="t5-small"
)

# 3. 요약할 영어 문장 입력
text = """
A special 25th anniversary edition of the extraordinary international bestseller, including a new Foreword by Paulo Coelho.
Combining magic, mysticism, wisdom and wonder into an inspiring tale of self-discovery, The Alchemist has become a modern classic, selling millions of copies around the world and transforming the lives of countless readers across generations.
Paulo Coelho's masterpiece tells the mystical story of Santiago, an Andalusian shepherd boy who yearns to travel in search of a worldly treasure. His quest will lead him to riches far different-and far more satisfying-than he ever imagined. Santiago's journey teaches us about the essential wisdom of listening to our hearts, of recognizing opportunity and learning to read the omens strewn along life's path, and, most importantly, to follow our dreams.
"""

# 4. 요약문 생성
summary = summarizer(text)

# 5. 요약문 가져오기
sum_text = summary[0]["summary_text"]

# 6. 요약문 출력
print(f"### 요약된 영어 문장 : {sum_text} ###")

# 7. 요약문 번역
# kr_sum_text = GoogleTranslator(source='en', target='ko').translate(sum_text)
kr_sum_text = trans_en_to_ko(sum_text)

# 8. 번역된 요약문 출력
print(f"### 번역된 한국어 문장 : {kr_sum_text} ###")