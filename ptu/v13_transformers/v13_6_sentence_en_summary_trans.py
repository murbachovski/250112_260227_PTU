# 1. 라이브러리 불러오기
from transformers import pipeline
from deep_translator import GoogleTranslator
# pip install deep-translator
# *영어 => 한국어로 번역하는 함수 정의

def trans_en_kr(english_text):
    trans_sen = GoogleTranslator(source="auto", target="ko").translate(english_text)
    return trans_sen

# 2. 요약 파이프라인 생성
summary = pipeline(
    "summarization",
    model="t5-small"
)

# 3. 요약할 영어 문장 입력
text = """
The Transformers library by Hugging Face provides state-of-the-art general-purpose architectures for natural language understanding and generation. It offers a wide range of pre-trained models that can be fine-tuned for various NLP tasks such as text classification, named entity recognition, question answering, and text summarization. The library is built on top of PyTorch and TensorFlow, making it easy to integrate into existing machine learning workflows. With its user-friendly API and extensive documentation, the Transformers library has become a popular choice among researchers and developers in the field of natural language processing.
"""

# 4. 요약문 생성
summary_text = summary(text)

# 5. 요약문 가져오기
english_summary = summary_text[0]['summary_text']

# 6. 영어 요약문을 한국어로 번역
trans_sum_ko = trans_en_kr(english_summary)
print(trans_sum_ko)