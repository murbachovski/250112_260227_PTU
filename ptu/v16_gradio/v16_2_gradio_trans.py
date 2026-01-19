import gradio as gr
from deep_translator import GoogleTranslator

# 번역 함수 정의
def trans_en_to_ko(text):
    translated = GoogleTranslator(
        source="en",
        target="ko"
    ).translate(text)
    
    return translated

# Gradio 웹 인터페이스 생성
gr.Interface(
    fn=trans_en_to_ko,
    inputs="text",
    outputs="text",
    title="Simple Translation Website",
    description="영어 문장을 입력하면 한국어로 번역됩니다."
).launch()