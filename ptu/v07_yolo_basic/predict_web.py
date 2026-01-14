import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# 페이지 설정
st.set_page_config(
    page_title="YOLO 표정 분류기",
    page_icon="😊",
    layout="wide"
)

# 제목
st.title("😊 YOLO 표정 분류 모델 추론")
st.markdown("---")

# 사이드바 - 모델 업로드
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 파일 업로드
    model_file = st.file_uploader(
        "YOLO 모델 파일 업로드 (.pt)",
        type=['pt'],
        help="학습된 YOLO 분류 모델 파일을 업로드하세요"
    )
    
    st.markdown("---")
    st.markdown("### 📋 모델 정보")
    st.info("""
    - **클래스**: Happy, Sad, Normal
    - **이미지 크기**: 256x256
    - **모델 타입**: YOLO Classification
    """)

# 모델이 업로드되었는지 확인
if model_file is not None:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(model_file.read())
        model_path = tmp_file.name
    
    try:
        # YOLO 모델 로드
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        st.sidebar.success("✅ 모델이 성공적으로 로드되었습니다!")
        
        # 탭 생성
        tab1, tab2 = st.tabs(["📷 이미지 업로드", "🎥 웹캠 추론"])
        
        # ========== 탭 1: 이미지 업로드 ==========
        with tab1:
            st.header("이미지 업로드 추론")
            
            uploaded_image = st.file_uploader(
                "이미지를 업로드하세요",
                type=['jpg', 'jpeg', 'png'],
                key="image_uploader"
            )
            
            if uploaded_image is not None:
                # 이미지 열기
                image = Image.open(uploaded_image)
                
                # 2열 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("원본 이미지")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("예측 결과")
                    
                    # 추론 버튼
                    if st.button("🔍 분류 시작", key="classify_btn"):
                        with st.spinner("분석 중..."):
                            # 예측 수행 (imgsz=256)
                            results = model.predict(
                                source=image,
                                imgsz=256,
                                verbose=False
                            )
                            
                            # 결과 추출
                            result = results[0]
                            top_class_idx = result.probs.top1
                            top_confidence = result.probs.top1conf.item()
                            class_name = result.names[top_class_idx]
                            
                            # 결과 표시
                            st.success("분석 완료!")
                            
                            # 큰 글씨로 결과 표시
                            st.markdown(f"""
                            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                                <h2 style='color: #1f77b4; margin: 0;'>{class_name}</h2>
                                <h1 style='color: #2ca02c; margin: 10px 0;'>{top_confidence*100:.2f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 이모지 표시
                            emoji_map = {
                                'Happy': '😊',
                                'Sad': '😢',
                                'Normal': '😐'
                            }
                            if class_name in emoji_map:
                                st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{emoji_map[class_name]}</h1>", unsafe_allow_html=True)
        
        # ========== 탭 2: 웹캠 추론 ==========
        with tab2:
            st.header("웹캠 실시간 추론")
            
            st.info("📸 아래 카메라 버튼을 클릭하여 사진을 촬영하면 자동으로 분류됩니다.")
            
            # Streamlit 내장 카메라 입력 사용
            camera_image = st.camera_input("카메라로 사진 촬영")
            
            if camera_image is not None:
                # 이미지 열기
                image = Image.open(camera_image)
                
                # 2열 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("촬영된 이미지")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("예측 결과")
                    
                    with st.spinner("분석 중..."):
                        # 예측 수행 (imgsz=256)
                        results = model.predict(
                            source=image,
                            imgsz=256,
                            verbose=False
                        )
                        
                        # 결과 추출
                        result = results[0]
                        top_class_idx = result.probs.top1
                        top_confidence = result.probs.top1conf.item()
                        class_name = result.names[top_class_idx]
                        
                        # 결과 표시
                        st.success("분석 완료!")
                        
                        # 이모지 맵
                        emoji_map = {
                            'Happy': '😊',
                            'Sad': '😢',
                            'Normal': '😐'
                        }
                        emoji = emoji_map.get(class_name, '😐')
                        
                        # 큰 글씨로 결과 표시
                        st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                            <h1 style='font-size: 80px; margin: 0;'>{emoji}</h1>
                            <h2 style='color: #1f77b4; margin: 10px 0;'>{class_name}</h2>
                            <h1 style='color: #2ca02c; margin: 10px 0;'>{top_confidence*100:.2f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.sidebar.error(f"❌ 모델 로드 중 오류 발생: {str(e)}")
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(model_path):
            os.unlink(model_path)

else:
    # 모델이 업로드되지 않은 경우
    st.info("👈 왼쪽 사이드바에서 YOLO 모델 파일(.pt)을 업로드해주세요.")
    
    st.markdown("""
    ### 사용 방법
    1. **모델 업로드**: 사이드바에서 학습된 YOLO 분류 모델 파일을 업로드합니다.
    2. **이미지 추론**: '이미지 업로드' 탭에서 이미지를 업로드하고 분류합니다.
    3. **웹캠 추론**: '웹캠 추론' 탭에서 실시간으로 표정을 분류합니다.
    
    ### 지원 기능
    - ✅ 256x256 이미지 크기로 추론
    - ✅ Happy, Sad, Normal 표정 분류
    - ✅ 최고 신뢰도 점수 표시
    - ✅ 실시간 웹캠 추론
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>YOLO 표정 분류기 | Powered by Ultralytics YOLO11</p>
</div>
""", unsafe_allow_html=True)