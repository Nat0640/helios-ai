import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 (โมเดลจะถูกดาวน์โหลดอัตโนมัติในครั้งแรก)
# เราใช้โมเดลที่เทรนมาสำหรับหาวัตถุทั่วไปก่อนเพื่อทดสอบ
model = YOLO("models/best.pt") 

st.set_page_config(layout="wide")
st.title("☀️ Helios AI - ผู้ช่วยวิเคราะห์แผงโซลาร์ลอยน้ำ")

uploaded_file = st.file_uploader("อัปโหลดภาพถ่ายความร้อนที่นี่...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image) # แปลงภาพเป็น array เพื่อให้ AI ประมวลผลได้

    st.subheader("ภาพต้นฉบับ")
    st.image(img_array, caption='ภาพที่อัปโหลด', use_container_width=True)

    st.subheader("ผลการวิเคราะห์จาก AI")
    with st.spinner('AI กำลังวิเคราะห์...'):
        # สั่งให้ AI ทำการวิเคราะห์ภาพ
        results = model(img_array)

        # นำผลลัพธ์มาวาดลงบนภาพ
        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption='ภาพที่ AI ตรวจจับ', use_container_width=True)