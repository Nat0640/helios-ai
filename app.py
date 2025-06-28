import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from collections import Counter
import pandas as pd # เพิ่ม Library Pandas เพื่อแสดงตารางสรุปสวยๆ
from PIL.ExifTags import TAGS, GPSTAGS

# ----------------- CONFIGURATION -----------------
st.set_page_config(layout="wide")
MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        return None

model = load_yolo_model(MODEL_PATH)

# ----------------- GPS ANALYSIS FUNCTION (ส่วนใหม่) -----------------
def get_gps_data(image):
    """
    แกะข้อมูล EXIF จากภาพเพื่อหาพิกัด GPS และแปลงเป็น Decimal Degrees
    """
    gps_data = {}
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None

        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for key, val in value.items():
                    gps_tag_name = GPSTAGS.get(key, key)
                    gps_data[gps_tag_name] = val
                break
        
        if not gps_data:
            return None

        # --- ฟังก์ชันแปลง DMS to DD ---
        def dms_to_dd(dms, ref):
            degrees = dms[0]
            minutes = dms[1] / 60.0
            seconds = dms[2] / 3600.0
            dd = degrees + minutes + seconds
            if ref in ['S', 'W']:
                dd *= -1
            return dd

        lat_dms = gps_data.get('GPSLatitude')
        lon_dms = gps_data.get('GPSLongitude')
        lat_ref = gps_data.get('GPSLatitudeRef')
        lon_ref = gps_data.get('GPSLongitudeRef')
        alt_rational = gps_data.get('GPSAltitude')

        if lat_dms and lon_dms and lat_ref and lon_ref:
            latitude = dms_to_dd(lat_dms, lat_ref)
            longitude = dms_to_dd(lon_dms, lon_ref)
            # แปลง IFDRational เป็น float ถ้ามีค่า
            altitude = float(alt_rational) if alt_rational is not None else None
            
            return {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude
            }
        return None

    except Exception:
        return None

# ----------------- UI & LAYOUT -----------------
st.title("☀️ Helios AI - เครื่องมือวิเคราะห์แผงโซลาร์ลอยน้ำ")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ ตั้งค่าการทำงาน")
    
    # 1. เพิ่มตัวเลือกโหมด
    app_mode = st.radio(
        "เลือกโหมดการทำงาน",
        ("วิเคราะห์ภาพเดี่ยว (Single Image Analysis)", "วิเคราะห์เป็นชุด (Batch Processing)")
    )
    
    st.header("🛠️ ตั้งค่าการวิเคราะห์")
    
    # 2. ปรับ File Uploader ให้รองรับหลายไฟล์
    if app_mode == "วิเคราะห์ภาพเดี่ยว (Single Image Analysis)":
        uploaded_files = st.file_uploader(
            "อัปโหลดภาพถ่ายความร้อน", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False # โหมดนี้รับทีละไฟล์
        )
    else:
        uploaded_files = st.file_uploader(
            "อัปโหลดภาพถ่ายความร้อน (ได้หลายไฟล์)", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True # โหมดนี้รับได้หลายไฟล์
        )

    confidence_threshold = st.slider(
        "เลือกค่าความมั่นใจ (Confidence)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40,
        step=0.05
    )
    st.info("AI จะแสดงเฉพาะผลลัพธ์ที่มีค่าความมั่นใจสูงกว่าที่เลือกไว้")

# ----------------- MAIN LOGIC (ปรับปรุง) -----------------

# --- โหมดที่ 1: วิเคราะห์ภาพเดี่ยว ---
if app_mode == "วิเคราะห์ภาพเดี่ยว (Single Image Analysis)":
    if uploaded_files and model:
        uploaded_file = uploaded_files 
        image = Image.open(uploaded_file)
        
        # --- ส่วนที่เพิ่มเข้ามา: เรียกใช้ฟังก์ชัน GPS ---
        gps_info = get_gps_data(image)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ภาพต้นฉบับและข้อมูล")
            st.image(image, caption=f"ชื่อไฟล์: {uploaded_file.name}", use_container_width=True)
            
            # --- แสดงผลข้อมูล GPS ---
            if gps_info:
                st.success("🛰️ ตรวจพบข้อมูล GPS")
                st.write(f"**Latitude:** `{gps_info['latitude']:.6f}`")
                st.write(f"**Longitude:** `{gps_info['longitude']:.6f}`")
                if gps_info['altitude']:
                    st.write(f"**Altitude:** `{gps_info['altitude']:.2f}` เมตร")
                
                # สร้าง Link ไป Google Maps
                gmaps_link = f"https://www.google.com/maps?q={gps_info['latitude']},{gps_info['longitude']}"
                st.markdown(f"**[📍 เปิดใน Google Maps]({gmaps_link})**", unsafe_allow_html=True)
                
                # แสดงแผนที่ในแอปเลย (Bonus!)
                map_df = pd.DataFrame([gps_info], columns=['latitude', 'longitude'])
                st.map(map_df, zoom=18) # ปรับ zoom ได้
            else:
                st.warning("ไม่พบข้อมูล GPS ในไฟล์นี้")

        with col2:
            st.subheader("ผลการวิเคราะห์จาก AI")
            with st.spinner('AI กำลังวิเคราะห์...'):
                results = model(image, conf=confidence_threshold)
                annotated_frame = results[0].plot() # ใช้วิธี plot ง่ายๆ ของ YOLO ในโหมดนี้
                st.image(annotated_frame, caption='ภาพที่ AI ตรวจจับ', use_container_width=True)
        
        # สรุปผลสำหรับภาพเดี่ยว
        detection_counts = Counter(results[0].names[int(c)] for c in results[0].boxes.cls)
        st.header("สรุปผลการตรวจจับ")
        if not detection_counts:
            st.success(f"ไม่พบความผิดปกติใดๆ ที่มีค่าความมั่นใจสูงกว่า {confidence_threshold:.0%}")
        else:
            st.warning(f"ตรวจพบความผิดปกติทั้งหมด {sum(detection_counts.values())} จุด")
            metric_cols = st.columns(len(detection_counts))
            for i, (name, count) in enumerate(detection_counts.items()):
                with metric_cols[i]:
                    st.metric(label=f"ประเภท: {name}", value=f"{count} จุด")

# --- โหมดที่ 2: วิเคราะห์เป็นชุด ---
elif app_mode == "วิเคราะห์เป็นชุด (Batch Processing)":
    if uploaded_files and model:
        st.header("ผลการวิเคราะห์แบบชุด")
        
        # เตรียมตัวแปรสำหรับเก็บผลลัพธ์ทั้งหมด
        results_list = []
        total_detections = Counter()
        files_with_anomalies = 0

        # ใช้ Progress Bar เพื่อให้ผู้ใช้เห็นความคืบหน้า
        progress_bar = st.progress(0, text="กำลังเริ่มต้น...")

        for i, uploaded_file in enumerate(uploaded_files):
            # อัปเดต Progress Bar
            progress_text = f"กำลังประมวลผลไฟล์: {uploaded_file.name} ({i+1}/{len(uploaded_files)})"
            progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
            
            image = Image.open(uploaded_file)
            # --- ส่วนที่เพิ่มเข้ามา: ดึงข้อมูล GPS ในลูป ---
            gps_info = get_gps_data(image)
            
            # รันโมเดล (verbose=False เพื่อไม่ให้ log รกหน้าจอ)
            results = model(image, conf=confidence_threshold, verbose=False)
            
            # นับจำนวนที่ตรวจพบในไฟล์นี้
            current_detections = Counter(results[0].names[int(c)] for c in results[0].boxes.cls)
            
            if not current_detections:
                summary_text = "✅ ไม่พบความผิดปกติ"
            else:
                summary_text = "⚠️ " + ", ".join([f"{name}: {count} จุด" for name, count in current_detections.items()])
                total_detections.update(current_detections)
                files_with_anomalies += 1
            
            file_result = {
                "ชื่อไฟล์ (File Name)": uploaded_file.name,
                "ผลการตรวจสอบ (Result)": summary_text
            }
            if gps_info:
                file_result["Latitude"] = f"{gps_info['latitude']:.6f}"
                file_result["Longitude"] = f"{gps_info['longitude']:.6f}"
                file_result["Google Maps"] = f"https://www.google.com/maps?q={gps_info['latitude']},{gps_info['longitude']}"
            else:
                file_result["Latitude"] = "-"
                file_result["Longitude"] = "-"
                file_result["Google Maps"] = "-"
            
            results_list.append(file_result)
        
        progress_bar.empty() # ซ่อน Progress Bar เมื่อเสร็จ

        # --- แสดงตารางสรุปผลรายไฟล์ ---
        st.subheader("สรุปผลรายไฟล์")
        df = pd.DataFrame(results_list)

        # --- ตั้งค่าให้คอลัมน์ Google Maps เป็น Link ที่คลิกได้ ---
        st.dataframe(
            df, 
            column_config={
                "Google Maps": st.column_config.LinkColumn(
                    "แผนที่",
                    display_text="📍 เปิดแผนที่"
                )
            },
            use_container_width=True, 
            hide_index=True
        )        

        # --- แสดงสรุปภาพรวมทั้งหมด ---
        st.subheader("สรุปภาพรวมทั้งหมด (Overall Summary)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("จำนวนไฟล์ทั้งหมด", f"{len(uploaded_files)} ไฟล์")
        with col2:
            st.metric("ไฟล์ที่พบความผิดปกติ", f"{files_with_anomalies} ไฟล์")
        with col3:
            st.metric("ไฟล์ปกติ", f"{len(uploaded_files) - files_with_anomalies} ไฟล์")

        if total_detections:
            st.markdown("---")
            st.write("#### จำนวนความผิดปกติแต่ละประเภทที่พบทั้งหมด:")
            metric_cols = st.columns(len(total_detections))
            for i, (name, count) in enumerate(total_detections.items()):
                 with metric_cols[i]:
                    st.metric(label=name, value=f"{count} จุด")


if not uploaded_files:
    st.info("กรุณาเลือกโหมดและอัปโหลดภาพในแถบเมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")

if not model:
     st.error("ไม่สามารถโหลดโมเดล AI ได้ กรุณาตรวจสอบว่าไฟล์ `models/best.pt` อยู่ในตำแหน่งที่ถูกต้อง")