import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import Counter
import pandas as pd
import subprocess
import json
import os
import tempfile

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

# ----------------- THERMAL ANALYSIS FUNCTIONS -----------------
def get_h20t_thermal_matrix(image_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_filename = tmp.name

        command = [
            'exiftool', '-j', '-PlanckR1', '-PlanckR2', '-PlanckB', 
            '-PlanckF', '-PlanckO', tmp_filename
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]

        R1 = metadata.get('PlanckR1')
        R2 = metadata.get('PlanckR2')
        B = metadata.get('PlanckB')
        F = metadata.get('PlanckF')
        O = metadata.get('PlanckO')
        
        # ตรวจสอบว่าได้ค่าที่จำเป็นครบหรือไม่
        if not all([R1, R2, B, F, O]):
             st.warning("ไม่พบค่า Planck's constants ที่จำเป็นใน metadata ของภาพ")
             return None

        command_raw = ['exiftool', '-b', '-RawThermalImage', tmp_filename]
        raw_result = subprocess.run(command_raw, capture_output=True, check=True)
        raw_thermal_bytes = raw_result.stdout

        raw_data = np.frombuffer(raw_thermal_bytes, dtype=np.uint16)
        
        # H20T มีขนาด 640x512
        if raw_data.size != 640 * 512:
            st.warning(f"ขนาดของ Raw Thermal Image ไม่ถูกต้อง (ได้ {raw_data.size}, คาดหวัง {640*512})")
            return None
            
        raw_data = raw_data.reshape((512, 640))

        temp_kelvin = B / np.log(R1 / (raw_data.astype(np.float32) - O) + F)
        temp_celsius = temp_kelvin - 273.15

        return temp_celsius

    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError, IndexError) as e:
        st.warning(f"ไม่สามารถอ่านข้อมูล Radiometric ได้: {e}. ไฟล์นี้อาจไม่ใช่ R-JPEG จาก H20T หรือไม่ได้ติดตั้ง ExifTool")
        return None
    finally:
        if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
            os.remove(tmp_filename)

def analyze_anomalies_with_temp(yolo_results, temp_matrix, original_img_size):
    if temp_matrix is None:
        return [], None

    T_ref = np.median(temp_matrix)
    anomalies_details = []
    res = yolo_results[0]
    
    thermal_img_h, thermal_img_w = temp_matrix.shape
    original_w, original_h = original_img_size

    # คำนวณอัตราส่วนสำหรับปรับพิกัด
    w_ratio = thermal_img_w / original_w
    h_ratio = thermal_img_h / original_h

    for box in res.boxes:
        class_name = res.names[int(box.cls)]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0]

        # ปรับพิกัด Bounding Box ให้ตรงกับขนาดของ Thermal Matrix
        tx1 = int(x1 * w_ratio)
        ty1 = int(y1 * h_ratio)
        tx2 = int(x2 * w_ratio)
        ty2 = int(y2 * h_ratio)

        anomaly_zone = temp_matrix[ty1:ty2, tx1:tx2]
        if anomaly_zone.size == 0: continue

        T_max = np.max(anomaly_zone)
        delta_T = T_max - T_ref

        anomalies_details.append({
            "class": class_name,
            "confidence": confidence,
            "T_max_C": T_max,
            "Delta_T_C": delta_T
        })

    return anomalies_details, T_ref

# ----------------- UI & MAIN LOGIC -----------------
model = load_yolo_model(MODEL_PATH)
st.title("☀️ Helios AI v2.0 - Advanced Thermal Analysis")

with st.sidebar:
    st.header("⚙️ ตั้งค่าการทำงาน")
    app_mode = st.radio(
        "เลือกโหมดการทำงาน",
        ("วิเคราะห์ภาพเดี่ยว (Single Image Analysis)", "วิเคราะห์เป็นชุด (Batch Processing)")
    )
    st.header("🛠️ ตั้งค่าการวิเคราะห์")
    if app_mode == "วิเคราะห์ภาพเดี่ยว (Single Image Analysis)":
        uploaded_files = st.file_uploader(
            "อัปโหลดภาพ R-JPEG", type=["jpg", "jpeg"], accept_multiple_files=False)
    else:
        uploaded_files = st.file_uploader(
            "อัปโหลดภาพ R-JPEG (ได้หลายไฟล์)", type=["jpg", "jpeg"], accept_multiple_files=True)
            
    confidence_threshold = st.slider("เลือกค่าความมั่นใจ (Confidence)", 0.0, 1.0, 0.40, 0.05)
    st.info("รองรับไฟล์ R-JPEG จาก DJI H20T")


if model is None:
    st.error("ไม่สามารถโหลดโมเดล AI ได้ กรุณาตรวจสอบว่าไฟล์ `models/best.pt` อยู่ในตำแหน่งที่ถูกต้อง")
    st.stop()

if not uploaded_files:
    st.info("กรุณาเลือกโหมดและอัปโหลดภาพในแถบเมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")
    st.stop()

# --- โหมดที่ 1: วิเคราะห์ภาพเดี่ยว ---
if app_mode == "วิเคราะห์ภาพเดี่ยว (Single Image Analysis)":
    uploaded_file = uploaded_files
    image = Image.open(uploaded_file)
    image_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ภาพต้นฉบับ")
        st.image(image, caption=f"ชื่อไฟล์: {uploaded_file.name}", use_container_width=True)
    
    yolo_results = model(image, conf=confidence_threshold, verbose=False)
    temp_matrix = get_h20t_thermal_matrix(image_bytes)
    anomalies, T_ref = analyze_anomalies_with_temp(yolo_results, temp_matrix, image.size)

    with col2:
        st.subheader("ผลการวิเคราะห์จาก AI")
        annotated_frame = yolo_results[0].plot()
        st.image(annotated_frame, caption='ภาพที่ AI ตรวจจับ', use_container_width=True)
        
    st.header("สรุปผลการตรวจจับเชิงลึก")
    if temp_matrix is None:
        st.error("ไม่สามารถประมวลผลข้อมูลอุณหภูมิได้")
    elif not anomalies:
        st.success(f"ไม่พบความผิดปกติใดๆ (ที่ Confidence > {confidence_threshold:.0%})")
        st.info(f"อุณหภูมิอ้างอิงของแผง (T_ref): **{T_ref:.1f}°C**")
    else:
        st.warning(f"ตรวจพบความผิดปกติ {len(anomalies)} จุด | T_ref: {T_ref:.1f}°C")
        df = pd.DataFrame(anomalies)
        df_display = df.style.format({
            "confidence": "{:.1%}", "T_max_C": "{:.1f}°C", "Delta_T_C": "{:+.1f}°C"
        }).background_gradient(cmap='Reds', subset=['T_max_C', 'Delta_T_C'])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# --- โหมดที่ 2: วิเคราะห์เป็นชุด ---
elif app_mode == "วิเคราะห์เป็นชุด (Batch Processing)":
    st.header("ผลการวิเคราะห์แบบชุด")
    results_list = []
    total_detections = Counter()
    files_with_anomalies_count = 0

    progress_bar = st.progress(0, text="กำลังเริ่มต้น...")

    for i, uploaded_file in enumerate(uploaded_files):
        progress_text = f"กำลังประมวลผล: {uploaded_file.name}"
        progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
        
        image = Image.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
        
        yolo_results = model(image, conf=confidence_threshold, verbose=False)
        temp_matrix = get_h20t_thermal_matrix(image_bytes)
        anomalies, T_ref = analyze_anomalies_with_temp(yolo_results, temp_matrix, image.size)
        
        if not anomalies:
            results_list.append({
                "ชื่อไฟล์": uploaded_file.name, "สถานะ": "✅ ปกติ", "ประเภทที่พบ": "-",
                "จำนวน": 0, "T_max (°C)": f"{T_ref:.1f}" if T_ref else "-", "ΔT (°C)": "-"
            })
        else:
            files_with_anomalies_count += 1
            highest_anomaly = max(anomalies, key=lambda x: x['T_max_C'])
            current_counts = Counter(a['class'] for a in anomalies)
            total_detections.update(current_counts)
            results_list.append({
                "ชื่อไฟล์": uploaded_file.name, "สถานะ": "⚠️ พบความผิดปกติ",
                "ประเภทที่พบ": ", ".join(current_counts.keys()), "จำนวน": len(anomalies),
                "T_max (°C)": f"{highest_anomaly['T_max_C']:.1f}", "ΔT (°C)": f"{highest_anomaly['Delta_T_C']:.1f}"
            })

    progress_bar.empty()

    st.subheader("สรุปผลรายไฟล์")
    df = pd.DataFrame(results_list)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("สรุปภาพรวมทั้งหมด (Overall Summary)")
    col1, col2, col3 = st.columns(3)
    col1.metric("จำนวนไฟล์ทั้งหมด", f"{len(uploaded_files)} ไฟล์")
    col2.metric("ไฟล์ที่พบความผิดปกติ", f"{files_with_anomalies_count} ไฟล์")
    col3.metric("ไฟล์ปกติ", f"{len(uploaded_files) - files_with_anomalies_count} ไฟล์")

    if total_detections:
        st.markdown("---")
        st.write("#### จำนวนความผิดปกติแต่ละประเภทที่พบทั้งหมด:")
        metric_cols = st.columns(len(total_detections))
        for i, (name, count) in enumerate(total_detections.items()):
            metric_cols[i].metric(label=name, value=f"{count} จุด")