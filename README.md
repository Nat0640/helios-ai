---
title: Helios AI Analyzer
emoji: ☀️
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---
# ☀️ Helios AI - เครื่องมือวิเคราะห์แผงโซลาร์ลอยน้ำ

แอปพลิเคชันสำหรับอัปโหลดและวิเคราะห์ภาพถ่ายความร้อนจากโดรนสำหรับ Floating Solar โดยใช้ AI ตรวจจับความผิดปกติ

## โหมดการทำงาน
- **วิเคราะห์ภาพเดี่ยว:** แสดงผลการวิเคราะห์, ข้อมูลอุณหภูมิ, และพิกัด GPS บนแผนที่
- **วิเคราะห์เป็นชุด:** ประมวลผลหลายไฟล์พร้อมกันและสรุปผลในรูปแบบตาราง

## เทคโนโลยีที่ใช้
- **Backend:** Python, Streamlit
- **AI Model:** YOLOv8 (Custom Trained)
- **Deployment:** Hugging Face Spaces