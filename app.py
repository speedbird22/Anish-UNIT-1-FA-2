import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# Load YOLOv5 model
model = YOLO("best.pt")  # Replace with your actual model path

# Compliance mapping
compliance_map = {
    'Hardhat': '✅ Compliant',
    'Safety Vest': '✅ Compliant',
    'Mask': '✅ Compliant',
    'NO-Hardhat': '❌ Missing Hardhat',
    'NO-Safety Vest': '❌ Missing Vest',
    'NO-Mask': '❌ Missing Mask',
    'Person': '👤 Worker',
    'machinery': '⚙️ Machinery',
    'vehicle': '🚗 Vehicle',
    'Safety Cone': '🟠 Cone'
}

# Streamlit UI setup
st.set_page_config(page_title="Construction PPE Dashboard", page_icon="👷", layout="wide")

with st.sidebar:
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.caption("Built with ❤️ using YOLOv5 and Streamlit")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>👷 PPE Compliance Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a construction site image to detect workers and assess safety gear compliance.</p>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None and len(boxes) > 0:
        data = []
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf = round(float(box.conf[0]) * 100, 2)
            category = compliance_map.get(label, 'Unknown')
            data.append({
                "name": label,
                "confidence": conf,
                "category": category
            })

        df = pd.DataFrame(data)

        # Top detection
        top = df.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🕵️ Detected", top['name'])
        with col2:
            st.metric("📊 Confidence", f"{top['confidence']}%")
        with col3:
            st.metric("🔍 Status", top['category'])

        st.markdown("### 📋 Detection Table")
        st.dataframe(df)

        st.markdown("### 📊 Compliance Summary")
        summary = df['name'].value_counts().to_dict()
        for cls, count in summary.items():
            label = compliance_map.get(cls, cls)
            st.write(f"- {label}: {count}")
    else:
        st.warning("🚫 No PPE-related objects detected. Try another image.")
else:
    st.info("📤 Please upload an image to begin analysis.")
