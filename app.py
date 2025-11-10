import streamlit as st
import torch
from PIL import Image
import pandas as pd

# Load YOLOv5 model from GitHub
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Compliance mapping for 10 classes
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

# Sidebar
with st.sidebar:
    st.header("📁 Upload Section")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.write("Built with ❤️ using YOLOv5 and Streamlit")

# Main Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>👷 PPE Compliance Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a construction site image to detect workers and assess safety gear compliance.</p>", unsafe_allow_html=True)

# Image and Detection
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    detections = results.pandas().xyxy[0]

    if not detections.empty:
        top = detections.iloc[0]
        label = top['name']
        conf = round(top['confidence'] * 100, 2)
        category = compliance_map.get(label, 'Unknown')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🕵️ Detected", label)
        with col2:
            st.metric("📊 Confidence", f"{conf}%")
        with col3:
            st.metric("🔍 Status", category)

        st.markdown("### 📋 Detection Table")
        st.dataframe(detections[['name', 'confidence', 'class']])

        st.markdown("### 📊 Compliance Summary")
        summary = detections['name'].value_counts().to_dict()
        for cls, count in summary.items():
            label = compliance_map.get(cls, cls)
            st.write(f"- {label}: {count}")
    else:
        st.warning("🚫 No PPE-related objects detected. Try another image.")

else:
    st.info("📤 Please upload an image to begin analysis.")
