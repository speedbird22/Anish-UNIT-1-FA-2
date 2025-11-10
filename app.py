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
st.set_page_config(page_title="👷 Construction PPE Dashboard", page_icon="🛠️", layout="centered")
st.markdown("<h1 style='text-align: center;'>👷 Construction Site PPE Compliance</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>📸 Upload an image to detect workers and assess safety compliance.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

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
        category = compliance_map.get(label, '❓ Unknown')

        st.markdown("### 🧾 Top Detection")
        st.success(f"🔍 **Detected:** {label}")
        st.info(f"📊 **Confidence:** {conf}%")
        if category.startswith("✅"):
            st.success(f"🟢 **Compliance:** {category}")
        elif category.startswith("❌"):
            st.warning(f"🔴 **Violation:** {category}")
        else:
            st.info(f"📁 **Category:** {category}")

        # Show full detection table
        st.markdown("### 📋 All Detections")
        st.dataframe(detections[['name', 'confidence', 'class']])

        # Compliance summary
        st.markdown("### 📊 Compliance Summary")
        summary = detections['name'].value_counts().to_dict()
        for cls, count in summary.items():
            label = compliance_map.get(cls, cls)
            st.write(f"🔸 {label}: {count}")
    else:
        st.error("🚫 No PPE-related objects detected. Please try another image.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>🛠️ Built with ❤️ using YOLOv5 and Streamlit</p>", unsafe_allow_html=True)
