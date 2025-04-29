import warnings
warnings.filterwarnings("ignore")  # Suppress deprecation warnings

import streamlit as st
from PIL import Image
import os, io

from yolov8_infer import detect_with_yolo
from sam_segment import detect_with_sam

# ——— App Styling ——————————————————————————————————————————
st.markdown("""
    <style>
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #1E90FF;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: grey;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='title'>🤖 Hybrid YOLO+SAM  Model To Detect Automated Plastic Bottle 🤖</div>",
    unsafe_allow_html=True
)

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# File uploader
st.markdown("### 📸 Upload an image to detect plastic bottles", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an image (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("🔍 Detect Plastic Bottles"):
        # YOLO Detection
        with st.spinner("Running YOLO detection..."):
            yolo_image, boxes = detect_with_yolo(image)
            st.image(yolo_image, caption="YOLO Detection", use_container_width=True)
            yolo_path = os.path.join(output_dir, "yolo_output.jpg")
            yolo_image.save(yolo_path)

        # SAM Segmentation
        with st.spinner("Applying SAM segmentation..."):
            sam_image = detect_with_sam(image, boxes)
            st.image(sam_image, caption="YOLO+SAM Detection", use_container_width=True)
            sam_path = os.path.join(output_dir, "yolosam_output.jpg")
            sam_image.save(sam_path)

        # Download Results
        st.markdown("### 📥 Download Results", unsafe_allow_html=True)
        
        yolo_buf = io.BytesIO()
        yolo_image.save(yolo_buf, format="JPEG")
        yolo_buf.seek(0)

        sam_buf = io.BytesIO()
        sam_image.save(sam_buf, format="JPEG")
        sam_buf.seek(0)

        st.download_button(
            "Download YOLO Result", yolo_buf,
            file_name="yolo_result.jpg", mime="image/jpeg"
        )
        st.download_button(
            "Download YOLO+SAM Result", sam_buf,
            file_name="yolosam_result.jpg", mime="image/jpeg"
        )

# Footer
st.markdown("<div class='footer'>Made with streamlit</div>", unsafe_allow_html=True)
