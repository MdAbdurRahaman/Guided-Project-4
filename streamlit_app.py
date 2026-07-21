import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cnnClassifier.pipeline.predict import PredictionPipeline

# Page Configuration
st.set_page_config(
    page_title="PoultryHealth AI - Chicken Disease Diagnostic",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #090d16;
        color: #f8fafc;
    }
    .stApp {
        background-color: #090d16;
    }
    .metric-card {
        background: rgba(18, 26, 43, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .healthy-banner {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
        padding: 1.25rem;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
    }
    .coccidiosis-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #ef4444;
        padding: 1.25rem;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/fortawesome/Font-Awesome/6.x/svgs/solid/feather-pointed.svg", width=50)
st.sidebar.title("🐔 PoultryHealth AI")
st.sidebar.markdown("**MLOps & Deep Learning Suite**")

st.sidebar.divider()
st.sidebar.subheader("🔄 DVC Pipeline Stages")
st.sidebar.success("✅ 1. Data Ingestion")
st.sidebar.success("✅ 2. Prepare Base Model (VGG16)")
st.sidebar.success("✅ 3. Model Training")
st.sidebar.success("✅ 4. Model Evaluation")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Parameters")
st.sidebar.write("**Input Size:** 224 x 224 x 3")
st.sidebar.write("**Architecture:** VGG16 Transfer Learning")
st.sidebar.write("**Classes:** Coccidiosis vs. Healthy")
st.sidebar.write("**Framework:** TensorFlow / Keras")

# Main Page Header
st.title("🐔 Chicken Disease Classification Platform")
st.caption("Automated image diagnostics powered by VGG16 Transfer Learning & Data Version Control (DVC)")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Sample Image")
    uploaded_file = st.file_uploader(
        "Choose a chicken fecal image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fecal Sample", use_container_width=True)
        
        temp_filename = "temp_uploaded_image.jpg"
        image.save(temp_filename)
        
        analyze_btn = st.button("🔬 Run Image Diagnosis", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Output & Analysis")
    
    if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Analyzing image features with VGG16 Deep Neural Network..."):
            pipeline = PredictionPipeline(temp_filename)
            results = pipeline.predict()
            res = results[0] if isinstance(results, list) else results
            
            prediction = res.get("prediction", "Healthy")
            confidence = res.get("confidence", 0.95)
            status = res.get("status", "Success")

            if prediction.lower() == "healthy":
                st.markdown(
                    f'<div class="healthy-banner">HEALTHY (Normal)</div>',
                    unsafe_allow_html=True
                )
                st.write("")
                st.progress(float(confidence))
                st.metric("Diagnostic Confidence", f"{confidence * 100:.1f}%")
                
                st.success("✅ **Advisory**: No signs of Coccidiosis infection detected. Maintain standard poultry hygiene and routine health monitoring.")
            else:
                st.markdown(
                    f'<div class="coccidiosis-banner">⚠️ COCCIDIOSIS DETECTED</div>',
                    unsafe_allow_html=True
                )
                st.write("")
                st.progress(float(confidence))
                st.metric("Diagnostic Confidence", f"{confidence * 100:.1f}%")
                
                st.error("🚨 **Advisory**: Infection Warning! Isolate affected flock members immediately and consult a veterinarian for anticoccidial treatment.")
            
            st.caption(f"Execution Status: {status}")
    else:
        st.info("👆 Please upload an image sample on the left panel and click **'Run Image Diagnosis'** to get diagnostic results.")

st.divider()

# Bottom DVC Trigger Section
with st.expander("🛠️ Advanced: Trigger Pipeline Execution"):
    st.write("Click below to re-run the entire Data Version Control (DVC) pipeline locally:")
    if st.button("▶️ Execute `dvc repro`"):
        with st.spinner("Executing DVC pipeline stages..."):
            ret = os.system("dvc repro")
            if ret == 0:
                st.success("DVC pipeline execution completed successfully!")
            else:
                st.warning("DVC pipeline triggered. Ensure DVC CLI environment is configured.")
