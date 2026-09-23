import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features
from recommendations import get_recommendation
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = PROJECT_ROOT / "results" / "models" / "svm_All24_RBF.pkl"
FEATURES_CSV = PROJECT_ROOT / "data" / "research_features.csv"

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------------------------
# Caching
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}. Ensure Phase 4 has been completed.")
        st.stop()
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_label_encoder():
    if not FEATURES_CSV.exists():
        st.error(f"Features CSV not found at {FEATURES_CSV}. Ensure Phase 2 has been completed.")
        st.stop()
    df = pd.read_csv(FEATURES_CSV)
    le = LabelEncoder()
    le.fit(df['label'].values)
    return le, [c for c in df.columns if c not in ['image_path', 'label', 'crop', 'condition', 'domain']]

# -------------------------------------------------------------------
# Core Pipeline (Do NOT modify research logic)
# -------------------------------------------------------------------
def extract_features_pipeline(img_rgb_array):
    """
    Reuses the exact frozen research pipeline.
    """
    # Simulate load_and_preprocess_image which normally reads from disk.
    # The pipeline expects BGR read, then convert to RGB and resize.
    # Here we are given RGB from PIL, so we just resize and blur.
    
    resized_rgb = cv2.resize(img_rgb_array, (256, 256))
    blurred_rgb = cv2.GaussianBlur(resized_rgb, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2GRAY)
    
    # Segmentation
    leaf_mask = isolate_leaf_otsu(blurred_rgb)
    seg = segment_diseased_region(blurred_rgb, hsv, leaf_mask)
    
    # Feature extraction
    feats_dict, f_vec, f_names = extract_all_features(
        blurred_rgb, hsv, gray,
        seg['diseased_mask'], seg['leaf_mask']
    )
    
    # Clean NaN/Inf as done in the benchmark
    for k, v in feats_dict.items():
        if np.isnan(v) or np.isinf(v):
            feats_dict[k] = 0.0
            
    return feats_dict, seg['leaf_mask'], seg['diseased_mask']

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def main():
    st.title("Leaf Disease Detection using Lightweight Multi-Crop SVM")
    st.markdown("##### Multi-crop disease classification using Color, GLCM Texture and Shape features")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About the project")
    st.sidebar.markdown(
        "This tool provides automated image-based classification for research and "
        "educational purposes. It evaluates the robustness of handcrafted-feature SVMs."
    )
    
    st.sidebar.header("Model Information")
    st.sidebar.markdown(
        """
        - **Model**: RBF SVM
        - **Features**: 24
        - **Dataset**: 1,952 images
        - **Crops**: 3 (Pepper, Potato, Tomato)
        - **Classes**: 10
        - **Evaluation config**: Primary evaluated configuration
        """
    )
    
    model = load_model()
    le, expected_features = load_label_encoder()
    
    # Main Area
    st.subheader("1. Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
            
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Leaf", type="primary"):
            st.subheader("2. Prediction Result")
            
            with st.spinner("Running inference pipeline..."):
                t_start = time.perf_counter()
                
                # 1. Feature Extraction
                t0_feat = time.perf_counter()
                try:
                    feats_dict, leaf_mask, diseased_mask = extract_features_pipeline(img_array)
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.stop()
                t1_feat = time.perf_counter()
                
                # Ensure 24 features are exactly in the expected order
                try:
                    X_vec = np.array([feats_dict[c] for c in expected_features]).reshape(1, -1)
                except KeyError as e:
                    st.error(f"Feature mapping failed. Missing expected feature: {e}")
                    st.stop()
                    
                # 2. SVM Prediction
                t0_pred = time.perf_counter()
                try:
                    pred_encoded = model.predict(X_vec)[0]
                    predicted_label = le.inverse_transform([pred_encoded])[0]
                    
                    confidence = "Not available (Probability=False in model)"
                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(X_vec)[0]
                            confidence = f"{probs[pred_encoded] * 100:.1f}%"
                        except:
                            pass
                            
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    st.stop()
                t1_pred = time.perf_counter()
                
                t_end = time.perf_counter()
                
            # Render Prediction
            rec = get_recommendation(predicted_label)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Crop**: {rec.get('crop', 'Unknown')}")
                st.markdown(f"**Predicted disease**: {rec.get('disease', predicted_label)}")
            with col2:
                status = rec.get('status', 'unknown')
                status_color = "green" if status == "healthy" else "red" if status == "diseased" else "orange"
                st.markdown(f"**Status**: :{status_color}[{status.upper()}]")
                st.markdown(f"**Model confidence**: {confidence}")
                
            st.info("Note: Classifier confidence is an algorithmic metric, not clinical/agricultural certainty.")
            
            # Masks visualization
            with st.expander("Show Leaf and Lesion Segmentations"):
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.image(leaf_mask, caption="Otsu Leaf Mask (Preview)")
                with m_col2:
                    st.image(diseased_mask, caption="K-Means Lesion Mask (Preview)")
            
            st.markdown("---")
            st.subheader("3. Treatment / Management Recommendation")
            
            st.markdown(f"**Summary:** {rec.get('summary', '')}")
            
            if status == 'healthy':
                st.markdown("#### Preventive Practices")
                for p in rec.get('prevention', []):
                    st.markdown(f"- {p}")
                    
                st.markdown("#### Observation & Monitoring")
                for m in rec.get('management', []):
                    st.markdown(f"- {m}")
            else:
                st.markdown("#### Management Practices")
                for m in rec.get('management', []):
                    st.markdown(f"- {m}")
                    
                st.markdown("#### Prevention")
                for p in rec.get('prevention', []):
                    st.markdown(f"- {p}")
                    
            st.warning(f"**Caution:** {rec.get('caution', '')}")
            
            if rec.get('source'):
                st.markdown("#### Sources")
                for s in rec['source']:
                    st.markdown(f"- [{s['title']}]({s['url']}) ({s['organization']})")
                    
            st.markdown("---")
            st.subheader("4. Technical Information")
            st.markdown("*Current inference timing (single image)*")
            feat_ms = (t1_feat - t0_feat) * 1000
            pred_ms = (t1_pred - t0_pred) * 1000
            end_ms = (t_end - t_start) * 1000
            st.text(f"Feature extraction time : {feat_ms:.2f} ms")
            st.text(f"SVM prediction time     : {pred_ms:.2f} ms")
            st.text(f"End-to-end inference    : {end_ms:.2f} ms")
            
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: This tool provides automated image-based classification for research and "
        "educational purposes. Predictions may be incorrect and should not be treated as definitive "
        "agricultural diagnosis. Verify important decisions with qualified agricultural professionals "
        "and applicable local guidance."
    )

if __name__ == "__main__":
    main()
