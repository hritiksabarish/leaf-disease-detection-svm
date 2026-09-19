import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to sys.path for direct module imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features
from pipeline import run_pipeline, load_model_pack

# ---------------------------------------------------------
# Page Configuration & Modern Agriculture Custom CSS Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Leaf Disease Detection Dashboard | IPA Group 12",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Styling & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
    }
    
    /* Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(27, 67, 50, 0.25);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #d8f3dc;
        font-size: 1.05rem;
        margin: 0;
        opacity: 0.95;
    }
    
    /* Result Card Styling */
    .prediction-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        border-color: #52b788;
        box-shadow: 0 8px 24px rgba(45, 106, 79, 0.12);
    }
    
    .badge-healthy {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    
    .badge-diseased {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-info-box {
        background-color: #ebf5ee;
        border-left: 4px solid #2d6a4f;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1b4332;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Section: Project Metadata & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=70)
    st.title("IPA Mini Project")
    st.markdown("**Group 12** | Semester Project")
    
    st.divider()
    
    st.markdown("### 👥 Group Members")
    st.markdown("""
    - Member 1 (Lead & CV)
    - Member 2 (ML & App)
    - Member 3 (Feature Extraction)
    - Member 4 (Dataset & Docs)
    """)
    
    st.divider()
    
    # Model status check
    model_pack = load_model_pack()
    if model_pack:
        acc = model_pack['metrics']['accuracy'] * 100.0
        st.success(f"✓ Trained SVM Model Loaded\n\n**Test Accuracy:** {acc:.2f}%")
    else:
        st.warning("⚠️ Trained model pack not found. Run `python src/classify.py` first.")

    st.markdown("---")
    st.caption("College Mini Project • Image Processing & ML")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🍃 Leaf Disease Detection System</h1>
    <p>Automated Plant Disease Diagnostics using Image Segmentation, GLCM Texture Features & SVM Classification</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Section: Image Upload or Sample Selection
# ---------------------------------------------------------
col_input1, col_input2 = st.columns([2, 1])

uploaded_file = None
selected_sample = None

with col_input1:
    uploaded_file = st.file_uploader(
        "Upload a Plant Leaf Image (JPG / PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a tomato or potato leaf."
    )

with col_input2:
    st.markdown("##### Or pick a sample image:")
    sample_dir = ROOT_DIR / "data" / "subset"
    sample_images = list(sample_dir.rglob("*.jpg")) + list(sample_dir.rglob("*.png"))
    
    if sample_images:
        sample_options = ["None"] + [f"{p.parent.name} / {p.name}" for p in sample_images[:10]]
        chosen = st.selectbox("Sample Images", sample_options)
        if chosen != "None":
            # Find selected path
            target_name = chosen.split(" / ")[-1]
            for p in sample_images:
                if p.name == target_name:
                    selected_sample = str(p)
                    break

# Determine input source
image_to_process = None
if uploaded_file is not None:
    image_to_process = uploaded_file.read()
elif selected_sample is not None:
    image_to_process = selected_sample

if image_to_process is not None:
    with st.spinner("Processing pipeline: Preprocessing → Segmentation → Feature Extraction → Prediction..."):
        pipeline_res = run_pipeline(image_to_process, model_pack=model_pack)
    
    pred_label = pipeline_res['predicted_label']
    confidence = pipeline_res['confidence']
    is_healthy = "healthy" in pred_label.lower()

    # ---------------------------------------------------------
    # Section 1: Side-by-Side Pipeline Visualizations
    # ---------------------------------------------------------
    st.markdown("### 📸 1. Image Segmentation & Pipeline Stages")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Original Preprocessed**")
        st.image(pipeline_res['prep']['resized_rgb'], use_column_width=True)
        st.caption("Resized 256x256 & Gaussian Blurred")

    with c2:
        st.markdown("**Diseased Region Segmented Mask**")
        st.image(pipeline_res['seg']['diseased_mask'], use_column_width=True, clamp=True)
        st.caption("Otsu Background Removal + HSV K-Means (k=3)")

    with c3:
        st.markdown("**Highlighted Disease Overlay**")
        st.image(pipeline_res['highlighted_overlay'], use_column_width=True)
        st.caption("Lesion Boundary Contours Highlighted in Red")

    st.divider()

    # ---------------------------------------------------------
    # Section 2: Prediction Results & Metrics
    # ---------------------------------------------------------
    st.markdown("### 📊 2. Diagnostic Prediction & Confidence")
    
    badge_class = "badge-healthy" if is_healthy else "badge-diseased"
    icon = "✅" if is_healthy else "🦠"
    
    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        st.markdown(f"""
        <div class="prediction-card">
            <h4>Diagnostic Output:</h4>
            <div class="{badge_class}">
                {icon} {pred_label.replace('___', ' - ')}
            </div>
            <h3 style="margin-top: 1rem; color: #1e293b;">Confidence Score: <b>{confidence:.1f}%</b></h3>
            <p style="color: #64748b; margin-bottom: 0;">Predicted using SVM classifier trained on extracted GLCM texture and color features.</p>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        if pipeline_res['probabilities']:
            st.markdown("**Class Probabilities Breakdown:**")
            df_probs = pd.DataFrame(
                list(pipeline_res['probabilities'].items()),
                columns=["Disease Class", "Probability"]
            ).sort_values(by="Probability", ascending=False)
            
            df_probs["Probability (%)"] = (df_probs["Probability"] * 100).round(1)
            st.dataframe(df_probs[["Disease Class", "Probability (%)"]], use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # Section 3: Extracted Features Breakdown
    # ---------------------------------------------------------
    st.markdown("### 🔬 3. Extracted Feature Vector (Color, Texture & Shape)")
    
    feats = pipeline_res['features_dict']
    
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        st.markdown("##### 🎨 Color Features (HSV)")
        st.json({
            "Mean Hue (H)": round(feats['color_mean_h'], 2),
            "Std Dev Hue": round(feats['color_std_h'], 2),
            "Mean Saturation (S)": round(feats['color_mean_s'], 2),
            "Mean Value (V)": round(feats['color_mean_v'], 2)
        })

    with f_col2:
        st.markdown("##### 🧱 GLCM Texture Features")
        st.json({
            "Contrast": round(feats['glcm_contrast'], 4),
            "Homogeneity": round(feats['glcm_homogeneity'], 4),
            "Energy": round(feats['glcm_energy'], 4),
            "Correlation": round(feats['glcm_correlation'], 4),
            "Dissimilarity": round(feats['glcm_dissimilarity'], 4)
        })

    with f_col3:
        st.markdown("##### 📐 Contour Shape Features")
        st.json({
            "Lesion Area": round(feats['shape_area'], 1),
            "Perimeter": round(feats['shape_perimeter'], 1),
            "Eccentricity": round(feats['shape_eccentricity'], 3),
            "Solidity": round(feats['shape_solidity'], 3),
            "Metric Ratio": round(feats['shape_compactness'], 3)
        })

else:
    st.info("👆 Please upload a leaf image or select a sample image above to see live disease detection results.")
