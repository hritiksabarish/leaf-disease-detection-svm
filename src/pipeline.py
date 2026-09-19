import os
import glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "model.joblib"

def load_model_pack():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(str(MODEL_PATH))

def run_pipeline(image_input, model_pack=None, save_output_name=None):
    """
    Runs the complete end-to-end pipeline on an image:
      1. Preprocessing
      2. Segmentation
      3. Feature Extraction
      4. Disease Classification
      5. Side-by-side 3-panel Visualization Generation

    Returns:
        results_dict containing intermediate masks, feature vector, predicted label, confidence, and visualization image array.
    """
    # 1. Preprocessing
    prep = load_and_preprocess_image(image_input)
    rgb_img = prep['blurred_rgb']
    hsv_img = prep['hsv']
    gray_img = prep['gray']

    # 2. Segmentation
    leaf_mask = isolate_leaf_otsu(rgb_img)
    seg = segment_diseased_region(rgb_img, hsv_img, leaf_mask)
    diseased_mask = seg['diseased_mask']
    highlighted_overlay = seg['highlighted_overlay']

    # 3. Feature Extraction
    feats_dict, f_vec, f_names = extract_all_features(
        rgb_img, hsv_img, gray_img, diseased_mask, leaf_mask
    )

    # 4. Classification Prediction
    predicted_label = "Unknown"
    confidence = 0.0
    probabilities = {}

    if model_pack is None:
        model_pack = load_model_pack()

    if model_pack is not None:
        model = model_pack['model']
        scaler = model_pack['scaler']
        le = model_pack['label_encoder']

        f_vec_scaled = scaler.transform(f_vec.reshape(1, -1))
        pred_idx = model.predict(f_vec_scaled)[0]
        predicted_label = le.inverse_transform([pred_idx])[0]

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(f_vec_scaled)[0]
            confidence = float(np.max(probs)) * 100.0
            probabilities = {le.classes_[i]: float(probs[i]) for i in range(len(probs))}
        else:
            confidence = 100.0
            probabilities = {predicted_label: 1.0}

    # 5. Build Side-by-Side 3-Panel Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original Image
    axes[0].imshow(prep['resized_rgb'])
    axes[0].set_title("1. Preprocessed Original", fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Panel 2: Segmented Diseased Mask
    axes[1].imshow(diseased_mask, cmap='hot')
    axes[1].set_title("2. Diseased Spot Mask", fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')

    # Panel 3: Highlighted Overlay + Prediction Badge
    axes[2].imshow(highlighted_overlay)
    axes[2].set_title(f"3. Detection Overlay\n[{predicted_label} ({confidence:.1f}%)]", 
                     fontsize=12, fontweight='bold', pad=10, color='darkgreen')
    axes[2].axis('off')

    plt.tight_layout()

    # Convert figure to array / save to results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_output_name:
        out_path = RESULTS_DIR / save_output_name
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        print(f"[OK] Saved pipeline visualization to: {out_path}")

    # Render figure to numpy buffer for downstream app usage
    fig.canvas.draw()
    try:
        viz_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        viz_rgb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        viz_rgb = viz_rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:4]
    plt.close(fig)

    return {
        'prep': prep,
        'seg': seg,
        'features_dict': feats_dict,
        'feature_vector': f_vec,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'probabilities': probabilities,
        'viz_rgb': viz_rgb,
        'highlighted_overlay': highlighted_overlay
    }

def run_test_sample_pipeline(num_samples=5):
    """
    Runs pipeline on sample images and saves side-by-side visualizations to results/.
    """
    model_pack = load_model_pack()
    subset_dir = PROJECT_ROOT / "data" / "subset"
    
    sample_images = list(subset_dir.rglob("*.jpg")) + list(subset_dir.rglob("*.png"))
    if not sample_images:
        print("[!] No dataset images found in data/subset/.")
        return

    print(f"\n--- Running Pipeline on {min(num_samples, len(sample_images))} Sample Images ---")
    np.random.seed(42)
    selected = np.random.choice(sample_images, min(num_samples, len(sample_images)), replace=False)

    for idx, img_p in enumerate(selected):
        out_name = f"pipeline_sample_{idx+1}.png"
        res = run_pipeline(str(img_p), model_pack=model_pack, save_output_name=out_name)
        print(f" Sample {idx+1} [{img_p.name}]: Predicted -> '{res['predicted_label']}' (Conf: {res['confidence']:.1f}%)")

if __name__ == "__main__":
    run_test_sample_pipeline(5)
