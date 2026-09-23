import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Ensure deterministic behavior
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "research_features.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "svm_All24_RBF.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

def apply_perturbation(rgb_img, p_type, p_level):
    if p_type == 'baseline':
        return rgb_img.copy()
        
    elif p_type == 'brightness':
        # p_level is percentage like -40, -20, 20, 40
        beta = int((p_level / 100.0) * 255)
        return cv2.convertScaleAbs(rgb_img, alpha=1.0, beta=beta)
        
    elif p_type == 'contrast':
        # p_level is percentage like -40, -20, 20, 40
        alpha = 1.0 + (p_level / 100.0)
        beta = 128 * (1.0 - alpha)
        return cv2.convertScaleAbs(rgb_img, alpha=alpha, beta=beta)
        
    elif p_type == 'noise':
        # p_level is std dev: 10 (mild), 25 (medium), 50 (strong)
        noise = np.random.normal(0, p_level, rgb_img.shape).astype(np.int16)
        return np.clip(rgb_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif p_type == 'blur':
        # p_level is kernel size: 3, 7, 15
        return cv2.GaussianBlur(rgb_img, (p_level, p_level), 0)
        
    return rgb_img

def extract_features_from_image(rgb_img):
    try:
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        leaf_mask = isolate_leaf_otsu(rgb_img)
        seg = segment_diseased_region(rgb_img, hsv_img, leaf_mask)
        
        feats_dict, f_vec, f_names = extract_all_features(
            rgb_img, hsv_img, gray_img,
            seg['diseased_mask'], seg['leaf_mask']
        )
        
        # Fallback for missing/NaN values
        for k, v in feats_dict.items():
            if np.isnan(v) or np.isinf(v):
                feats_dict[k] = 0.0
                
        return feats_dict, f_names
    except Exception as e:
        return None, None

def main():
    print("Loading original feature matrix to identify test split...")
    df = pd.read_csv(FEATURES_CSV)
    
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])
    
    _, df_test = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, stratify=df['encoded_label'])
    print(f"Test set size: {len(df_test)}")
    
    # Load Model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    feature_cols = [c for c in df.columns if c not in ['image_path', 'label', 'crop', 'condition', 'domain', 'encoded_label']]
    
    perturbations = [
        ('baseline', 0, 'Baseline'),
        ('brightness', -40, '-40%'),
        ('brightness', -20, '-20%'),
        ('brightness', 20, '+20%'),
        ('brightness', 40, '+40%'),
        ('contrast', -40, '-40%'),
        ('contrast', -20, '-20%'),
        ('contrast', 20, '+20%'),
        ('contrast', 40, '+40%'),
        ('noise', 10, 'Mild'),
        ('noise', 25, 'Medium'),
        ('noise', 50, 'Strong'),
        ('blur', 3, 'Small'),
        ('blur', 7, 'Medium'),
        ('blur', 15, 'Strong')
    ]
    
    results = []
    baseline_acc = 0
    baseline_f1 = 0
    
    for p_type, p_level, p_label in perturbations:
        print(f"\nProcessing Perturbation: {p_type} ({p_label})")
        
        y_true = []
        y_pred = []
        failures = 0
        
        for idx, row in df_test.iterrows():
            img_path = PROJECT_ROOT / row['image_path']
            # Load BGR and convert to RGB
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                failures += 1
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (256, 256))
            
            # Apply Perturbation
            img_pert = apply_perturbation(img_rgb, p_type, p_level)
            
            # Blur specifically to match standard preprocessing before extraction
            img_pert_blurred = cv2.GaussianBlur(img_pert, (3, 3), 0)
            
            # Extract
            feats_dict, f_names = extract_features_from_image(img_pert_blurred)
            
            if feats_dict is None:
                failures += 1
                continue
                
            # Create feature vector
            X_vec = np.array([feats_dict[c] for c in feature_cols]).reshape(1, -1)
            
            pred_encoded = model.predict(X_vec)[0]
            
            y_true.append(row['encoded_label'])
            y_pred.append(pred_encoded)
            
        acc = accuracy_score(y_true, y_pred) if y_true else 0
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else 0
        
        if p_type == 'baseline':
            baseline_acc = acc
            baseline_f1 = f1
            
        results.append({
            'perturbation': p_type,
            'severity': p_label,
            'accuracy': acc,
            'macro_f1': f1,
            'num_images': len(y_true),
            'failures': failures
        })
        print(f"Acc: {acc:.4f} | F1: {f1:.4f} | Failures: {failures}")
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "robustness_perturbations.csv", index=False)
    
    # Calculate Degradation
    degradation = []
    for _, row in df_results.iterrows():
        if row['perturbation'] == 'baseline':
            continue
        
        abs_drop = baseline_f1 - row['macro_f1']
        rel_drop = (abs_drop / baseline_f1) * 100 if baseline_f1 > 0 else 0
        
        degradation.append({
            'perturbation': row['perturbation'],
            'severity': row['severity'],
            'baseline_macro_f1': baseline_f1,
            'perturbed_macro_f1': row['macro_f1'],
            'absolute_drop': abs_drop,
            'relative_drop_percent': rel_drop
        })
        
    df_deg = pd.DataFrame(degradation)
    df_deg.to_csv(RESULTS_DIR / "robustness_degradation.csv", index=False)
    
    # -------------------------------------
    # PLOTS
    # -------------------------------------
    def plot_robustness(p_type, title, filename):
        sub = df_results[df_results['perturbation'] == p_type]
        if sub.empty: return
        
        plt.figure(figsize=(8, 5))
        plt.plot(sub['severity'], sub['macro_f1'], marker='o', linestyle='-', color='b', label='Macro-F1')
        plt.axhline(y=baseline_f1, color='r', linestyle='--', label='Baseline')
        plt.title(title)
        plt.xlabel('Severity')
        plt.ylabel('Macro-F1')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True)
        plt.savefig(FIGURES_DIR / filename)
        plt.close()
        
    plot_robustness('brightness', 'Robustness against Brightness Shift', 'brightness_robustness.png')
    plot_robustness('contrast', 'Robustness against Contrast Shift', 'contrast_robustness.png')
    plot_robustness('noise', 'Robustness against Gaussian Noise', 'noise_robustness.png')
    plot_robustness('blur', 'Robustness against Gaussian Blur', 'blur_robustness.png')
    
    print("\nRobustness evaluation complete!")

if __name__ == "__main__":
    main()
