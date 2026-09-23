import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SUBSET_DIR = PROJECT_ROOT / "data" / "subset"
RESULTS_DIR = PROJECT_ROOT / "results"

def build_dataset(data_dir=DATA_SUBSET_DIR):
    """
    Iterates through image directories in data_dir, extracts feature vectors and labels.
    """
    X = []
    y = []
    image_paths = []
    
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes in dataset: {classes}")

    total_physical = 0
    total_extracted = 0

    for cls in classes:
        cls_dir = data_dir / cls
        # Use set to avoid duplicates on case-insensitive systems
        files_set = set(cls_dir.glob("*.jpg")).union(set(cls_dir.glob("*.png"))).union(set(cls_dir.glob("*.JPG"))).union(set(cls_dir.glob("*.JPEG"))).union(set(cls_dir.glob("*.jpeg")))
        files = list(files_set)
        
        print(f"Processing class '{cls}': {len(files)} images...")
        total_physical += len(files)
        
        for idx, img_path in enumerate(files):
            try:
                prep = load_and_preprocess_image(str(img_path))
                leaf_mask = isolate_leaf_otsu(prep['blurred_rgb'])
                seg = segment_diseased_region(prep['blurred_rgb'], prep['hsv'], leaf_mask)
                _, f_vec, f_names = extract_all_features(
                    prep['blurred_rgb'], prep['hsv'], prep['gray'],
                    seg['diseased_mask'], seg['leaf_mask']
                )
                
                X.append(f_vec)
                y.append(cls)
                image_paths.append(str(img_path))
                total_extracted += 1
            except Exception as e:
                print(f"  [!] Failed to process {img_path.name}: {e}")

    print(f"\nPhysical image files found: {total_physical}")
    print(f"Samples passed to feature extraction: {total_extracted}")
    
    # Ensure no duplicates in final path list
    assert len(image_paths) == len(set(image_paths)), "Duplicate paths found in final list!"

    return np.array(X), np.array(y), feature_names_list if 'feature_names_list' in locals() else f_names, image_paths

def train_and_evaluate(data_dir=DATA_SUBSET_DIR):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- 1. Building Feature Matrix ---")
    X, y, f_names, paths = build_dataset(data_dir)
    print(f"Extracted {X.shape[0]} feature vectors of dimension {X.shape[1]}")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    # 80/20 Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM Model with RBF Kernel
    print("\n--- 2. Training SVM Classifier ---")
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions & Evaluation
    y_pred = svm_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n================ MODEL EVALUATION METRICS ================")
    print(f" Accuracy:  {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}%")
    print(f" Recall:    {rec * 100:.2f}%")
    print(f" F1 Score:  {f1 * 100:.2f}%")
    print("==========================================================")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True
    )
    plt.title('Confusion Matrix - Leaf Disease Detection (SVM)', fontsize=14, pad=15)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix to: {cm_path}")

    # Also train Random Forest for comparison
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    print(f"[Info] Random Forest Accuracy comparison: {rf_acc * 100:.2f}%")

    # Select best model (SVM) & Save Bundle
    model_bundle = {
        'model': svm_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': f_names,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        }
    }

    model_path = RESULTS_DIR / "model.joblib"
    joblib.dump(model_bundle, str(model_path))
    print(f"[OK] Saved trained model pack to: {model_path}")

    return model_bundle, acc

if __name__ == "__main__":
    train_and_evaluate()
