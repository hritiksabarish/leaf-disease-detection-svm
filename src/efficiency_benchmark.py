import os
import sys
import time
import pickle
import platform
import numpy as np
import pandas as pd
import cv2
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set deterministic state
np.random.seed(42)

from preprocess import load_and_preprocess_image
from segment import isolate_leaf_otsu, segment_diseased_region
from features import extract_all_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "research_features.csv"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

def gather_env_info():
    with open(REPORTS_DIR / "benchmark_environment.md", "w") as f:
        f.write("# Benchmark Environment\n\n")
        f.write(f"- **OS**: {platform.system()} {platform.release()} ({platform.version()})\n")
        f.write(f"- **CPU**: {platform.processor()}\n")
        f.write(f"- **Python Version**: {sys.version.split(' ')[0]}\n")
        f.write(f"- **NumPy Version**: {np.__version__}\n")
        f.write(f"- **OpenCV Version**: {cv2.__version__}\n")
        f.write(f"- **Scikit-Learn Version**: {sklearn.__version__}\n")
        f.write(f"- **Pandas Version**: {pd.__version__}\n")
        f.write("- **Hardware Used**: CPU (no GPU acceleration configured in OpenCV/Sklearn)\n")

def extract_features(img_path):
    # End-to-end feature extraction exactly as used in deployment
    prep = load_and_preprocess_image(str(img_path))
    leaf_mask = isolate_leaf_otsu(prep['blurred_rgb'])
    seg = segment_diseased_region(prep['blurred_rgb'], prep['hsv'], leaf_mask)
    feats_dict, _, f_names = extract_all_features(
        prep['blurred_rgb'], prep['hsv'], prep['gray'],
        seg['diseased_mask'], seg['leaf_mask']
    )
    for k, v in feats_dict.items():
        if np.isnan(v) or np.isinf(v):
            feats_dict[k] = 0.0
    return feats_dict

def main():
    gather_env_info()
    
    print("Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    
    feature_cols = [c for c in df.columns if c not in ['image_path', 'label', 'crop', 'condition', 'domain']]
    X = df[feature_cols].values
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.20, random_state=42, stratify=y
    )
    
    configs = ['All24', 'Selected20']
    kernels = ['Linear', 'RBF', 'Quadratic', 'Polynomial']
    
    results = []
    
    # 1. Feature Extraction Benchmark
    print("Benchmarking Feature Extraction...")
    # Use 50 random test images to get stable timing
    sample_imgs = df_test['image_path'].sample(50, random_state=42).tolist()
    extraction_times = []
    
    # Warmup
    _ = extract_features(PROJECT_ROOT / sample_imgs[0])
    
    for img in sample_imgs:
        t0 = time.perf_counter()
        _ = extract_features(PROJECT_ROOT / img)
        t1 = time.perf_counter()
        extraction_times.append((t1 - t0) * 1000) # in ms
        
    feat_mean_ms = np.mean(extraction_times)
    feat_std_ms = np.std(extraction_times)
    print(f"  Mean feature extraction time: {feat_mean_ms:.2f} ms")
    
    # 2. Model Benchmarks
    for config in configs:
        for kernel in kernels:
            model_path = MODELS_DIR / f"svm_{config}_{kernel}.pkl"
            if not model_path.exists():
                print(f"Skipping {config} {kernel} (Not found)")
                continue
                
            print(f"Benchmarking {config} - {kernel}...")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Model Size
            model_size_kb = os.path.getsize(model_path) / 1024
            
            # Support Vectors
            svm_clf = model.named_steps['classifier']
            n_support = np.sum(svm_clf.n_support_) if hasattr(svm_clf, 'n_support_') else 0
            num_features = 20 if config == 'Selected20' else 24
            
            # Training Time (Refit on full X_train)
            train_times = []
            for _ in range(3): # repeat 3 times
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                t1 = time.perf_counter()
                train_times.append(t1 - t0)
            train_mean_s = np.mean(train_times)
            train_std_s = np.std(train_times)
            
            # SVM Prediction Time
            # Warmup
            _ = model.predict(X_test[:5])
            
            pred_times = []
            for _ in range(5): # 5 passes over the whole test set
                t0 = time.perf_counter()
                _ = model.predict(X_test)
                t1 = time.perf_counter()
                # Time per image in ms
                pred_times.append(((t1 - t0) / len(X_test)) * 1000)
                
            pred_mean_ms = np.mean(pred_times)
            pred_std_ms = np.std(pred_times)
            
            end_to_end_mean_ms = feat_mean_ms + pred_mean_ms
            end_to_end_std_ms = np.sqrt(feat_std_ms**2 + pred_std_ms**2)
            throughput = 1000.0 / end_to_end_mean_ms
            
            results.append({
                'feature_configuration': config,
                'kernel': kernel,
                'features': num_features,
                'support_vectors': n_support,
                'model_size_kb': model_size_kb,
                'training_time_mean_s': train_mean_s,
                'training_time_std_s': train_std_s,
                'svm_prediction_time_mean_ms': pred_mean_ms,
                'svm_prediction_time_std_ms': pred_std_ms,
                'feature_extraction_time_mean_ms': feat_mean_ms,
                'feature_extraction_time_std_ms': feat_std_ms,
                'end_to_end_time_mean_ms': end_to_end_mean_ms,
                'end_to_end_time_std_ms': end_to_end_std_ms,
                'throughput_images_per_sec': throughput
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "efficiency_benchmark.csv", index=False)
    
    # -------------------------------------
    # PLOTS
    # -------------------------------------
    # Model Size
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='kernel', y='model_size_kb', hue='feature_configuration')
    plt.title('Model Size by Kernel and Feature Configuration')
    plt.ylabel('Model Size (KB)')
    plt.savefig(FIGURES_DIR / "model_size_comparison.png")
    plt.close()
    
    # Training Time
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='kernel', y='training_time_mean_s', hue='feature_configuration')
    plt.title('Training Time by Kernel and Feature Configuration')
    plt.ylabel('Training Time (s)')
    plt.savefig(FIGURES_DIR / "training_time_comparison.png")
    plt.close()
    
    # Inference Time (Prediction only)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results, x='kernel', y='svm_prediction_time_mean_ms', hue='feature_configuration')
    plt.title('SVM Prediction Time (Per Image) by Kernel')
    plt.ylabel('Prediction Time (ms)')
    plt.savefig(FIGURES_DIR / "inference_time_comparison.png")
    plt.close()
    
    # End-to-End Latency
    plt.figure(figsize=(10, 6))
    # For a stacked look, we just plot total, since extraction is constant across configs
    sns.barplot(data=df_results, x='kernel', y='end_to_end_time_mean_ms', hue='feature_configuration')
    plt.title('End-to-End Latency (Feature Extraction + Prediction)')
    plt.ylabel('Total Latency (ms)')
    plt.savefig(FIGURES_DIR / "end_to_end_latency.png")
    plt.close()
    
    print("\nEfficiency benchmark complete!")

if __name__ == "__main__":
    main()
