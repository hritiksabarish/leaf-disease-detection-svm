import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure deterministic behavior
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "research_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    
    # Extract features and labels
    feature_cols = [c for c in df.columns if c not in ['image_path', 'label', 'crop', 'condition', 'domain']]
    X = df[feature_cols].values
    y_raw = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    
    # 1. Train/Test Split (80/20) - untouched final test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Dataset Size: {len(df)}")
    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    print(f"Feature Count: {X.shape[1]}")
    
    # 2. VarianceThreshold Sanity Filter (on train only)
    print("\nRunning VarianceThreshold...")
    var_filter = VarianceThreshold()
    var_filter.fit(X_train)
    retained_features = var_filter.get_support(indices=True)
    print(f"Features removed by VarianceThreshold: {X.shape[1] - len(retained_features)}")
    
    # 3. SelectKBest CV Experiment
    print("\nRunning SelectKBest CV Experiment...")
    k_values = [5, 10, 15, 20, len(feature_cols)]
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    
    for k in k_values:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("classifier", SVC(kernel='rbf', C=10.0, random_state=RANDOM_STATE))
        ])
        
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv_strategy,
            scoring={'acc': 'accuracy', 'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted'},
            n_jobs=-1, return_train_score=False
        )
        
        results.append({
            'method': 'SelectKBest',
            'k': k,
            'cv_accuracy_mean': np.mean(scores['test_acc']),
            'cv_accuracy_std': np.std(scores['test_acc']),
            'cv_macro_f1_mean': np.mean(scores['test_macro_f1']),
            'cv_macro_f1_std': np.std(scores['test_macro_f1']),
            'cv_weighted_f1_mean': np.mean(scores['test_weighted_f1']),
            'cv_weighted_f1_std': np.std(scores['test_weighted_f1']),
            'feature_count': k
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / "feature_selection.csv", index=False)
    print(df_results[['k', 'cv_accuracy_mean', 'cv_macro_f1_mean']])
    
    # 4. Create Visualizations
    plt.figure(figsize=(8, 5))
    plt.errorbar(df_results['k'], df_results['cv_macro_f1_mean'], yerr=df_results['cv_macro_f1_std'], fmt='-o', capsize=5)
    plt.title('Feature Count vs CV Macro-F1')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Mean CV Macro-F1')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "feature_count_vs_macro_f1.png")
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(df_results['k'], df_results['cv_accuracy_mean'], yerr=df_results['cv_accuracy_std'], fmt='-o', capsize=5, color='orange')
    plt.title('Feature Count vs CV Accuracy')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "feature_count_vs_accuracy.png")
    plt.close()
    
    # 5. Feature Ranking (On Training Data Only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train_scaled, y_train)
    
    scores = selector.scores_
    
    def get_group(fname):
        if fname.startswith('color_'): return 'COLOR'
        if fname.startswith('glcm_'): return 'TEXTURE'
        if fname.startswith('leaf_'): return 'LEAF_SHAPE'
        if fname.startswith('lesion_'): return 'LESION_SHAPE'
        return 'UNKNOWN'
        
    ranking_data = []
    for i, name in enumerate(feature_cols):
        ranking_data.append({
            'feature': name,
            'feature_group': get_group(name),
            'score': scores[i]
        })
    df_ranking = pd.DataFrame(ranking_data).sort_values(by='score', ascending=False)
    df_ranking.to_csv(RESULTS_DIR / "feature_ranking.csv", index=False)
    
    # 6. Select Best K Configuration
    # We choose the K that achieves highest CV Macro-F1, but if multiple K are within 0.005, we prefer the simpler model
    max_f1 = df_results['cv_macro_f1_mean'].max()
    threshold = max_f1 - 0.005 # Tolerance of 0.5%
    candidate_ks = df_results[df_results['cv_macro_f1_mean'] >= threshold]
    best_k = candidate_ks['k'].min() # Smallest k within threshold
    print(f"\nSelected best K = {best_k} (Max F1: {max_f1:.4f}, Tolerance F1: {threshold:.4f})")
    
    # 7. Final Held-Out Test Evaluation
    # Pipeline for Selected K
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=best_k)),
        ("classifier", SVC(kernel='rbf', C=10.0, random_state=RANDOM_STATE))
    ])
    final_pipeline.fit(X_train, y_train)
    
    # Extract selected feature names
    support = final_pipeline.named_steps["selector"].get_support()
    selected_feature_names = np.array(feature_cols)[support]
    
    y_pred = final_pipeline.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred)
    test_macro_prec = precision_score(y_test, y_pred, average='macro')
    test_macro_rec = recall_score(y_test, y_pred, average='macro')
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- FINAL HELD-OUT TEST (SELECTED FEATURES) ---")
    print(f"Accuracy:    {test_acc:.4f}")
    print(f"Macro F1:    {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Selected K={best_k} Features)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_selection_confusion_matrix.png")
    plt.close()
    
    # 8. Compare against All-Feature Baseline
    baseline_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel='rbf', C=10.0, random_state=RANDOM_STATE))
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)
    
    base_acc = accuracy_score(y_test, y_pred_base)
    base_macro_f1 = f1_score(y_test, y_pred_base, average='macro')
    base_weighted_f1 = f1_score(y_test, y_pred_base, average='weighted')
    
    print("\n--- BASELINE (ALL 24 FEATURES) ---")
    print(f"Accuracy:    {base_acc:.4f}")
    print(f"Macro F1:    {base_macro_f1:.4f}")
    print(f"Weighted F1: {base_weighted_f1:.4f}")
    
    # Dump report info
    report_data = {
        'best_k': int(best_k),
        'selected_features': selected_feature_names.tolist(),
        'test_acc': test_acc,
        'test_macro_f1': test_macro_f1,
        'test_weighted_f1': test_weighted_f1,
        'base_acc': base_acc,
        'base_macro_f1': base_macro_f1,
        'base_weighted_f1': base_weighted_f1
    }
    with open(RESULTS_DIR / "phase3_results.json", "w") as f:
        json.dump(report_data, f)

if __name__ == "__main__":
    main()
