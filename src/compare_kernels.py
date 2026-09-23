import os
import json
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure deterministic behavior
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "research_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(FEATURES_CSV)
    
    feature_cols = [c for c in df.columns if c not in ['image_path', 'label', 'crop', 'condition', 'domain']]
    X = df[feature_cols].values
    y_raw = df['label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Dataset Size: {len(df)}")
    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Define hyperparameter grids
    grids = {
        'Linear': {
            'classifier__kernel': ['linear'],
            'classifier__C': [0.1, 1, 10, 100]
        },
        'RBF': {
            'classifier__kernel': ['rbf'],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 0.001, 0.01, 0.1]
        },
        'Quadratic': {
            'classifier__kernel': ['poly'],
            'classifier__degree': [2],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 0.001, 0.01, 0.1]
        },
        'Polynomial': {
            'classifier__kernel': ['poly'],
            'classifier__degree': [3],
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 0.001, 0.01, 0.1]
        }
    }
    
    scoring = {
        'macro_f1': 'f1_macro',
        'acc': 'accuracy',
        'weighted_f1': 'f1_weighted',
        'macro_prec': 'precision_macro',
        'macro_rec': 'recall_macro'
    }
    
    configs = {
        'All24': None,
        'Selected20': 20
    }
    
    all_cv_results = []
    final_test_results = []
    
    best_estimators = {}
    
    for config_name, k in configs.items():
        print(f"\n======================================")
        print(f"Evaluating Configuration: {config_name}")
        print(f"======================================")
        
        for kernel_name, param_grid in grids.items():
            print(f"--- Kernel: {kernel_name} ---")
            
            if k is None:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(random_state=RANDOM_STATE))
                ])
            else:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=f_classif, k=k)),
                    ("classifier", SVC(random_state=RANDOM_STATE))
                ])
                
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv_strategy, scoring=scoring,
                refit='macro_f1', n_jobs=-1, return_train_score=False
            )
            
            grid_search.fit(X_train, y_train)
            
            # Extract CV results
            cv_res = grid_search.cv_results_
            best_idx = grid_search.best_index_
            
            all_cv_results.append({
                'feature_configuration': config_name,
                'kernel': kernel_name,
                'degree': cv_res['param_classifier__degree'][best_idx] if 'param_classifier__degree' in cv_res else np.nan,
                'C': cv_res['param_classifier__C'][best_idx],
                'gamma': cv_res['param_classifier__gamma'][best_idx] if 'param_classifier__gamma' in cv_res else np.nan,
                'cv_accuracy_mean': cv_res['mean_test_acc'][best_idx],
                'cv_accuracy_std': cv_res['std_test_acc'][best_idx],
                'cv_macro_f1_mean': cv_res['mean_test_macro_f1'][best_idx],
                'cv_macro_f1_std': cv_res['std_test_macro_f1'][best_idx],
                'cv_macro_precision_mean': cv_res['mean_test_macro_prec'][best_idx],
                'cv_macro_recall_mean': cv_res['mean_test_macro_rec'][best_idx],
                'cv_weighted_f1_mean': cv_res['mean_test_weighted_f1'][best_idx],
                'fit_time_mean': cv_res['mean_fit_time'][best_idx],
                'fit_time_std': cv_res['std_fit_time'][best_idx]
            })
            
            best_model = grid_search.best_estimator_
            best_estimators[f"{config_name}_{kernel_name}"] = best_model
            
            # --- FINAL TEST EVALUATION ---
            t0 = time.time()
            best_model.fit(X_train, y_train) # refit just in case
            fit_time = time.time() - t0
            
            # Save model to get size
            model_path = MODELS_DIR / f"svm_{config_name}_{kernel_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            model_size_kb = os.path.getsize(model_path) / 1024
            
            svm_clf = best_model.named_steps['classifier']
            n_support = np.sum(svm_clf.n_support_) if hasattr(svm_clf, 'n_support_') else 0
            
            y_pred = best_model.predict(X_test)
            
            test_acc = accuracy_score(y_test, y_pred)
            test_macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            test_macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            test_macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            test_weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            final_test_results.append({
                'feature_configuration': config_name,
                'kernel': kernel_name,
                'test_accuracy': test_acc,
                'test_macro_precision': test_macro_prec,
                'test_macro_recall': test_macro_rec,
                'test_macro_f1': test_macro_f1,
                'test_weighted_f1': test_weighted_f1,
                'training_time_sec': fit_time,
                'support_vectors': n_support,
                'model_size_kb': model_size_kb
            })
            
            # Generate Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix ({config_name}, Kernel: {kernel_name})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"confusion_{config_name}_{kernel_name.lower()}.png")
            plt.close()
            
    # Save CV Results
    df_cv = pd.DataFrame(all_cv_results)
    df_cv.to_csv(RESULTS_DIR / "kernel_comparison_cv.csv", index=False)
    
    # Save Final Test Results
    df_test = pd.DataFrame(final_test_results)
    df_test.to_csv(RESULTS_DIR / "kernel_comparison_test.csv", index=False)
    
    # -------------------------------------
    # PLOTS
    # -------------------------------------
    
    # CV Macro-F1 Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_cv, x='kernel', y='cv_macro_f1_mean', hue='feature_configuration', errorbar=None)
    
    # Add error bars manually
    x_pos = np.arange(len(grids))
    width = 0.4
    for i, config in enumerate(['All24', 'Selected20']):
        sub = df_cv[df_cv['feature_configuration'] == config]
        plt.errorbar(x_pos + (i * width) - width/2, sub['cv_macro_f1_mean'], yerr=sub['cv_macro_f1_std'], 
                     fmt='none', ecolor='black', capsize=5)
        
    plt.title('Mean CV Macro-F1 by Kernel and Feature Configuration')
    plt.ylabel('CV Macro-F1')
    plt.xlabel('Kernel')
    plt.legend(title='Configuration')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "kernel_cv_macro_f1.png")
    plt.close()
    
    # Test Accuracy Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_test, x='kernel', y='test_accuracy', hue='feature_configuration')
    plt.title('Held-Out Test Accuracy by Kernel')
    plt.ylabel('Accuracy')
    plt.xlabel('Kernel')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "kernel_test_accuracy.png")
    plt.close()
    
    # Test Macro-F1 Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_test, x='kernel', y='test_macro_f1', hue='feature_configuration')
    plt.title('Held-Out Test Macro-F1 by Kernel')
    plt.ylabel('Macro-F1')
    plt.xlabel('Kernel')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(FIGURES_DIR / "kernel_test_macro_f1.png")
    plt.close()

if __name__ == "__main__":
    main()
