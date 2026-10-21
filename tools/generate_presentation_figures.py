import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
OUT_DIR = "results/presentation_figures"
os.makedirs(OUT_DIR, exist_ok=True)

inventory = []

def add_inv(filename, source, data_used, description):
    inventory.append({
        "Filename": filename,
        "Source CSV/Report": source,
        "Data Used": data_used,
        "Description": description
    })

# -------------------------------------------------------------------------
# A. DATASET / CLASS DISTRIBUTION
# -------------------------------------------------------------------------
manifest = pd.read_csv("data/research_manifest.csv")
class_counts = manifest['local_class'].value_counts().sort_values(ascending=True)

plt.figure(figsize=(12, 8))
bars = plt.barh(class_counts.index, class_counts.values, color=sns.color_palette("viridis", len(class_counts)))
plt.title("Dataset Class Distribution (1,952 images total)", pad=20, weight='bold')
plt.xlabel("Number of Images", weight='bold')
plt.ylabel("Crop & Disease Class", weight='bold')
for bar in bars:
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
             f"{int(bar.get_width())}", va='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_dataset_class_distribution.png", dpi=300)
plt.close()
add_inv("01_dataset_class_distribution.png", "data/research_manifest.csv", "local_class counts", "Horizontal bar chart showing the final 1,952 image distribution across 10 classes.")

# -------------------------------------------------------------------------
# B. FEATURE SELECTION
# -------------------------------------------------------------------------
fs_df = pd.read_csv("results/feature_selection.csv")
# columns: method, k, cv_accuracy_mean, cv_macro_f1_mean, cv_macro_f1_std

plt.figure(figsize=(10, 6))
plt.plot(fs_df['k'], fs_df['cv_macro_f1_mean'] * 100, marker='o', linewidth=3, markersize=10, color='#1f77b4')
plt.fill_between(fs_df['k'], 
                 (fs_df['cv_macro_f1_mean'] - fs_df['cv_macro_f1_std']) * 100, 
                 (fs_df['cv_macro_f1_mean'] + fs_df['cv_macro_f1_std']) * 100, 
                 alpha=0.2, color='#1f77b4')
plt.title("Selected Features vs CV Macro-F1", pad=15, weight='bold')
plt.xlabel("Number of Selected Features (k)", weight='bold')
plt.ylabel("Cross-Validation Macro-F1 (%)", weight='bold')
plt.xticks(fs_df['k'])
for idx, row in fs_df.iterrows():
    plt.annotate(f"{row['cv_macro_f1_mean']*100:.1f}%", 
                 (row['k'], row['cv_macro_f1_mean']*100), 
                 textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_feature_count_vs_cv_macro_f1.png", dpi=300)
plt.close()
add_inv("02_feature_count_vs_cv_macro_f1.png", "results/feature_selection.csv", "k, cv_macro_f1_mean, cv_macro_f1_std", "Line plot of 5-fold CV Macro-F1 performance as feature count increases.")

fr_df = pd.read_csv("results/feature_ranking.csv")
# columns: feature, feature_group, score
top_features = fr_df.head(15).sort_values(by='score', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(top_features['feature'], top_features['score'], color=sns.color_palette("rocket", len(top_features)))
plt.title("Top 15 Features by ANOVA F-Value (Feature Ranking)", pad=20, weight='bold')
plt.xlabel("ANOVA F-Score", weight='bold')
plt.ylabel("Engineered Feature", weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_ranking.png", dpi=300)
plt.close()
add_inv("04_feature_ranking.png", "results/feature_ranking.csv", "feature, score", "Bar chart of the top 15 features ranked by ANOVA F-value.")

if os.path.exists("results/figures/confusion_Selected20_rbf.png"):
    shutil.copy("results/figures/confusion_Selected20_rbf.png", f"{OUT_DIR}/05_feature_selection_confusion_matrix.png")
    add_inv("05_feature_selection_confusion_matrix.png", "results/figures/confusion_Selected20_rbf.png", "Selected20 RBF Image", "Confusion matrix for the Selected20 configuration (copied from existing figures).")

# -------------------------------------------------------------------------
# C. KERNEL COMPARISON
# -------------------------------------------------------------------------
kc_test = pd.read_csv("results/kernel_comparison_test.csv")
# columns: feature_configuration, kernel, test_accuracy, test_macro_f1
all24 = kc_test[kc_test['feature_configuration'] == 'All24'].sort_values(by='test_accuracy', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(all24['kernel'], all24['test_accuracy'] * 100, color=sns.color_palette("deep", len(all24)))
plt.title("Kernel vs Held-out Accuracy (All 24 Features)", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Accuracy (%)", weight='bold')
plt.ylim(0, 100)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height():.2f}%", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_kernel_accuracy_comparison.png", dpi=300)
plt.close()
add_inv("06_kernel_accuracy_comparison.png", "results/kernel_comparison_test.csv", "kernel, test_accuracy", "Bar chart of test accuracy across four SVM kernels.")

plt.figure(figsize=(10, 6))
bars = plt.bar(all24['kernel'], all24['test_macro_f1'] * 100, color=sns.color_palette("muted", len(all24)))
plt.title("Kernel vs Held-out Macro-F1 (All 24 Features)", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Macro-F1 (%)", weight='bold')
plt.ylim(0, 100)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height():.2f}%", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_kernel_macro_f1_comparison.png", dpi=300)
plt.close()
add_inv("07_kernel_macro_f1_comparison.png", "results/kernel_comparison_test.csv", "kernel, test_macro_f1", "Bar chart of test Macro-F1 across four SVM kernels.")

kc_cv = pd.read_csv("results/kernel_comparison_cv.csv")
# columns: feature_configuration, kernel, cv_macro_f1_mean, cv_macro_f1_std
all24_cv = kc_cv[kc_cv['feature_configuration'] == 'All24'].sort_values(by='cv_macro_f1_mean', ascending=False)

plt.figure(figsize=(10, 6))
plt.errorbar(all24_cv['kernel'], all24_cv['cv_macro_f1_mean'] * 100, 
             yerr=all24_cv['cv_macro_f1_std'] * 100, fmt='o', color='purple', 
             linewidth=3, capsize=8, markersize=10)
plt.title("Kernel Cross-Validation Macro-F1", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Mean CV Macro-F1 (%)", weight='bold')
plt.ylim(min((all24_cv['cv_macro_f1_mean'] - all24_cv['cv_macro_f1_std']) * 100) - 5, 100)
for idx, row in all24_cv.iterrows():
    plt.annotate(f"{row['cv_macro_f1_mean']*100:.2f}%", 
                 (row['kernel'], row['cv_macro_f1_mean']*100), 
                 textcoords="offset points", xytext=(15,0), ha='left', va='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_kernel_cv_macro_f1.png", dpi=300)
plt.close()
add_inv("08_kernel_cv_macro_f1.png", "results/kernel_comparison_cv.csv", "kernel, cv_macro_f1_mean, cv_macro_f1_std", "Error bar plot of 5-fold CV performance for kernels.")

eff = pd.read_csv("results/efficiency_benchmark.csv")
# columns: feature_configuration,kernel,features,support_vectors,model_size_kb,training_time_mean_s...
eff_all24 = eff[eff['feature_configuration'] == 'All24'].sort_values(by='model_size_kb', ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(eff_all24['kernel'], eff_all24['model_size_kb'], color='teal')
plt.title("Serialized Model Size by Kernel", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Model Size (KB)", weight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f"{bar.get_height():.1f} KB", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_kernel_model_size.png", dpi=300)
plt.close()
add_inv("09_kernel_model_size.png", "results/efficiency_benchmark.csv", "kernel, model_size_kb", "Bar chart showing physical size of serialized kernel models.")

plt.figure(figsize=(10, 6))
bars = plt.bar(eff_all24['kernel'], eff_all24['support_vectors'], color='coral')
plt.title("Support Vectors by Kernel", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Number of Support Vectors", weight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f"{int(bar.get_height())}", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_kernel_support_vectors.png", dpi=300)
plt.close()
add_inv("10_kernel_support_vectors.png", "results/efficiency_benchmark.csv", "kernel, support_vectors", "Bar chart of the number of support vectors selected per kernel.")

plt.figure(figsize=(10, 6))
bars = plt.bar(eff_all24['kernel'], eff_all24['training_time_mean_s'] * 1000, color='darkorange')
plt.title("Training Time by Kernel (1,561 samples)", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Training Time (ms)", weight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f"{bar.get_height():.1f} ms", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_kernel_training_time.png", dpi=300)
plt.close()
add_inv("11_kernel_training_time.png", "results/efficiency_benchmark.csv", "kernel, training_time_mean_s", "Bar chart of the training duration per kernel in milliseconds.")

# -------------------------------------------------------------------------
# D. ROBUSTNESS
# -------------------------------------------------------------------------
rob = pd.read_csv("results/robustness_degradation.csv")
# columns: perturbation, severity, baseline_macro_f1, perturbed_macro_f1...
rob['type'] = rob['perturbation'].apply(lambda x: x.split('_')[0])

def plot_robustness(ptype, filename, title):
    subset = rob[rob['type'] == ptype].copy()
    if subset.empty: return
    
    severity_order = {'Mild': 1, 'Medium': 2, 'Strong': 3}
    def get_sort_key(val):
        val_str = str(val).replace('%', '')
        try:
            return float(val_str)
        except ValueError:
            return severity_order.get(val_str, 99)
            
    subset['sort_key'] = subset['severity'].apply(get_sort_key)
    subset = subset.sort_values(by='sort_key')
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset['severity'].astype(str), subset['perturbed_macro_f1'] * 100, marker='D', color='firebrick', linewidth=3, markersize=10)
    plt.title(f"Controlled Perturbation Robustness — {title}", pad=15, weight='bold')
    plt.xlabel(f"Perturbation Severity", weight='bold')
    plt.ylabel("Macro-F1 (%)", weight='bold')
    plt.ylim(0, 100)
    
    # baseline 79.27
    plt.axhline(79.27, color='grey', linestyle='--', label='Baseline (79.27%)')
    plt.legend()
    
    for idx, row in subset.iterrows():
        plt.annotate(f"{row['perturbed_macro_f1']*100:.1f}%", 
                     (str(row['severity']), row['perturbed_macro_f1']*100), 
                     textcoords="offset points", xytext=(0,10), ha='center', weight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=300)
    plt.close()
    add_inv(filename, "results/robustness_degradation.csv", f"perturbation type {ptype}", f"Line chart of robustness degradation under {ptype} perturbations.")

plot_robustness('brightness', '12_robustness_brightness.png', 'Brightness')
plot_robustness('contrast', '13_robustness_contrast.png', 'Contrast')
plot_robustness('noise', '14_robustness_noise.png', 'Gaussian Noise')
plot_robustness('blur', '15_robustness_blur.png', 'Gaussian Blur')

# -------------------------------------------------------------------------
# E. EFFICIENCY
# -------------------------------------------------------------------------
# eff_all24 already loaded
rbf_eff = eff_all24[eff_all24['kernel'] == 'RBF'].iloc[0]
latencies = {
    'Feature Extraction': rbf_eff['feature_extraction_time_mean_ms'],
    'SVM Prediction': rbf_eff['svm_prediction_time_mean_ms']
}

plt.figure(figsize=(10, 6))
bars = plt.bar(latencies.keys(), latencies.values(), color=['#9467bd', '#8c564b'])
plt.title("Inference Latency Breakdown (RBF All24)", pad=15, weight='bold')
plt.ylabel("Latency (ms / image)", weight='bold')
plt.yscale('log')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
             f"{bar.get_height():.2f} ms", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/16_efficiency_latency.png", dpi=300)
plt.close()
add_inv("16_efficiency_latency.png", "results/efficiency_benchmark.csv", "feature/svm time for RBF All24", "Bar chart comparing feature extraction vs SVM prediction latency (log scale).")

plt.figure(figsize=(10, 6))
bars = plt.bar(eff_all24['kernel'], eff_all24['throughput_images_per_sec'], color='cadetblue')
plt.title("Inference Throughput by Kernel (Images / Sec)", pad=15, weight='bold')
plt.xlabel("SVM Kernel", weight='bold')
plt.ylabel("Throughput (FPS)", weight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f"{bar.get_height():.2f}", ha='center', weight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/17_efficiency_throughput.png", dpi=300)
plt.close()
add_inv("17_efficiency_throughput.png", "results/efficiency_benchmark.csv", "kernel, throughput_images_per_sec", "Bar chart of processed images per second for each kernel.")

# -------------------------------------------------------------------------
# F. FINAL MODEL PERFORMANCE
# -------------------------------------------------------------------------
if os.path.exists("results/figures/confusion_All24_rbf.png"):
    shutil.copy("results/figures/confusion_All24_rbf.png", f"{OUT_DIR}/18_final_confusion_matrix.png")
    add_inv("18_final_confusion_matrix.png", "results/figures/confusion_All24_rbf.png", "All24 RBF Image", "Final validated confusion matrix for the RBF All24 model (copied from existing figures).")


# Write inventory
with open("reports/presentation_figures_inventory.md", "w") as f:
    f.write("# Presentation Figures Inventory\n\n")
    f.write("| Filename | Status | Source | Data Used | Description |\n")
    f.write("|----------|--------|--------|-----------|-------------|\n")
    for item in inventory:
        status = "COPIED" if "figures/" in item["Source CSV/Report"] else "REGENERATED / NEW"
        f.write(f"| `{item['Filename']}` | {status} | `{item['Source CSV/Report']}` | {item['Data Used']} | {item['Description']} |\n")

print("All figures generated successfully.")
