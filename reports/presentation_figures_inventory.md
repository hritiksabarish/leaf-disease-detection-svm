# Presentation Figures Inventory

| Filename | Status | Source | Data Used | Description |
|----------|--------|--------|-----------|-------------|
| `01_dataset_class_distribution.png` | REGENERATED / NEW | `data/research_manifest.csv` | local_class counts | Horizontal bar chart showing the final 1,952 image distribution across 10 classes. |
| `02_feature_count_vs_cv_macro_f1.png` | REGENERATED / NEW | `results/feature_selection.csv` | k, cv_macro_f1_mean, cv_macro_f1_std | Line plot of 5-fold CV Macro-F1 performance as feature count increases. |
| `04_feature_ranking.png` | REGENERATED / NEW | `results/feature_ranking.csv` | feature, score | Bar chart of the top 15 features ranked by ANOVA F-value. |
| `05_feature_selection_confusion_matrix.png` | COPIED | `results/figures/confusion_Selected20_rbf.png` | Selected20 RBF Image | Confusion matrix for the Selected20 configuration (copied from existing figures). |
| `06_kernel_accuracy_comparison.png` | REGENERATED / NEW | `results/kernel_comparison_test.csv` | kernel, test_accuracy | Bar chart of test accuracy across four SVM kernels. |
| `07_kernel_macro_f1_comparison.png` | REGENERATED / NEW | `results/kernel_comparison_test.csv` | kernel, test_macro_f1 | Bar chart of test Macro-F1 across four SVM kernels. |
| `08_kernel_cv_macro_f1.png` | REGENERATED / NEW | `results/kernel_comparison_cv.csv` | kernel, cv_macro_f1_mean, cv_macro_f1_std | Error bar plot of 5-fold CV performance for kernels. |
| `09_kernel_model_size.png` | REGENERATED / NEW | `results/efficiency_benchmark.csv` | kernel, model_size_kb | Bar chart showing physical size of serialized kernel models. |
| `10_kernel_support_vectors.png` | REGENERATED / NEW | `results/efficiency_benchmark.csv` | kernel, support_vectors | Bar chart of the number of support vectors selected per kernel. |
| `11_kernel_training_time.png` | REGENERATED / NEW | `results/efficiency_benchmark.csv` | kernel, training_time_mean_s | Bar chart of the training duration per kernel in milliseconds. |
| `12_robustness_brightness.png` | REGENERATED / NEW | `results/robustness_degradation.csv` | perturbation type brightness | Line chart of robustness degradation under brightness perturbations. |
| `13_robustness_contrast.png` | REGENERATED / NEW | `results/robustness_degradation.csv` | perturbation type contrast | Line chart of robustness degradation under contrast perturbations. |
| `14_robustness_noise.png` | REGENERATED / NEW | `results/robustness_degradation.csv` | perturbation type noise | Line chart of robustness degradation under noise perturbations. |
| `15_robustness_blur.png` | REGENERATED / NEW | `results/robustness_degradation.csv` | perturbation type blur | Line chart of robustness degradation under blur perturbations. |
| `16_efficiency_latency.png` | REGENERATED / NEW | `results/efficiency_benchmark.csv` | feature/svm time for RBF All24 | Bar chart comparing feature extraction vs SVM prediction latency (log scale). |
| `17_efficiency_throughput.png` | REGENERATED / NEW | `results/efficiency_benchmark.csv` | kernel, throughput_images_per_sec | Bar chart of processed images per second for each kernel. |
| `18_final_confusion_matrix.png` | COPIED | `results/figures/confusion_All24_rbf.png` | All24 RBF Image | Final validated confusion matrix for the RBF All24 model (copied from existing figures). |
