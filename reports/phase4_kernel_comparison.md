# Phase 4 — SVM Kernel Comparison

## Objective
To rigorously compare four SVM kernels (Linear, RBF, Polynomial, Quadratic) and their optimal hyperparameters to determine how different decision boundaries behave on the multi-crop handcrafted feature representation.

## Dataset
- **Total Images**: 1,952
- **Classes**: 10
- **Split**: 1,561 (80%) Training / 391 (20%) Held-Out Test

## Feature configurations
Two configurations were evaluated to determine kernel performance both with and without feature selection.

### All 24 features
The complete, unreduced representation consisting of Color, Texture, Leaf Shape, and Lesion Shape features.

### Selected 20 features
Dimensionality reduction using `SelectKBest(f_classif, k=20)`. The feature selection was fitted explicitly *inside* the cross-validation folds to prevent data leakage.

## Experimental protocol
- **Strategy**: Grid Search across the hyperparameter space.
- **Scoring Metric**: Macro-F1 (primary) due to mild class imbalance.
- **Train/test split**: Stratified 80/20.

## Cross-validation
5-fold Stratified CV on the 80% training set (`random_state=42`).

## Hyperparameter grids
- **C**: [0.1, 1, 10, 100] (All kernels)
- **gamma**: ['scale', 0.001, 0.01, 0.1] (RBF, Polynomial, Quadratic)

## Kernel definitions

### Linear
`SVC(kernel="linear")`

### RBF
`SVC(kernel="rbf")`

### Polynomial
`SVC(kernel="poly", degree=3)`

### Quadratic
`SVC(kernel="poly", degree=2)`

## CV results
*Highest Mean CV Macro-F1 per kernel:*

**All 24 Features:**
- **RBF**: 77.95%
- **Polynomial**: 73.17%
- **Linear**: 72.47%
- **Quadratic**: 70.22%

**Selected 20 Features:**
- **RBF**: 79.48%
- **Linear**: 73.32%
- **Polynomial**: 72.90%
- **Quadratic**: 69.80%

*Note: On the training set, RBF paired with feature selection achieved the highest CV score (79.48%).*

## Selected hyperparameters
The best hyperparameters found during training CV:

**All 24 Features:**
- **RBF**: `C=10.0`, `gamma='scale'`
- **Linear**: `C=1.0`
- **Polynomial**: `C=10.0`, `gamma='scale'`
- **Quadratic**: `C=10.0`, `gamma='scale'`

**Selected 20 Features:**
- **RBF**: `C=100.0`, `gamma=0.01`
- **Linear**: `C=1.0`
- **Polynomial**: `C=10.0`, `gamma='scale'`
- **Quadratic**: `C=10.0`, `gamma='scale'`

## Final held-out test results
*Evaluated exactly once on the untouched 20% test set using the best CV hyperparameters.*

**All 24 Features:**
- **RBF**: Accuracy = **79.28%** | Macro-F1 = **79.27%**
- **Linear**: Accuracy = 75.19% | Macro-F1 = 74.81%
- **Polynomial**: Accuracy = 72.89% | Macro-F1 = 73.05%
- **Quadratic**: Accuracy = 72.37% | Macro-F1 = 72.69%

**Selected 20 Features:**
- **RBF**: Accuracy = **78.00%** | Macro-F1 = **77.98%**
- **Linear**: Accuracy = 75.95% | Macro-F1 = 75.73%
- **Polynomial**: Accuracy = 73.14% | Macro-F1 = 73.23%
- **Quadratic**: Accuracy = 69.56% | Macro-F1 = 69.99%

**Conclusion**: Under the specified 5-fold CV protocol, the **RBF** kernel significantly outperformed all other kernels across both configurations. It achieved the highest held-out test Macro-F1 (79.27%) using all 24 features. 

## Confusion matrices
Saved to `results/figures/confusion_<config>_<kernel>.png`. They reveal that while RBF performs best, there remains some confusion between visually similar classes (e.g. Early Blight vs Late Blight).

## Training time
All models trained extremely quickly:
- **RBF (All24)**: ~0.20 seconds
- **Linear (All24)**: ~0.18 seconds
- **Polynomial (All24)**: ~0.14 seconds
- **Quadratic (All24)**: ~0.10 seconds

## Support vectors
- **RBF (All24)**: 1,114
- **Linear (All24)**: 950
- **Polynomial (All24)**: 1,083
- **Quadratic (All24)**: 1,108
*(Note: With 1,561 training samples, requiring ~1,000 support vectors indicates a highly complex, non-linear decision boundary).*

## Model size
- **RBF (All24)**: 294.7 KB
- **Linear (All24)**: 251.8 KB
- **Polynomial (All24)**: 286.6 KB
- **Quadratic (All24)**: 293.1 KB

## Feature-selection interaction
Feature selection ($k=20$) behaved inconsistently across kernels. While it improved Linear SVM test performance (74.8% $\rightarrow$ 75.7%), it degraded RBF test performance (79.2% $\rightarrow$ 77.9%) and Quadratic test performance (72.6% $\rightarrow$ 69.9%). The RBF kernel with all 24 features remains the most robust representation.

## Methodological Note
I explicitly confirm that no test-set metric was used to choose `C`, `gamma`, `degree`, kernel, or feature count. The test set was touched exactly once per model after all hyperparameters were finalized via training CV.

## Limitations
- Final test set was untouched during tuning.
- Results apply to the current lab dataset.
- Field robustness has not yet been tested.
- Inference-time benchmarking is not yet complete.
