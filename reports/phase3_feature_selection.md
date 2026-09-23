# Phase 3 — Feature Selection

## Objective
To rigorously evaluate which of the 24 handcrafted features are most discriminative for multi-crop leaf disease classification, and determine if dimensionality reduction via feature selection can improve or maintain model performance.

## Dataset
- **Location**: `data/research_features.csv`
- **Total Images**: 1,952
- **Classes**: 10 (Pepper, Potato, Tomato)
- **Feature Count**: 24

## Experimental protocol
- **Base Classifier**: SVM (kernel=RBF, C=10.0)
- **Selection Methods**: `VarianceThreshold` (sanity check), `SelectKBest` with ANOVA F-test (primary experiment).
- **Evaluation**: 5-Fold Stratified Cross-Validation on the training set, tracking Macro-F1 and Accuracy.

## Leakage prevention
Feature selection was implemented strictly *inside* the Scikit-Learn `Pipeline`. The scaler (`StandardScaler`) and selector (`SelectKBest`) were fitted exclusively on each CV training fold, preventing any data leakage from the validation or test splits. The final 20% test set was kept entirely untouched until the final, post-CV evaluation.

## Train/test split
- **Strategy**: Stratified 80/20 split (`random_state=42`).
- **Train Size**: 1,561
- **Test Size**: 391

## Cross-validation
- **Strategy**: `StratifiedKFold` (n_splits=5, shuffle=True, random_state=42).

## Methods

### VarianceThreshold
Applied to the training set as a sanity filter.
- **Features removed**: 0. 
- *Conclusion*: As established in Phase 2, all 24 features possess variance; none are strictly constant.

### SelectKBest
Evaluated for $k \in \{5, 10, 15, 20, 24\}$. 

## Results
*Scores are CV Train-Fold Averages:*
| $k$ Features | CV Accuracy | CV Macro-F1 |
|:---:|:---:|:---:|
| 5 | 50.73% | 50.42% |
| 10 | 66.75% | 66.57% |
| 15 | 77.77% | 77.86% |
| **20** | **78.48%** | **78.61%** |
| 24 | 77.84% | 77.95% |

## Selected feature configuration
Based on the CV results, $k=20$ achieved the highest CV Macro-F1 score (78.61%). It slightly outperforms the full 24-feature set during CV while simplifying the feature space. Therefore, $k=20$ was selected as the optimal configuration.

## Feature ranking
*ANOVA F-scores calculated exclusively on the training set (for interpretation):*
1. `leaf_compactness` (103.60)
2. `color_mean_v` (87.11)
3. `lesion_area` (80.78)
4. `lesion_solidity` (74.61)
5. `lesion_extent` (71.13)
... *(Full ranking in `results/feature_ranking.csv`)*

## Feature-group analysis
The top 20 selected features consist of:
- **COLOR**: 5 features (Excluded `color_std_s`)
- **TEXTURE**: 5 features (Excluded `glcm_asm`)
- **LEAF_SHAPE**: 6 features (None excluded)
- **LESION_SHAPE**: 4 features (Excluded `lesion_eccentricity`, `lesion_perimeter`)

All feature groups strongly contribute to the top 20, validating the multi-faceted (Color + Texture + Shape) representation strategy.

## Final held-out test result
*Using the chosen $k=20$ pipeline on the untouched 20% test set:*
- **Accuracy**: 78.52%
- **Macro F1**: 78.56%
- **Weighted F1**: 78.43%

## Comparison with all 24 features
*Using the baseline $k=24$ pipeline on the same 20% test set:*
- **Accuracy**: 79.28%
- **Macro F1**: 79.27%
- **Weighted F1**: 79.11%

*Conclusion*: While CV suggested $k=20$ was marginally better, the final test evaluation shows the full 24-feature baseline actually retains a slight ~0.7% advantage. Both configurations perform very similarly (~79%), indicating that the SVM is robust to the additional 4 features and dimensionality reduction to $k=20$ does not fundamentally alter classification behavior.

## Limitations
- Feature selection was performed without using the final test set; final test set was evaluated only after selection decisions.
- Results are based on the current lab dataset.
- Field robustness has not yet been evaluated.
- The 79% accuracy ceiling indicates that the RBF SVM (with default $C=10, \gamma=scale$) may require hyperparameter tuning or kernel optimization (reserved for Phase 4) to handle the complexity of 10 crop-disease classes.
