# Corrected Baseline Experiment

## Dataset

- **Source**: Authentic `emmarex/plantdisease` dataset downloaded from Kaggle.
- **Classes**: 
  - `Potato___Early_blight`
  - `Potato___healthy`
  - `Tomato___Late_blight`
  - `Tomato___healthy`
- **Images per class**: 
  - Potato Early Blight: 205
  - Potato Healthy: 157
  - Tomato Late Blight: 205
  - Tomato Healthy: 205
- **Total physical images**: 772

## Dataset validation

- **Valid**: 772
- **Invalid**: 0
- **Duplicates**: 0
- **Synthetic images**: 0 (all previous synthetic placeholders have been securely removed).

## Data loading

- **Physical files discovered**: 772
- **Samples extracted**: 772
- **Duplicate paths**: 0 (The Windows `.glob()` case-insensitivity bug has been fixed).

## Original model

- **Preprocessing**: Resize to 256x256, Gaussian Blur, RGB & HSV conversion.
- **Segmentation**: Otsu thresholding for background removal, HSV K-means ($k=3$) for necrotic spot isolation.
- **Features**: Color (HSV Mean/Std), GLCM Texture (Contrast, Dissimilarity, etc.), and Contour Shape (Area, Compactness, etc.).
- **Number of features**: 18
- **Classifier**: Support Vector Machine (SVM)
- **Kernel**: RBF
- **C**: 10.0
- **Train/test split**: 80/20 Stratified Split (`random_state=42` used for determinism).

## Results

- **Accuracy**: 93.55%
- **Precision**: 93.76%
- **Recall**: 93.55%
- **F1**: 93.53%

## Important note

The previously reported ~98% baseline was **INVALID** because:
1. Tomato classes contained synthetic placeholder data due to a Kaggle directory string mismatch.
2. Windows globbing caused identical physical `.jpg` images to be loaded twice.
3. The duplicated samples could cross the train/test split, causing catastrophic data leakage and artificially high scores.

Therefore, the previous result must not be used in the research report or PPT.

## Current status

The corrected baseline is now **trustworthy**. The dataset is authentic, balanced, properly evaluated without data leakage, and fully reproducible.
