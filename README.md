# Leaf Disease Detection Using Image Segmentation & Feature Extraction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue.svg)](https://opencv.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io)

A complete college mini-project implementation for automated **Leaf Disease Detection** utilizing classical computer vision techniques (Otsu thresholding, HSV K-Means clustering), texture analysis (GLCM), color and contour shape feature extraction, and Support Vector Machine (SVM) classification.

---

## 📁 Project Structure

```
leaf-disease-detection/
├── data/
│   └── subset/             # PlantVillage subset (~150-200 images per class)
│       ├── Tomato___Late_blight/
│       ├── Tomato___healthy/
│       ├── Potato___Early_blight/
│       └── Potato___healthy/
├── src/
│   ├── preprocess.py       # Image resizing (256x256), Gaussian Blur, RGB/HSV conversion
│   ├── segment.py          # Otsu background removal & HSV K-Means (k=3) spot extraction
│   ├── features.py         # Color (HSV mean/std), GLCM texture, shape contour features
│   ├── classify.py         # Dataset builder, 80/20 split, SVM training & metrics
│   └── pipeline.py         # End-to-end processing & 3-panel visualization generator
├── scripts/
│   └── download_data.py    # Kaggle API downloader & dataset subset generator
├── notebooks/
│   └── demo.ipynb          # Step-by-step Jupyter Notebook walkthrough
├── results/                # Output visualizations, confusion matrix & saved model pack
├── app.py                  # Custom-styled Streamlit web application
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores credentials, cached datasets, and bytecode
└── README.md               # Setup and usage guide
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Dependencies
Clone or open this repository, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup via Kaggle API
Run the automated dataset download script:
```bash
python scripts/download_data.py
```
> **Note**: The script checks for `KAGGLE_API_TOKEN` environment variable or `kaggle.json` credentials. It downloads the dataset (`emmarex/plantdisease`), extracts the 4 target classes into `data/subset/` capped at 200 images per class, and ensures fallback samples are ready.

---

## 🚀 Running the Project

### 1. Train the Classifier & Evaluate Metrics
Train the SVM model on extracted color, GLCM texture, and contour shape features:
```bash
python src/classify.py
```
**Output:**
- Prints Accuracy, Precision, Recall, and F1 Score.
- Saves `confusion_matrix.png` to `results/`.
- Saves serialized model pack to `results/model.joblib`.

### 2. Run Pipeline on Sample Images
Process sample images end-to-end and generate side-by-side visualizations (`Original` | `Segmented Mask` | `Highlighted Overlay`):
```bash
python src/pipeline.py
```
Outputs `pipeline_sample_1.png` ... `pipeline_sample_5.png` in `results/`.

### 3. Launch the Streamlit Web Application
Start the interactive UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. Features include:
- Upload any leaf photo or choose pre-loaded test samples.
- Side-by-side 3-panel pipeline visualization.
- Diagnostic card with disease prediction & confidence percentage.
- Feature breakdown table (Color, GLCM Texture, Shape metrics).

---

## 🔬 Image Processing & ML Pipeline Details

1. **Preprocessing (`preprocess.py`)**:
   - Resizes images to $256 \times 256$.
   - Applies $5 \times 5$ Gaussian Blur for noise reduction.
   - Converts RGB $\rightarrow$ Grayscale and HSV color spaces.

2. **Segmentation (`segment.py`)**:
   - **Background Isolation**: Otsu's thresholding combined with Excess Green (ExG) index ($2G - R - B$).
   - **Diseased Spot Extraction**: K-means clustering ($k=3$) in HSV space + necrotic color thresholding to segment lesion spots from healthy green foliage.

3. **Feature Extraction (`features.py`)**:
   - **Color**: Mean and standard deviation of H, S, V channels within the diseased ROI.
   - **Texture**: Gray Level Co-occurrence Matrix (GLCM) metrics (`contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`).
   - **Shape**: Lesion Contour `area`, `perimeter`, `eccentricity`, `extent`, `solidity`, and `compactness` metric ratio ($4\pi A / P^2$).

4. **Classification (`classify.py`)**:
   - Standardized scaling via `StandardScaler`.
   - Support Vector Classifier (SVM) with RBF Kernel ($C=10.0$).
   - 80/20 train/test stratified split.

---

## 📜 Group Metadata

- **Project**: Leaf Disease Detection Using Image Segmentation & Feature Extraction
- **Course/Group**: IPA Project Group 12
