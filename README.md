# Lightweight Multi-Crop Leaf Disease Detection using SVM

## Overview
This project presents a lightweight, handcrafted-feature machine learning framework for plant leaf disease detection. It processes raw leaf images through a series of deterministic steps to extract meaningful agricultural metrics, which are then classified by a Support Vector Machine (SVM).

The complete pipeline includes:
- Image preprocessing (blurring, colorspace conversion)
- Leaf and lesion segmentation (Otsu thresholding, K-Means clustering)
- Color features extraction
- GLCM texture features extraction
- Leaf shape features extraction
- Lesion shape features extraction
- Feature selection analysis (ANOVA F-value)
- SVM kernel comparison (Linear, RBF, Polynomial, Quadratic)
- Controlled perturbation robustness evaluation
- Efficiency benchmarking
- Treatment recommendation knowledge base
- Streamlit application

## Dataset
The dataset utilized in the research contains:
- **1,952 authentic images**
- **3 crops** (Pepper, Potato, Tomato)
- **10 classes** (Healthy and Diseased variants)

*Note: The raw image dataset is kept outside this repository due to its large size and dataset-management considerations. The repository contains the source code, tests, pipeline algorithms, and the final validated model artifact required for inference.*

## Model
The **Primary evaluated configuration** identified during Phase 4 is:
**RBF SVM + All24 features**

On the rigorously held-out test set, this model achieved:
- **Accuracy**: 79.28%
- **Macro-F1**: 79.27%

*Note: These are held-out research evaluation metrics over the entire dataset test split, and are NOT the absolute accuracy guarantee of every individual uploaded image.*

## Efficiency
The framework was formally benchmarked for computational efficiency on a standard CPU.
- **End-to-end latency**: ~58.66 ms/image
- **Throughput**: ~17.05 images/sec
- **SVM prediction time**: ~0.12 ms/image

*Note: These measurements were obtained in the documented desktop CPU benchmark environment. Feature extraction accounts for >99.8% of the processing time. We do not claim experimentally validated mobile deployment, though the microscopic model footprint (<300 KB) suggests strong suitability for edge environments.*

## Robustness
Robustness was comprehensively evaluated using controlled/synthetic perturbations across 15 degradation states (blur, noise, contrast, brightness).
*Explicit limitation: Natural field-image validation was not completed due to upstream dataset authentication limits.*

## Application
To launch the integrated Streamlit application (which includes the frozen diagnostic model and the deterministic agricultural treatment knowledge base), run:

```bash
python -m streamlit run src/app.py
```

## Limitations
- **Scope**: Diagnosis is strictly limited to the trained 3 crops and 10 classes.
- **Robustness**: Evaluated via controlled perturbations only; natural field evaluation is unavailable.
- **Confidence**: Probability confidence is unavailable because the SVC uses `probability=False` to optimize training efficiency.
- **Hardware**: Actual mobile/edge deployment was not experimentally tested.
- **Recommendations**: The integrated treatment knowledge base is strictly informational and relies entirely on the classifier's algorithmic prediction.

## Disclaimer
⚠️ The application is a research/educational prototype and should not be treated as definitive agricultural diagnosis. Verify important decisions with qualified agricultural professionals and applicable local guidance.
