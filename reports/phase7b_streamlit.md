# Phase 7B — Streamlit Application Integration

## 1. Application architecture
The application acts strictly as a presentation and integration layer over the existing, frozen Phase 4 / Phase 6 machine learning pipeline. It is housed in `src/app.py` and relies entirely on Streamlit.

**Explicit Rule Status**: The Streamlit application reuses the frozen research inference pipeline and does not retrain or modify the evaluated model.

## 2. Streamlit interface
The interface is designed with a clean, academic structure. It includes:
- **Sidebar**: Context on the dataset, features, and model configurations.
- **Section 1**: Image Uploader (JPG/PNG).
- **Section 2**: Execution control ("Analyze Leaf" button) to prevent unnecessary re-rendering.
- **Section 3**: Diagnostic prediction (Crop, Disease, Status, and Confidence).
- **Section 4**: The integrated Agricultural Treatment Recommendation.
- **Section 5**: Real-time technical metrics (timing) demonstrating the lightweight claim.

## 3. Model loading
The application uses `@st.cache_resource` to securely load `svm_All24_RBF.pkl` directly from the `results/models/` artifacts directory exactly once per server lifecycle. It handles missing models gracefully with `st.error()` and `st.stop()`.

## 4. Inference pipeline
The uploaded RGB image buffer directly enters the existing research logic:
- `cv2.resize(256x256) -> cv2.GaussianBlur(5x5)`
- `isolate_leaf_otsu()`
- `segment_diseased_region()`
- `extract_all_features()`

This guarantees zero computational deviation between the UI prediction and the Phase 4 test set benchmarks. A user optional expander allows visualization of the intermediate Otsu and K-Means masks without modifying their algorithms.

## 5. Feature compatibility
To strictly enforce feature ordering, the application loads the cached feature schema dynamically from the columns in `data/research_features.csv` via the label encoder loading function. This ensures the 24 input parameters are perfectly sorted exactly as they were during `GridSearchCV` training.

## 6. Recommendation integration
After the classifier emits the raw prediction string (e.g., `Tomato___Late_blight`), the UI calls `get_recommendation()` from `src/recommendations.py`. The resulting KB payload is rendered identically, suppressing "Disease Treatment" sections if the status equals `healthy`.

## 7. Error handling
Handled cases include:
- Missing models / feature data CSVs.
- Corrupted/unreadable image uploads (handled with `try/except` around PIL).
- Feature extraction math faults.
- Classifier prediction failures.
- Unmatched KB labels (graceful "unknown" fallback).

## 8. Performance handling
`st.cache_resource` prevents Python from thrashing the disk or continuously deserializing the SVM on every button click. Inference is triggered purely sequentially through the "Analyze Leaf" button.

## 9. Testing
Unit testing (`tests/test_app.py`) verified:
- Model loading works.
- Label encoder mapping dynamically recovers exactly 24 feature strings.
- Passing a synthetic numpy array directly through `extract_features_pipeline` into the loaded SVM predictor successfully returns a valid string prediction without crashing or dropping dimensions.

## 10. Known limitations
- The SVM model created in Phase 4 does not expose probabilistic estimates (`predict_proba` is disabled by default in scikit-learn `SVC`). Therefore, "Confidence" displays as "Not available".
- Timing metrics rendered in the browser include minor memory/I-O UI overhead and should not replace the formal Phase 6 benchmarks.

## 11. Usage instructions
1. Activate the virtual environment.
2. Run `streamlit run src/app.py`.
3. Upload a leaf image from the dataset or field.
4. Review the automated prediction and associated management advice.
