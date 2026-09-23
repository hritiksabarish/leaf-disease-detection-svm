# Phase 6 — Efficiency Benchmarking

## Objective
To quantitatively measure the computational efficiency, model size, and deployment feasibility of the handcrafted-feature SVM framework to determine if it meets the criteria of a "lightweight" architecture.

## Hardware/software environment
The benchmarking environment was rigorously documented in `reports/benchmark_environment.md`. It utilized a Windows desktop CPU running Python 3.12. The experiment did *not* utilize GPU acceleration, as the core OpenCV/Scikit-Learn pipeline is fundamentally CPU-oriented.

## Models and Feature configurations
The benchmark measured the exact finalized model artifacts from Phase 4 across both configurations (All 24 Features and Selected 20 Features) for all four kernel types.

## Feature-extraction benchmark
The feature extraction pipeline includes image decoding, resizing, Gaussian blur, Otsu leaf segmentation, K-Means lesion masking, and the extraction of 24 handcrafted color, texture, and shape descriptors.
- **Mean feature extraction time**: ~58.54 ms per image (std: ~13.93 ms)
- *(Note: Feature extraction is constant regardless of the downstream SVM configuration)*

## SVM inference benchmark
Once the 24 (or 20) numeric features are extracted, the trained SVM must predict the class.
- **RBF (All24)**: ~0.12 ms per image
- **Linear (All24)**: ~0.04 ms per image
- **Polynomial (All24)**: ~0.04 ms per image
- **Quadratic (All24)**: ~0.03 ms per image
- *(Note: Predictions are nearly instantaneous, with RBF being marginally slower due to computing the exponential distance against ~1,100 support vectors)*

## End-to-end latency
Combining the image processing pipeline and the SVM predictor:
- **Total latency (RBF All24)**: ~58.66 ms per image
- **System throughput**: ~17.05 images per second (single-threaded CPU)

## Training-time benchmark
Training the models from scratch on the 1,561-image training set (assuming features are pre-extracted and cached):
- **RBF (All24)**: ~0.10 seconds
- **Linear (All24)**: ~0.11 seconds
- **Polynomial (All24)**: ~0.09 seconds

## Model size and Support vectors
- **RBF (All24)**: 294.7 KB (1,114 support vectors)
- **Polynomial (All24)**: 286.6 KB (1,083 support vectors)
- **Linear (All24)**: 251.8 KB (950 support vectors)
- **RBF (Selected20)**: 220.6 KB (941 support vectors)

## Memory
Peak memory was not continuously profiled to avoid profiler overhead distorting the millisecond-scale latency measurements. However, the model serialization size (<300 KB) heavily implies an extremely small RAM footprint during inference, bounded primarily by the memory required to hold a single 256x256 image matrix during OpenCV operations.

## Interpretation
The framework strongly justifies the "lightweight" claim:
1. **The SVM is microscopic**: With serialized sizes under 300 KB and training times of ~0.1 seconds, the ML component is highly deployable to edge environments.
2. **Inference is dominated by OpenCV**: Feature extraction requires ~58.5 ms, whereas SVM prediction requires ~0.1 ms. The computer vision pipeline accounts for **>99.8%** of the computational cost.
3. **CPU Edge-Ready**: At ~17 FPS on a standard CPU thread, the system is more than fast enough for real-time video diagnosis or instant mobile-app processing without requiring a GPU.

## Limitations
1. Measurements were taken on a desktop CPU. Performance on low-power mobile or IoT chips (like a Raspberry Pi) will be slower and should be independently benchmarked if edge deployment is the primary target.
2. The benchmark did not explicitly isolate the cost of GLCM calculation versus K-Means clustering inside the 58.5 ms extraction window, though GLCM is typically the most expensive operation.
