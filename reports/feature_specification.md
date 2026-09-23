# Feature Specification

## Overview
The phase 2 feature pipeline extracts exactly **24 handcrafted features** from each leaf image. It distinguishes explicitly between leaf shape and lesion shape, eliminating methodological label leakage. It also uses a bounded region and explicitly handles background quantization for texture features, preventing background dominance.

## Groups and Features

### 1. COLOR (6 features)
Calculated from the HSV image masked by the diseased region (if present) or the entire leaf (if healthy).
- **`color_mean_h`**: Mean Hue.
- **`color_mean_s`**: Mean Saturation.
- **`color_mean_v`**: Mean Value.
- **`color_std_h`**: Standard deviation of Hue.
- **`color_std_s`**: Standard deviation of Saturation.
- **`color_std_v`**: Standard deviation of Value.
*Reason for inclusion*: Fundamental indicators of necrotic (brown/yellow) vs healthy (green) tissue.

### 2. TEXTURE (6 features)
Calculated from the Grayscale image masked by the diseased region (or leaf), quantized to 32 levels. Background pixels are assigned to a special 33rd bin which is sliced out before normalization.
- **`glcm_contrast`**: Local intensity variation.
- **`glcm_dissimilarity`**: Average absolute difference of neighbor pairs.
- **`glcm_homogeneity`**: Closeness of the distribution of elements to the GLCM diagonal.
- **`glcm_energy`**: Sum of squared elements in the GLCM (angular second moment).
- **`glcm_correlation`**: Linear dependency of gray levels on neighboring pixels.
- **`glcm_asm`**: Angular Second Moment.
*Reason for inclusion*: Captures structural changes caused by fungal/bacterial spread on the leaf surface.

### 3. LEAF_SHAPE (6 features)
Calculated exclusively from the `leaf_mask`.
- **`leaf_area`**: Total pixel area of the leaf contour.
- **`leaf_perimeter`**: Contour perimeter.
- **`leaf_eccentricity`**: Eccentricity of the fitted ellipse (0 = circle).
- **`leaf_extent`**: Ratio of area to the bounding box area.
- **`leaf_solidity`**: Ratio of area to the convex hull area.
- **`leaf_compactness`**: Metric ratio ($4 \pi \times Area / Perimeter^2$).
*Reason for inclusion*: Quantifies the macro-structure and edge integrity of the leaf. Separating leaf shape from lesion shape fixes previous data leakage.

### 4. LESION_SHAPE (6 features)
Calculated exclusively from the `diseased_mask`. If the leaf is healthy and the validity check determines there is no convincing evidence of a lesion, the mask is completely empty. In this case, a deterministic fallback is used: all lesion shape descriptors (`lesion_area`, `lesion_perimeter`, etc.) evaluate strictly to `0.0`. This prevents `NaN`s and explicitly teaches the model the absence of lesions.
- **`lesion_area`**: Total pixel area of all diseased spots.
- **`lesion_perimeter`**: Combined perimeter of the spots.
- **`lesion_eccentricity`**: Eccentricity of the largest lesion spot.
- **`lesion_extent`**: Extent of the largest lesion spot.
- **`lesion_solidity`**: Solidity of the largest lesion spot.
- **`lesion_compactness`**: Compactness of the largest lesion spot.
*Reason for inclusion*: Directly measures the spread, clustering, and geometry of the infection. Healthy images gracefully receive 0.0, which naturally trains the model to recognize healthy states.
