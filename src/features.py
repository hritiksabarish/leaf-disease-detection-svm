import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_color_features(hsv_image, mask):
    """
    Extracts mean and standard deviation of H, S, V channels in the masked region.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        mask = np.ones((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8)

    pixels = hsv_image[mask > 0]
    
    mean_h = np.mean(pixels[:, 0]) if len(pixels) > 0 else 0.0
    std_h = np.std(pixels[:, 0]) if len(pixels) > 0 else 0.0
    
    mean_s = np.mean(pixels[:, 1]) if len(pixels) > 0 else 0.0
    std_s = np.std(pixels[:, 1]) if len(pixels) > 0 else 0.0
    
    mean_v = np.mean(pixels[:, 2]) if len(pixels) > 0 else 0.0
    std_v = np.std(pixels[:, 2]) if len(pixels) > 0 else 0.0

    return {
        'color_mean_h': mean_h,
        'color_std_h': std_h,
        'color_mean_s': mean_s,
        'color_std_s': std_s,
        'color_mean_v': mean_v,
        'color_std_v': std_v,
    }

def extract_glcm_features(gray_image, mask):
    """
    Extracts GLCM texture features, strictly removing background bias by using a background class.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return { 'glcm_contrast': 0.0, 'glcm_dissimilarity': 0.0, 'glcm_homogeneity': 0.0, 
                 'glcm_energy': 0.0, 'glcm_correlation': 0.0, 'glcm_asm': 0.0 }

    # Get bounding box to reduce computation
    x, y, w, h = cv2.boundingRect(mask)
    crop_gray = gray_image[y:y+h, x:x+w]
    crop_mask = mask[y:y+h, x:x+w]

    # Quantize to 32 levels
    region_quantized = (crop_gray // 8).astype(np.uint8)
    # Assign background to a special 33rd bin (index 32)
    region_quantized[crop_mask == 0] = 32

    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Compute unnormalized GLCM with 33 levels
    glcm_full = graycomatrix(
        region_quantized,
        distances=distances,
        angles=angles,
        levels=33,
        symmetric=True,
        normed=False
    )

    # Slice out the background bin (index 32)
    glcm = glcm_full[:32, :32, :, :]
    
    # Re-normalize
    glcm_sum = np.sum(glcm, axis=(0, 1), keepdims=True)
    glcm_sum[glcm_sum == 0] = 1 # Avoid division by zero
    glcm_normed = glcm / glcm_sum

    contrast = np.mean(graycoprops(glcm_normed, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm_normed, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm_normed, 'homogeneity'))
    energy = np.mean(graycoprops(glcm_normed, 'energy'))
    correlation = np.mean(graycoprops(glcm_normed, 'correlation'))
    asm = np.mean(graycoprops(glcm_normed, 'ASM'))

    return {
        'glcm_contrast': float(contrast),
        'glcm_dissimilarity': float(dissimilarity),
        'glcm_homogeneity': float(homogeneity),
        'glcm_energy': float(energy),
        'glcm_correlation': float(correlation),
        'glcm_asm': float(asm)
    }

def extract_shape_features(mask, prefix='shape_'):
    """
    Extracts geometric contour shape features.
    Prefix allows separation of leaf vs lesion features.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return {
            f'{prefix}area': 0.0,
            f'{prefix}perimeter': 0.0,
            f'{prefix}eccentricity': 0.0,
            f'{prefix}extent': 0.0,
            f'{prefix}solidity': 0.0,
            f'{prefix}compactness': 0.0
        }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            f'{prefix}area': 0.0,
            f'{prefix}perimeter': 0.0,
            f'{prefix}eccentricity': 0.0,
            f'{prefix}extent': 0.0,
            f'{prefix}solidity': 0.0,
            f'{prefix}compactness': 0.0
        }

    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = area / float(rect_area) if rect_area > 0 else 0.0

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0.0

    compactness = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    eccentricity = 0.0
    if len(largest_contour) >= 5:
        try:
            (cx, cy), (ma, MA), angle = cv2.fitEllipse(largest_contour)
            a = MA / 2.0
            b = ma / 2.0
            if a > 0 and a >= b:
                eccentricity = np.sqrt(1 - (b / a)**2)
        except Exception:
            eccentricity = 0.0

    return {
        f'{prefix}area': float(total_area),
        f'{prefix}perimeter': float(total_perimeter),
        f'{prefix}eccentricity': float(eccentricity),
        f'{prefix}extent': float(extent),
        f'{prefix}solidity': float(solidity),
        f'{prefix}compactness': float(compactness)
    }

def extract_all_features(rgb_image, hsv_image, gray_image, diseased_mask, leaf_mask):
    """
    Combines color, texture, and distinct leaf/lesion shape features.
    """
    # Color and Texture should only measure the diseased region if it exists
    color_mask = diseased_mask if np.count_nonzero(diseased_mask) > 0 else leaf_mask
    texture_mask = diseased_mask if np.count_nonzero(diseased_mask) > 0 else leaf_mask

    color_feats = extract_color_features(hsv_image, color_mask)
    glcm_feats = extract_glcm_features(gray_image, texture_mask)
    
    # SEPARATE Shape Features to prevent leakage!
    leaf_shape_feats = extract_shape_features(leaf_mask, prefix='leaf_')
    lesion_shape_feats = extract_shape_features(diseased_mask, prefix='lesion_')

    all_feats = {}
    all_feats.update(color_feats)
    all_feats.update(glcm_feats)
    all_feats.update(leaf_shape_feats)
    all_feats.update(lesion_shape_feats)

    feature_names = sorted(all_feats.keys())
    feature_vector = np.array([all_feats[k] for k in feature_names], dtype=np.float32)

    return all_feats, feature_vector, feature_names
