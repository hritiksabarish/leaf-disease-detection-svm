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
    Extracts GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM).
    """
    if mask is not None and np.count_nonzero(mask) > 0:
        # Mask out non-region pixels
        region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    else:
        region = gray_image

    # Reduce gray levels to 32 bins for speed & robustness
    region_quantized = (region // 8).astype(np.uint8)
    
    # Compute GLCM at distances 1, 3 and angles 0, 45, 90, 135 deg
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        region_quantized,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    asm = np.mean(graycoprops(glcm, 'ASM'))

    return {
        'glcm_contrast': float(contrast),
        'glcm_dissimilarity': float(dissimilarity),
        'glcm_homogeneity': float(homogeneity),
        'glcm_energy': float(energy),
        'glcm_correlation': float(correlation),
        'glcm_asm': float(asm)
    }

def extract_shape_features(mask):
    """
    Extracts geometric contour shape features: area, perimeter, eccentricity, extent, solidity, metric ratio.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return {
            'shape_area': 0.0,
            'shape_perimeter': 0.0,
            'shape_eccentricity': 0.0,
            'shape_extent': 0.0,
            'shape_solidity': 0.0,
            'shape_compactness': 0.0
        }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'shape_area': 0.0,
            'shape_perimeter': 0.0,
            'shape_eccentricity': 0.0,
            'shape_extent': 0.0,
            'shape_solidity': 0.0,
            'shape_compactness': 0.0
        }

    # Aggregate over contours
    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

    # Largest contour for shape parameters
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Bounding Box Extent
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect_area = w * h
    extent = area / float(rect_area) if rect_area > 0 else 0.0

    # Convex Hull Solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0.0

    # Compactness / Metric Ratio (4 * pi * area / perimeter^2)
    compactness = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Eccentricity via Minimum Enclosing Ellipse or FitEllipse
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
        'shape_area': float(total_area),
        'shape_perimeter': float(total_perimeter),
        'shape_eccentricity': float(eccentricity),
        'shape_extent': float(extent),
        'shape_solidity': float(solidity),
        'shape_compactness': float(compactness)
    }

def extract_all_features(rgb_image, hsv_image, gray_image, diseased_mask, leaf_mask):
    """
    Combines color, texture, and shape features into a dictionary and 1D vector.
    """
    # Use diseased_mask if non-empty, otherwise leaf_mask (for healthy leaves)
    feature_mask = diseased_mask if np.count_nonzero(diseased_mask) > 0 else leaf_mask

    color_feats = extract_color_features(hsv_image, feature_mask)
    glcm_feats = extract_glcm_features(gray_image, feature_mask)
    shape_feats = extract_shape_features(feature_mask)

    all_feats = {}
    all_feats.update(color_feats)
    all_feats.update(glcm_feats)
    all_feats.update(shape_feats)

    # Feature vector as float32 array
    feature_names = sorted(all_feats.keys())
    feature_vector = np.array([all_feats[k] for k in feature_names], dtype=np.float32)

    return all_feats, feature_vector, feature_names
