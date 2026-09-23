import cv2
import numpy as np

def isolate_leaf_otsu(rgb_img):
    """
    Isolates leaf from background using Otsu thresholding combined with Green-Difference index.
    
    Returns:
        leaf_mask: uint8 binary mask (255 for leaf, 0 for background)
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    
    # Compute Excess Green index: 2G - R - B (highlights plant foliage)
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Otsu's Thresholding on ExG & Grayscale
    _, otsu_exg = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine masks
    combined = cv2.bitwise_or(otsu_exg, otsu_gray)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest contour (the leaf)
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        return clean_mask
    
    return leaf_mask

def segment_diseased_region(rgb_img, hsv_img, leaf_mask, k=3):
    """
    Segments diseased spots from healthy leaf tissue using HSV color thresholding + K-means clustering.
    
    Returns:
        dict containing:
            'leaf_mask': binary mask of leaf
            'diseased_mask': binary mask of diseased spots
            'healthy_mask': binary mask of healthy leaf area
            'segmented_roi': diseased region extracted
            'highlighted_overlay': RGB image with diseased contours highlighted in red
    """
    if np.count_nonzero(leaf_mask) == 0:
        # Fallback if empty
        leaf_mask = np.full((rgb_img.shape[0], rgb_img.shape[1]), 255, dtype=np.uint8)

    # 1. Extract leaf pixels in HSV space
    hsv_pixels = hsv_img[leaf_mask > 0].astype(np.float32)
    
    if len(hsv_pixels) < k:
        diseased_mask = np.zeros_like(leaf_mask)
        healthy_mask = leaf_mask.copy()
    else:
        # Set OpenCV RNG seed for deterministic K-Means
        cv2.setRNGSeed(42)
        
        # K-Means clustering on HSV values of leaf pixels
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(hsv_pixels, k, None, criteria, 10, flags)

        # Reconstruct label map for the leaf
        cluster_map = np.zeros(leaf_mask.shape, dtype=np.uint8)
        cluster_map[leaf_mask > 0] = labels.flatten() + 1 # 1-indexed clusters

        # Determine which cluster is healthy green vs diseased (necrotic/yellow/brown/dark)
        # Healthy green hue range in OpenCV HSV is ~35 to 85
        cluster_health_scores = []
        for c in range(1, k + 1):
            c_hsv = centers[c - 1]
            hue = c_hsv[0]
            sat = c_hsv[1]
            val = c_hsv[2]
            
            # Distance from ideal healthy green hue (~50) and high saturation
            green_dist = abs(hue - 50)
            health_score = green_dist - (sat * 0.1) # lower green_dist = healthier
            cluster_health_scores.append((health_score, c))

        # Sort clusters by health score (healthiest green first, most diseased last)
        cluster_health_scores.sort()
        
        # Diseased clusters are the non-healthy clusters with lower hue / dark / yellow values
        diseased_mask = np.zeros_like(leaf_mask)
        for _, c in cluster_health_scores[1:]: # Take less healthy clusters
            diseased_mask[cluster_map == c] = 255

        # Also apply explicit HSV color thresholding for necrotic brown/yellow spots
        # Hue ranges: Brown/Yellow: 0-30 or 150-180
        h_channel = hsv_img[:, :, 0]
        s_channel = hsv_img[:, :, 1]
        v_channel = hsv_img[:, :, 2]
        
        necrotic_hsv = (
            ((h_channel <= 32) | (h_channel >= 150) | (s_channel <= 40) | (v_channel <= 70)) & 
            (leaf_mask > 0)
        )
        diseased_mask[necrotic_hsv] = 255

    # Morphological cleanup on diseased mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    diseased_mask = cv2.bitwise_and(diseased_mask, leaf_mask)
    
    # --- LESION VALIDITY CHECK ---
    # To prevent shadows/veins on perfectly healthy leaves from being universally
    # labeled as lesions, verify the candidate mask contains genuine necrotic color.
    # Necrotic: Brown/Yellow/Red (Hue <= 33 or Hue >= 150) with sufficient saturation & value.
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    
    true_necrotic = (
        ((h_channel <= 33) | (h_channel >= 150)) & 
        (s_channel > 25) & 
        (v_channel > 25) & 
        (diseased_mask > 0)
    )
    
    leaf_area = np.count_nonzero(leaf_mask)
    true_necrotic_area = np.count_nonzero(true_necrotic)
    
    # Require at least 0.5% of the leaf to be definitively necrotic, OR at least 50 absolute pixels
    if true_necrotic_area < (0.005 * leaf_area) and true_necrotic_area < 50:
        diseased_mask = np.zeros_like(leaf_mask)
    # -----------------------------

    healthy_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(diseased_mask))

    # Segmented ROI
    segmented_roi = cv2.bitwise_and(rgb_img, rgb_img, mask=diseased_mask)

    # Highlighted Overlay: draw bright red contours around diseased regions on RGB image
    highlighted_overlay = rgb_img.copy()
    contours, _ = cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours in vibrant red (RGB: 239, 68, 68) with semi-transparent fill
    cv2.drawContours(highlighted_overlay, contours, -1, (239, 68, 68), 2)
    
    # Add subtle tint on diseased area
    overlay_tint = highlighted_overlay.copy()
    overlay_tint[diseased_mask > 0] = (
        overlay_tint[diseased_mask > 0] * 0.7 + np.array([220, 38, 38]) * 0.3
    ).astype(np.uint8)
    
    highlighted_overlay = cv2.addWeighted(highlighted_overlay, 0.75, overlay_tint, 0.25, 0)

    return {
        'leaf_mask': leaf_mask,
        'diseased_mask': diseased_mask,
        'healthy_mask': healthy_mask,
        'segmented_roi': segmented_roi,
        'highlighted_overlay': highlighted_overlay
    }
