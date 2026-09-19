import cv2
import numpy as np

def load_and_preprocess_image(image_input, target_size=(256, 256), blur_kernel=(5, 5)):
    """
    Loads and preprocesses an input image (file path, array, or bytes).
    
    Args:
        image_input: str path, np.ndarray (BGR or RGB), or bytes.
        target_size: (width, height) tuple for resizing.
        blur_kernel: (width, height) Gaussian blur kernel.
        
    Returns:
        dict containing:
            'original_rgb': RGB image (uint8)
            'resized_rgb': Resized RGB image (256x256)
            'blurred_rgb': Gaussian blurred RGB image
            'gray': Grayscale image
            'hsv': HSV color space image
    """
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image_input.copy()
    else:
        raise TypeError("Unsupported image_input type.")

    # 1. Resize image to fixed size
    resized_rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    # 2. Gaussian Blur for noise reduction
    blurred_rgb = cv2.GaussianBlur(resized_rgb, blur_kernel, 0)

    # 3. Color Space Conversions
    gray = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2HSV)

    return {
        'original_rgb': rgb,
        'resized_rgb': resized_rgb,
        'blurred_rgb': blurred_rgb,
        'gray': gray,
        'hsv': hsv
    }
