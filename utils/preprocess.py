import numpy as np
import cv2
from PIL import Image
import imutils
import tensorflow as tf

IMG_SIZE = (224, 224)


def crop_brain_contour(image_np):
    """
    Crop the brain region from an MRI image using contour detection.
    Removes dark background and focuses the model on brain tissue.
    Technique from: MohamedAliHabib/Brain-Tumor-Detection

    Args:
        image_np: numpy array (H, W, 3) in BGR or RGB format

    Returns:
        Cropped numpy array, or original if cropping fails
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) == 0:
            return image_np

        c = max(cnts, key=cv2.contourArea)
        ext_left  = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top   = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot   = tuple(c[c[:, :, 1].argmax()][0])

        cropped = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]

        # Guard against degenerate crops
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return image_np

        return cropped
    except Exception:
        return image_np


def load_and_preprocess_image(file_storage):
    """
    Preprocess an uploaded MRI image for MobileNetV2 inference.

    Pipeline:
        1. Open & convert to RGB
        2. Brain-region crop (removes background noise)
        3. Resize to 224×224
        4. Apply MobileNetV2 preprocess_input  →  pixel range [-1, 1]

    Args:
        file_storage: werkzeug FileStorage from Flask upload

    Returns:
        (image_array, pil_image_original)
          image_array: shape (1, 224, 224, 3), dtype float32, range [-1, 1]
          pil_image_original: PIL Image (resized, pre-crop) for display
    """
    pil_image = Image.open(file_storage).convert("RGB")
    pil_display = pil_image.resize(IMG_SIZE)  # for display purposes

    # Convert to numpy for OpenCV cropping
    img_np = np.array(pil_image)

    # Brain crop
    img_np = crop_brain_contour(img_np)

    # Resize to model input size
    img_np = cv2.resize(img_np, IMG_SIZE, interpolation=cv2.INTER_CUBIC)

    # MobileNetV2 preprocessing: scales [0,255] → [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_np.astype("float32")
    )
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, pil_display
