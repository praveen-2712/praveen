import cv2
import numpy as np
import torch
from skimage.exposure import match_histograms
import os


class Preprocess:
    """
    Engineering-grade clinical preprocessor for Neurologix Pro V3.2.

    Pipeline per slice:
      1. Histogram matching to golden reference  (scanner normalisation)
      2. CLAHE contrast enhancement              (BUG-06 fix)
      3. Resize to 224x224

    Brain-mask pipeline (BUG-07 fix):
      - Bilateral filter (edge-preserving denoise)
      - 5 % border strip (removes scanner overlays/text)
      - Otsu threshold on the interior ROI
      - Morphological open/close
      - Keep largest connected contour
    """

    def __init__(self, golden_ref_path=None):
        if golden_ref_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            golden_ref_path = os.path.join(base_dir, "assets", "golden_reference.jpg")

        if os.path.exists(golden_ref_path):
            self.golden_ref = cv2.imread(golden_ref_path, cv2.IMREAD_GRAYSCALE)
            print(f"[Preprocess] Scanner Normalisation Active "
                  f"(Reference: {os.path.basename(golden_ref_path)})")
        else:
            self.golden_ref = None
            print("[Preprocess] WARNING: Normalisation Offline (Golden Reference Missing)")

        # BUG-06 FIX: CLAHE instance (reused across calls for efficiency)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_clahe" in state:
            del state["_clahe"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ──────────────────────────────────────────────────────────────────
    #  Intensity normalisation
    # ──────────────────────────────────────────────────────────────────

    def apply_normalization(self, image_np):
        """Standardise scan intensity via Histogram Matching to golden ref."""
        if self.golden_ref is None:
            return image_np

        process_img = (image_np if len(image_np.shape) == 2
                       else cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY))
        ref = self.golden_ref
        if len(ref.shape) == 3:
            ref = ref.squeeze()
        if len(process_img.shape) == 3:
            process_img = process_img.squeeze()

        matched = match_histograms(process_img, ref, channel_axis=None)
        return matched.astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────
    #  Brain mask (BUG-07 fix)
    # ──────────────────────────────────────────────────────────────────

    def get_brain_mask(self, image_np):
        """
        Robust brain mask using:
         - Bilateral filter   : edge-preserving denoising
         - Border strip       : removes scanner text / bright frame artefacts
         - Interior-only Otsu : prevents scanner border from skewing threshold
         - Morph open/close   : fills holes, removes noise
         - Largest contour    : isolates the brain parenchyma
        """
        gray = (image_np if len(image_np.shape) == 2
                else cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY))

        h, w = gray.shape

        # Step 1: bilateral filter (preserves tissue boundaries)
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 2: strip outer 5 % border (scanner text / bright frame)
        border_y = max(1, int(0.05 * h))
        border_x = max(1, int(0.05 * w))
        roi = filtered[border_y: h - border_y, border_x: w - border_x]

        # Step 3: Otsu on the interior ROI
        _, thresh_roi = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Place back into full-size mask
        thresh = np.zeros_like(filtered)
        thresh[border_y: h - border_y, border_x: w - border_x] = thresh_roi

        # Step 4: morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask   = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, kernel, iterations=2)

        # Step 5: keep only the single largest contour (the brain)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        full_mask = np.zeros_like(mask)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(full_mask, [c], -1, 255, -1)

        return full_mask

    # ──────────────────────────────────────────────────────────────────
    #  Brain crop
    # ──────────────────────────────────────────────────────────────────

    def crop_brain_contour(self, image_np):
        """Crop image to brain bounding box for focused analysis."""
        mask   = self.get_brain_mask(image_np)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image_np
        x, y, bw, bh = cv2.boundingRect(coords)
        return image_np[y: y + bh, x: x + bw]

    # ──────────────────────────────────────────────────────────────────
    #  2.5D stack (BUG-06 fix: CLAHE added)
    # ──────────────────────────────────────────────────────────────────

    def prepare_stack(self, slices, use_norm=True, use_clahe=True, img_size=224):
        """
        Build a (img_size, img_size, 3) 2.5D stack from three consecutive slices.

        Per slice pipeline:
          histogram normalise → CLAHE enhance → resize img_size×img_size
        """
        processed_slices = []
        for s in slices:
            # Ensure grayscale
            if len(s.shape) == 3:
                s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

            # 1. Scanner normalisation
            if use_norm:
                s = self.apply_normalization(s)

            # 2. CLAHE: boost local contrast for lesion visibility (BUG-06)
            if use_clahe:
                s = self._clahe.apply(s)

            # 3. Resize to model input size
            s_final = cv2.resize(s, (img_size, img_size),
                                 interpolation=cv2.INTER_AREA)
            processed_slices.append(s_final)

        # (H, W, 3) — channels are (t-1, t, t+1)
        stack = np.stack(processed_slices, axis=-1)
        return stack
