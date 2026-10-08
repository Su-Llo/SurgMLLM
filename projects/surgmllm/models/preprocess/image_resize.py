"""The second image path used by the SAM2-H encoder."""

import numpy as np
from PIL import Image


class DirectResize:
    def __init__(self, target_length: int = 1024) -> None:
        self.target_length = int(target_length)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got {image.shape}")
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        return np.asarray(
            pil_image.resize((self.target_length, self.target_length), Image.Resampling.BILINEAR)
        )
