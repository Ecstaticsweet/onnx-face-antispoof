"""Image preprocessing utilities."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2


def preprocess_face(
    face_crop: np.ndarray,
    target_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess face crop for inference.

    Args:
        face_crop: RGB image (H, W, 3)
        target_size: Model input size
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed tensor (1, 3, target_size, target_size)
    """
    # Resize with high-quality interpolation
    resized = cv2.resize(
        face_crop,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )

    # Convert to float
    processed = resized.astype(np.float32)

    # Normalize to [0, 1]
    if normalize:
        processed = processed / 255.0

    # Transpose to (C, H, W)
    processed = np.transpose(processed, (2, 0, 1))

    # Add batch dimension (1, C, H, W)
    processed = np.expand_dims(processed, axis=0)

    return processed


def preprocess_batch(
    face_crops: List[np.ndarray],
    target_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess batch of face crops for inference.

    Args:
        face_crops: List of RGB images
        target_size: Model input size
        normalize: Whether to normalize to [0, 1]

    Returns:
        Batched tensor (N, 3, target_size, target_size)
    """
    batch = []

    for face_crop in face_crops:
        processed = preprocess_face(face_crop, target_size, normalize)
        batch.append(processed)

    # Stack into single batch (N, 3, H, W)
    batched = np.vstack(batch)

    return batched


def expand_bbox(
    bbox: Dict[str, float],
    img_width: int,
    img_height: int,
    expansion: float = 1.5,
) -> Dict[str, int]:
    """
    Expand bounding box with safe boundary handling.

    Args:
        bbox: Dictionary with 'x', 'y', 'width', 'height'
        img_width: Image width in pixels
        img_height: Image height in pixels
        expansion: Expansion factor (1.5 recommended)

    Returns:
        Expanded bbox clipped to image boundaries
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    # Calculate center
    cx = x + w / 2
    cy = y + h / 2

    # Expand
    new_w = w * expansion
    new_h = h * expansion

    # Recalculate top-left
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    # Clip to image boundaries
    x1 = max(0, int(new_x))
    y1 = max(0, int(new_y))
    x2 = min(img_width, int(new_x + new_w))
    y2 = min(img_height, int(new_y + new_h))

    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def crop_face(
    image: np.ndarray,
    bbox: Dict[str, int],
) -> np.ndarray:
    """
    Crop face from image using bounding box.

    Args:
        image: RGB image (H, W, 3)
        bbox: Bounding box with 'x', 'y', 'width', 'height'

    Returns:
        Cropped face image
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    face_crop = image[y : y + h, x : x + w]
    return face_crop


def enhance_lighting(
    image: np.ndarray,
    mean_intensity: float,
    std_intensity: float,
) -> np.ndarray:
    """
    Apply adaptive enhancement based on lighting conditions.

    Warning: May amplify noise or artifacts. Use with caution.

    Args:
        image: RGB image
        mean_intensity: Mean intensity from quality check
        std_intensity: Std intensity from quality check

    Returns:
        Enhanced image
    """
    enhanced = image.copy()

    # Gamma correction for dark images
    if mean_intensity < 50:
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    # CLAHE for low contrast
    elif std_intensity < 20:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced
