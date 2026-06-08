import base64
import io

import cv2
import numpy as np
from PIL import Image


def load_and_gray(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def binarize(gray):
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def remove_noise(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def deskew(binary):
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = binary.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def horizontal_projection(binary):
    return np.sum(binary, axis=1)


def extract_line_positions(binary, threshold_ratio=0.2):
    projection = horizontal_projection(binary)
    max_val = np.max(projection)
    if max_val == 0:
        return []

    threshold = max_val * threshold_ratio
    lines = []
    in_line = False
    start = 0

    for i, value in enumerate(projection):
        if value > threshold and not in_line:
            in_line = True
            start = i
        elif value <= threshold and in_line:
            end = i
            if end - start > 5:
                lines.append((start, end))
            in_line = False

    if in_line:
        lines.append((start, len(projection) - 1))

    return lines


def crop_lines(binary, lines, padding_top=10, padding_bottom=10):
    line_images = []
    h = binary.shape[0]
    for start, end in lines:
        start = max(0, start - padding_top)
        end = min(h, end + padding_bottom)
        line_images.append(binary[start:end, :])
    return line_images


def normalize_height(img, target_height=64):
    h, w = img.shape
    if h == 0:
        return img
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_height))


def _array_to_png_base64(image_array):
    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _read_image_bytes(image_bytes):
    array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image. Upload a valid PNG or JPEG file.")
    return image_bgr


def extract_lines_from_bytes(image_bytes, return_debug=False):
    """Extract line crops from a document image."""
    image_bgr = _read_image_bytes(image_bytes)
    gray = load_and_gray(image_bgr)
    blurred = denoise(gray)
    binary = binarize(blurred)
    cleaned = remove_noise(binary)
    deskewed = deskew(cleaned)
    line_positions = extract_line_positions(deskewed)
    line_images = crop_lines(deskewed, line_positions)

    lines = []
    for idx, line_img in enumerate(line_images):
        normalized = normalize_height(line_img, 64)
        rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        lines.append(
            {
                "index": idx,
                "image_base64": _array_to_png_base64(rgb),
            }
        )

    result = {
        "line_count": len(lines),
        "lines": lines,
    }

    if return_debug:
        result["debug"] = {
            "original": _array_to_png_base64(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
            "deskewed": _array_to_png_base64(deskewed),
        }

    return result
