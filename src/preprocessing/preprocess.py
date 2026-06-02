import cv2
import numpy as np

image_path = "./download.jpg"
output_folder = "./"

def load_and_gray(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def denoise(gray):
    return cv2.GaussianBlur(gray, (5,5), 0)

def binarize(gray):
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary

def remove_noise(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned

def deskew(binary):
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        binary,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

def horizontal_projection(binary):
    return np.sum(binary, axis=1)

def normalize_height(img, target_height=64):
    h, w = img.shape
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))

def extract_line_positions(binary, threshold_ratio=0.2):
    projection = horizontal_projection(binary)

    max_val = np.max(projection)
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
            if end - start > 5:  # remove noise lines
                lines.append((start, end))
            in_line = False

    # handle last line
    if in_line:
        lines.append((start, len(projection)-1))

    return lines
    
def crop_lines(binary, lines, padding_top=10, padding_bottom=10):
    line_images = []

    h = binary.shape[0]

    for (start, end) in lines:

        start = max(0, start - padding_top)
        end = min(h, end + padding_bottom)

        line_img = binary[start:end, :]
        line_images.append(line_img)

    return line_images

def save_lines(lines, output_folder):
    import os
    os.makedirs(output_folder, exist_ok=True)

    for i, line in enumerate(lines):
        line = normalize_height(line,64)
        cv2.imwrite(f"{output_folder}/line_{i}.png", line)
        

def process_document(image_path, output_folder):

    gray = load_and_gray(image_path)

    blurred = denoise(gray)

    binary = binarize(blurred)

    cleaned = remove_noise(binary)
    # cleaned=binary

    deskewed = deskew(cleaned)

    lines = extract_line_positions(deskewed)

    line_images = crop_lines(deskewed, lines)

    save_lines(line_images, output_folder)

    print(f"{len(line_images)} lines extracted")
    
process_document(image_path, output_folder)