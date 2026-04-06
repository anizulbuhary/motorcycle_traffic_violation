# === Standard Library ===
import hashlib  # For generating a hash
import os
import re
from pathlib import Path
from urllib.parse import urlparse

# === Third-party Libraries ===
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
# import easyocr

# === Django ===
from django.conf import settings
from django.utils.text import get_valid_filename

# Initialize the EasyOCR reader with the desired language (e.g., English)
#reader = easyocr.Reader(['en'])  # Load the English language model

# Construct absolute paths to the YOLO model files
license_plate_path = os.path.join(settings.BASE_DIR, 'motorcycle_violation', 'yolo_models', 'license_plate_detection_n.pt')
license_plate_char_path = os.path.join(settings.BASE_DIR, 'motorcycle_violation', 'yolo_models', 'license_plate_char_detetction_m.pt')
motorcycle_model_path = os.path.join(settings.BASE_DIR, 'motorcycle_violation', 'yolo_models', 'yolo11n.pt')

# Load the models using the absolute paths
license_plate_model = YOLO(license_plate_path)  # New model
license_plate_char_model = YOLO(license_plate_char_path)
motorcycle_model = YOLO(motorcycle_model_path)

def is_too_close_to_corner(x1, y1, x2, y2, image_width, image_height, margin=10):
    """
    Checks if any corner of the bounding box is too close to the image's corners.
    
    Parameters:
    - x1, y1: Top-left corner of the bounding box.
    - x2, y2: Bottom-right corner of the bounding box.
    - image_width, image_height: Dimensions of the image.
    - margin: The margin distance to consider as "too close" to the corner.
    
    Returns:
    - bool: True if any corner is too close to the image's corners, False otherwise.
    """
    # Check proximity to image corners
    corners = [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x1, y2),  # Bottom-left
        (x2, y2)   # Bottom-right
    ]
    
    image_corners = [
        (0, 0),  # Top-left corner
        (image_width, 0),  # Top-right corner
        (0, image_height),  # Bottom-left corner
        (image_width, image_height)  # Bottom-right corner
    ]
    
    for corner in corners:
        for img_corner in image_corners:
            dist = np.sqrt((corner[0] - img_corner[0]) ** 2 + (corner[1] - img_corner[1]) ** 2)
            if dist <= margin:
                return True
    return False


def extract_double_line_license_plate_text(bounding_boxes, image_height, image_width):
    """
    Extracts license plate text from bounding boxes for a double-line plate based on the provided rules:
    - Separates characters into rows above and below the mid-line.
    - Sorts rows in x-direction.
    - Follows rules for mandatory and optional parts of the license plate.
    """
    def class_id_to_char(class_id):
        if 0 <= class_id <= 25:  # A-Z
            return chr(class_id + ord('A'))
        elif 26 <= class_id <= 35:  # 0-9
            return chr(class_id - 26 + ord('0'))
        return "?"

    # Separate bounding boxes into first and second rows based on mid-line
    valid_boxes = [
        (x1, y1, x2, y2, confidence, class_id)
        for x1, y1, x2, y2, confidence, class_id in bounding_boxes
        if not is_too_close_to_corner(x1, y1, x2, y2, image_width, image_height)
    ]

    if not valid_boxes:
        return "???-????"

    # Calculate the mid-line
    y_centers = [(y1 + y2) / 2 for _, y1, _, y2, _, _ in valid_boxes]
    if not y_centers:
        return "???-????"
    mid_y = np.mean(y_centers)

    # Split into first row (above mid-line) and second row (below mid-line)
    first_row = [box for box in valid_boxes if (box[1] + box[3]) / 2 < mid_y]
    second_row = [box for box in valid_boxes if (box[1] + box[3]) / 2 >= mid_y]

    # Sort rows by x-coordinates
    first_row.sort(key=lambda b: b[0])  # Sort by x1
    second_row.sort(key=lambda b: b[0])  # Sort by x1

    # Extract parts of the license plate
    # First row: alphabets (large text) and optional small digits
    large_text_first_row = "".join(
        class_id_to_char(int(box[5])) for box in first_row if class_id_to_char(int(box[5])).isalpha()
    ).ljust(3, "?")[:3]

    small_digit_text_first_row = "".join(
        class_id_to_char(int(box[5])) for box in first_row if class_id_to_char(int(box[5])).isdigit()
    )
    if len(small_digit_text_first_row) < 2:
        small_digit_text_first_row = small_digit_text_first_row.ljust(2, "?")
    elif len(small_digit_text_first_row) > 2:
        small_digit_text_first_row = small_digit_text_first_row[:2]

    # Second row: digits (large digit text) and optional small letter
    large_digit_text_second_row = "".join(
        class_id_to_char(int(box[5])) for box in second_row if class_id_to_char(int(box[5])).isdigit()
    )
    if len(large_digit_text_second_row) < 3:  # If fewer than 3 digits, pad with ?
        large_digit_text_second_row = large_digit_text_second_row.ljust(3, "?")
    elif len(large_digit_text_second_row) > 4:  # Limit to maximum of 4 digits
        large_digit_text_second_row = large_digit_text_second_row[:4]

    small_letter_text_second_row = "".join(
        class_id_to_char(int(box[5])) for box in second_row if class_id_to_char(int(box[5])).isalpha()
    )
    small_letter_text_second_row = small_letter_text_second_row[:1] if small_letter_text_second_row else ""

    # Combine parts to form the license plate text
    license_plate = f"{large_text_first_row}-{large_digit_text_second_row}"
    if small_digit_text_first_row.strip("?"):
        license_plate += f"-{small_digit_text_first_row}"
    if small_letter_text_second_row:
        license_plate += f"-{small_letter_text_second_row}"

    return license_plate



def extract_single_line_license_plate_text(bounding_boxes, image_height, image_width):
    """
    Extracts license plate text from bounding boxes for a single-line plate based on the provided rules,
    ensuring safe handling of empty lists and resolving overlapping boxes.
    """
    import numpy as np

    def center_x(box):
        return (box[0] + box[2]) / 2

    def class_id_to_char(class_id):
        if 0 <= class_id <= 25:  # A-Z
            return chr(class_id + ord('A'))
        elif 26 <= class_id <= 35:  # 0-9
            return chr(class_id - 26 + ord('0'))
        return "?"

    valid_boxes = [
        (x1, y1, x2, y2, (x2 - x1) * (y2 - y1), y2 - y1, confidence, class_id)
        for x1, y1, x2, y2, confidence, class_id in bounding_boxes
        if not is_too_close_to_corner(x1, y1, x2, y2, image_width, image_height)
    ]

    if not valid_boxes:
        return "???-????"

    avg_height = np.mean([box[5] for box in valid_boxes])

    # Compare largest and smallest bounding box areas
    largest_area = max(box[4] for box in valid_boxes)
    smallest_area = min(box[4] for box in valid_boxes)

    if smallest_area >= 0.5 * largest_area:
        # No small boxes, treat all as large
        large_boxes = valid_boxes
        small_boxes = []
    else:
        # Separate into large and small boxes
        large_boxes = [box for box in valid_boxes if box[5] >= avg_height]
        small_boxes = [box for box in valid_boxes if box[5] < avg_height]

    large_letters = [box for box in large_boxes if class_id_to_char(int(box[7])).isalpha()]
    large_digits = [box for box in large_boxes if class_id_to_char(int(box[7])).isdigit()]
    small_digits = [box for box in small_boxes if class_id_to_char(int(box[7])).isdigit()]
    small_letters = [box for box in small_boxes if class_id_to_char(int(box[7])).isalpha()]

    large_letters.sort(key=lambda b: center_x(b))
    large_digits.sort(key=lambda b: center_x(b))
    small_digits.sort(key=lambda b: center_x(b))
    small_letters.sort(key=lambda b: center_x(b))

    # Large letters (mandatory)
    large_text = "".join(class_id_to_char(int(box[7])) for box in large_letters[:3]).rjust(3, "?")

    # Large digits (mandatory, 3 or 4 digits)
    large_digit_text = "".join(class_id_to_char(int(box[7])) for box in large_digits)
    if len(large_digit_text) < 3:
        large_digit_text = large_digit_text.ljust(3, "?")
    elif len(large_digit_text) > 4:
        large_digit_text = large_digit_text[:4]

    # Small digits (optional, up to 2 digits)
    small_digit_text = "".join(class_id_to_char(int(box[7])) for box in small_digits[:2])
    if len(small_digit_text) < 2:
        small_digit_text = small_digit_text.ljust(2, "?")

    # Small letters (optional, single letter)
    small_letter_text = class_id_to_char(int(small_letters[0][7])) if small_letters else ""

    # Combine the license plate parts
    license_plate_parts = [large_text, large_digit_text]
    if small_digit_text.strip("?"):  # Only add if there are valid small digits
        license_plate_parts.append(small_digit_text)
    if small_letter_text:  # Only add if there is a valid small letter
        license_plate_parts.append(small_letter_text)

    license_plate = "-".join(license_plate_parts)

    return license_plate

# Define the helper function to check if a bounding box touches the image border.
def touches_border(x1, y1, x2, y2, image_width, image_height, threshold=0):
    """
    Returns True if any side of the bounding box touches the border of the image.
    :param x1: left coordinate
    :param y1: top coordinate
    :param x2: right coordinate
    :param y2: bottom coordinate
    :param image_width: width of the image
    :param image_height: height of the image
    :param threshold: allowable margin from the border (default 0 means exact border)
    """
    return (x1 <= threshold or y1 <= threshold or 
            x2 >= image_width - threshold or y2 >= image_height - threshold)

def is_double_line_license_plate(bounding_boxes):
    """
    Determines if the bounding boxes belong to a double-line plate based on the variation 
    in the y-coordinate of the center points, normalized by the average bounding box height.
    """
    if not bounding_boxes:
        return False

    # Calculate the center y-coordinates and heights of each bounding box
    centers_y = [(y1 + y2) / 2 for x1, y1, x2, y2, _, _ in bounding_boxes]
    heights = [y2 - y1 for _, y1, _, y2, _, _ in bounding_boxes]
    
    # Calculate the standard deviation of the y-coordinates
    y_variation = np.std(centers_y)
    
    # Calculate the average height of the bounding boxes
    avg_height = np.mean(heights)
    
    # Avoid division by zero
    if avg_height == 0:
        return False

    # Normalize the variation by the average height
    normalized_variation = y_variation / avg_height

    # Define a threshold for normalized variation to classify as double-line
    threshold = 0.4  # Adjust this based on your dataset
    print(f"normalized: {normalized_variation}")
    return normalized_variation > threshold

def rotate_image(image, angle):
    """
    Rotates an image (numpy array) by the given angle (in degrees).
    Positive angles mean counter-clockwise rotation.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



# -------------------------------------------------------------------------------
# The function to process license plate images and return the best reading.
# -------------------------------------------------------------------------------
def license_plate_img_to_text(image_paths):
    """
    For each image in image_paths, process three variations (original, rotated 10° anticlockwise,
    and rotated 10° clockwise), obtain a license plate reading for each, and return the reading
    that best matches the regex pattern (or the one with the highest average detection confidence if none match).
    """
    results_texts = []

    # Compile the regex for a valid license plate reading.
    regex_pattern = re.compile(r'^[A-Z]{3}-\d{3,4}(-\d{2})?(-[A-Z])?$')

    for image_path in image_paths:
        # Skip images that have "no_license_plate" in their filename.
        if "no_license_plate" in os.path.basename(image_path):
            results_texts.append("can_not_read")
            continue

        try:
            # Open and resize the image.
            pil_image = Image.open(image_path)
            original_width, original_height = pil_image.size
            target_width = 1000
            target_height = int(original_height * (target_width / original_width))
            resized_pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
            original_image = cv2.cvtColor(np.array(resized_pil_image), cv2.COLOR_RGB2BGR)

            # Create three variations.
            variations = {
                "original": original_image.copy(),
                "rotated_10": rotate_image(original_image.copy(), 10),    # 10° anticlockwise
                "rotated_-10": rotate_image(original_image.copy(), -10)     # 10° clockwise
            }

            results_variations = {}

            # Process each variation.
            for key, var_image in variations.items():
                # Convert the image (var_image is a NumPy array in BGR) to JPEG bytes.
                _, im_encoded = cv2.imencode('.jpg', var_image)
                im_bytes = im_encoded.tobytes()

                # LICENSE_PLATE_CHAR_API_URL = "http://127.0.0.1:8001/detect/license_plate_char"
                response = requests.post(settings.LICENSE_PLATE_CHAR_API_URL, files={'file': im_bytes}, timeout=30)
                if response.status_code != 200:
                    results_variations[key] = {'license_plate_text': "can_not_read", 'avg_conf': 0}
                    continue
                api_results = response.json()  # Expecting a list of detection dicts.
                # Convert API results into a NumPy array with columns: x1, y1, x2, y2, conf, cls.
                if len(api_results) > 0:
                    detections = np.array([
                        [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['confidence'], det['class_id']]
                        for det in api_results
                    ])
                else:
                    detections = np.empty((0,6))

                # Identify bounding boxes for class 36 (ignore these).
                class_36_boxes = [
                    (x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in detections if int(cls) == 36
                ]

                def is_inside_class_36(x, y):
                    """Check if a coordinate is inside any class 36 bounding box."""
                    for x1, y1, x2, y2 in class_36_boxes:
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            return True
                    return False

                # Filter detections: ignore class 36 and boxes that are too close to corners,
                # inside a class 36 box, or touching the border.
                filtered_detections = [
                    (x1, y1, x2, y2, conf, cls)
                    for x1, y1, x2, y2, conf, cls in detections
                    if int(cls) != 36
                    and not is_too_close_to_corner(x1, y1, x2, y2, target_width, target_height)
                    and not is_inside_class_36((x1 + x2) / 2, (y1 + y2) / 2)
                    and not touches_border(x1, y1, x2, y2, target_width, target_height)
                ]

                if not filtered_detections:
                    results_variations[key] = {
                        'license_plate_text': "can_not_read",
                        'avg_conf': 0
                    }
                    continue

                # Determine whether the plate is double-line or single-line.
                if is_double_line_license_plate(filtered_detections):
                    license_plate_text = extract_double_line_license_plate_text(filtered_detections, target_height, target_width)
                else:
                    license_plate_text = extract_single_line_license_plate_text(filtered_detections, target_height, target_width)

                # Compute the average confidence of the detections.
                confidences = [conf for (_, _, _, _, conf, _) in filtered_detections]
                avg_conf = np.mean(confidences) if confidences else 0

                results_variations[key] = {
                    'license_plate_text': license_plate_text,
                    'avg_conf': avg_conf
                }

            # Decide which variation is best.
            candidates = []
            for key, result in results_variations.items():
                text = result['license_plate_text']
                conf = result['avg_conf']
                if regex_pattern.fullmatch(text):
                    candidates.append((key, text, conf))

            if candidates:
                chosen = max(candidates, key=lambda x: x[2])
            else:
                overall = [(key, result['license_plate_text'], result['avg_conf'])
                           for key, result in results_variations.items()]
                chosen = max(overall, key=lambda x: x[2]) if overall else ("none", "can_not_read", 0)

            results_texts.append(chosen[1])

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results_texts.append("can_not_read")

    return results_texts


def save_uploaded_image(image):
    """Save the uploaded image to the media directory."""
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists

    filename = get_valid_filename(image.name)  # Sanitize filename
    image_path = os.path.join(upload_dir, filename)

    # Explicitly save the file to prevent Django from deleting it
    with open(image_path, 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)

    print(f"Image successfully saved at: {image_path}")  # Debugging line
    return image_path


# API endpoints
# LICENSE_PLATE_API_URL = "http://127.0.0.1:8001/detect/license_plate"
# MOTORCYCLE_API_URL = "http://127.0.0.1:8001/detect/motorcycle"

def detect_license_plate(cropped_image_path):
    """
    Detect and crop the license plate from the cropped violation image using API calls.
    Handles scenarios with multiple plates and motorcycles.
    """
    try:
        # Call the License Plate Detection API.
        with open(cropped_image_path, 'rb') as img_file:
            response = requests.post(settings.LICENSE_PLATE_API_URL, files={'file': img_file}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"License Plate API error: {response.text}")
        plates = response.json()  # Expecting a list of dicts
        print("License Plate API results:", plates)
        if not isinstance(plates, list):
            raise ValueError("Invalid response from License Plate API: Expected a list")
        
        license_plate_dir = os.path.join(settings.MEDIA_ROOT, 'cropped_license_plate')
        os.makedirs(license_plate_dir, exist_ok=True)
        
        # No plates detected.
        if not plates:
            return "no_license_plate"
        
        # Convert bbox values (floats) to integers.
        for plate in plates:
            plate["bbox"] = [int(round(coord)) for coord in plate["bbox"]]
        
        # If only one plate is detected, crop and return it.
        if len(plates) == 1:
            return crop_and_save(cropped_image_path, plates[0]["bbox"], license_plate_dir)
        
        # If multiple plates are detected, call the Motorcycle Detection API.
        with open(cropped_image_path, 'rb') as img_file:
            response = requests.post(settings.MOTORCYCLE_API_URL, files={'file': img_file}, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Motorcycle API error: {response.text}")
        motorcycles = response.json()  # Expecting a list of dicts
        print("Motorcycle API results:", motorcycles)
        if not isinstance(motorcycles, list):
            raise ValueError("Invalid response from Motorcycle API: Expected a list")
        
        # Convert motorcycle bbox values to integers.
        for moto in motorcycles:
            moto["bbox"] = [int(round(coord)) for coord in moto["bbox"]]
        
        # If no motorcycles detected, choose the plate with highest confidence.
        if not motorcycles:
            best_plate = max(plates, key=lambda plate: plate["confidence"])
            return crop_and_save(cropped_image_path, best_plate["bbox"], license_plate_dir)
        
        # If multiple motorcycles detected, select the one with the largest bounding box area.
        best_motorcycle = max(motorcycles, key=lambda m: calculate_area(m["bbox"]))
        return select_closest_plate_to_motorcycle(cropped_image_path, best_motorcycle, plates, license_plate_dir)
    
    except Exception as e:
        print(f"Error in detect_license_plate: {str(e)}")
        return "api_error"


def calculate_area(bbox):
    """
    Calculate the area of a bounding box given as a list [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def crop_and_save(image_path, bbox, save_dir):
    """
    Crop and save the region defined by bbox from the image at image_path.
    Expects bbox as a list [x1, y1, x2, y2] with integer values.
    """
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    cropped_object = img[y1:y2, x1:x2]
    
    filename = f'{os.path.basename(image_path).split(".")[0]}_cropped.jpg'
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, cropped_object)
    
    return save_path


def select_closest_plate_to_motorcycle(image_path, motorcycle, plates, save_dir):
    """
    Select the license plate closest to the motorcycle's center.
    Expects both motorcycle and plate detections as dictionaries with a "bbox" key
    containing a list [x1, y1, x2, y2].
    """
    motorcycle_center = calculate_center(motorcycle["bbox"])
    closest_plate = None
    min_distance = float('inf')
    
    for plate in plates:
        plate_center = calculate_center(plate["bbox"])
        distance = calculate_distance(motorcycle_center, plate_center)
        if distance < min_distance:
            min_distance = distance
            closest_plate = plate
    
    if closest_plate:
        return crop_and_save(image_path, closest_plate["bbox"], save_dir)
    
    return "no_license_plate"


def calculate_center(bbox):
    """
    Calculate the center (x, y) of a bounding box given as a list [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points (x, y).
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)



def process_detected_violations(api_results, img_annotated, img_raw, image):
    print("API Results:", api_results)

    """Process the detected violations and crop the images."""
    detected_violations = []
    cropped_image_filenames = []
    license_plate_crops = []

    cropped_images_dir = os.path.join(settings.MEDIA_ROOT, 'cropped_images')
    os.makedirs(cropped_images_dir, exist_ok=True)

    for idx, detection in enumerate(api_results):
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        class_name = detection['class_name']
        class_id = detection['class_id']

        # Call the reusable function to draw bounding boxes
        draw_bounding_box(img_annotated, x1, y1, x2, y2, class_name, confidence, class_id)

        # Store detected violation info
        detected_violations.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': confidence, 'class': class_name
        })

        # Crop the image and save it with a unique filename
        cropped = img_raw[y1:y2, x1:x2]
        filename = f"{image.name.split('.')[0]}_{idx}.jpg"
        cv2.imwrite(os.path.join(cropped_images_dir, filename), cropped)
        cropped_image_filenames.append(filename)

        # Use the license plate model to predict and crop license plate from the cropped image
        license_plate_crop = detect_license_plate(os.path.join(cropped_images_dir, filename))
        license_plate_crops.append(license_plate_crop)
    
    upscaled_license_plate_crops = upscale_images(license_plate_crops)
    
    return detected_violations, cropped_image_filenames, upscaled_license_plate_crops


def draw_bounding_box(image, x1, y1, x2, y2, class_name, confidence, class_id):
    """Draw bounding box and label on the image."""
    box_color = (0, 255, 0) if class_id in [1, 2] else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(image, f"{class_name}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

def crop_and_save_image(img_raw, x1, y1, x2, y2, image, idx, cropped_images_dir): 
    """Crop the image and save it with a unique filename without resizing."""
    # Calculate padded coordinates for cropping
    x1_padded, y1_padded = max(x1 - 40, 0), max(y1 - 40, 0)
    x2_padded, y2_padded = min(x2 + 40, img_raw.shape[1]), min(y2 + 40, img_raw.shape[0])
    
    # Crop the image
    cropped_image = img_raw[y1_padded:y2_padded, x1_padded:x2_padded]

    # Generate a unique filename for the cropped image
    unique_str = f"{x1}_{y1}_{x2}_{y2}_{image.name}_{idx}"
    cropped_image_filename = hashlib.md5(unique_str.encode()).hexdigest() + ".jpg"
    cropped_image_path = os.path.join(cropped_images_dir, cropped_image_filename)

    # Save the cropped image without resizing
    cv2.imwrite(cropped_image_path, cropped_image)

    return cropped_image_filename

def save_annotated_image(img_annotated, image_name):
    """Save the annotated image."""
    annotated_image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'annotated_' + image_name)
    cv2.imwrite(annotated_image_path, img_annotated)
    return annotated_image_path


def upscale_images(license_plate_crops, target_width=1000):
    """
    Upscale license plate crop images to a specified width while maintaining the aspect ratio.

    Args:
        license_plate_crops (list): List of URLs or file paths to the license plate crops.
        target_width (int): Desired width for the upscaled images.

    Returns:
        list: List of file paths to the upscaled images.
    """
    upscaled_images = []

    for idx, crop_url in enumerate(license_plate_crops):
        # Extract the basename from the URL (filename of the image)
        crop_basename = os.path.basename(urlparse(crop_url).path)

        # Check if the basename is equal to "no_license_plate"
        if crop_basename == "no_license_plate":
            # Append the original crop URL to the list without upscaling
            upscaled_images.append(crop_url)
        else:
            # Open the image using Pillow
            with Image.open(crop_url) as img:
                # Calculate the new height to maintain aspect ratio
                aspect_ratio = img.height / img.width
                target_height = int(target_width * aspect_ratio)

                # Resize the image with anti-aliasing
                upscaled_img = img.resize((target_width, target_height), Image.LANCZOS)

                # Overwrite the original image
                upscaled_img.save(crop_url)

                # Append the updated image path to the list
                upscaled_images.append(crop_url)

    return upscaled_images

# Set the device for model inference (CUDA or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Real-ESRGAN model once
# model_realESRGAN = RealESRGAN(device, scale=4)
# model_realESRGAN.load_weights('weights/RealESRGAN_x4.pth', download=True)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_license_plate_bike_front(
    ler_text="LER", 
    number_text="1234", 
    city_text="PUNJAB", 
    manufacture_year="", 
    single_text=""
):
    '''
    Create a license plate image for a motorcycle.'''
    # Define paths
    output_dir = os.path.join(settings.MEDIA_ROOT, "generated_license_plates")
    flower_image_path = os.path.join(settings.STATICFILES_DIRS[0], "motorcycle_violation/images/flower.png")
    font_dir = os.path.join(settings.STATICFILES_DIRS[0], "motorcycle_violation/fonts")
    # Print the paths
    # Ensure directories exist
    ensure_directory_exists(output_dir)

    # Generate the output file name
    output_file = os.path.join(output_dir, f"{ler_text}{number_text}{manufacture_year}{single_text}.jpg")

    # Load fonts
    eurostile_large = ImageFont.truetype(os.path.join(font_dir, "Sauerkrauto.ttf"), 200)  # For "LER" and "1234"
    eurostile_medium = ImageFont.truetype(os.path.join(font_dir, "Sauerkrauto.ttf"), 100)  # For "20"
    eurostile_small = ImageFont.truetype(os.path.join(font_dir, "Sauerkrauto.ttf"), 75)  # For "20"
    punjab_font = ImageFont.truetype("arial.ttf", 25)  # For "PUNJAB"

    # Create a blank white image (outside the border)
    width, height = 1010, 185
    plate = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(plate)

    # Draw the green strip with padding of 5 pixels (white space before the green)
    green_color = (0, 177, 91)
    green_strip_width = 140
    horizontal_padding = 8  # Padding between green strip and the rest of the plate
    vertical_padding = 10  # Padding from the top and bottom
    draw.rectangle(
        [
            horizontal_padding, 
            vertical_padding, 
            green_strip_width + horizontal_padding, 
            height - vertical_padding
        ], 
        fill=green_color
    )

    # Add text for the city (default "PUNJAB")
    draw.text((30, 140), city_text, fill="black", font=punjab_font)

    # Add flower image (wheat graphic) while maintaining the aspect ratio
    flower_image = Image.open(flower_image_path)
    flower_image = flower_image.resize((120, 120))
    plate.paste(flower_image, (20, 20), flower_image)  # Position the image on the plate

    # Add "LER-" (default "LER" with the hyphen)
    x_pos = 155
    for char in ler_text:
        draw.text((x_pos, -45), char, fill="black", font=eurostile_large)
        x_pos += eurostile_large.getlength(char)

    # Add the manufacture year (e.g., "20")
    x_pos = 900  # Position for the manufacture year "20"
    for char in manufacture_year:
        if char != " ":  # Skip if the character is a space
            draw.text((x_pos, -5), char, fill="black", font=eurostile_small)
            x_pos += eurostile_small.getlength(char)

    # Add the customizable number text (default "1234")
    x_pos = 480
    for char in number_text:
        if char != " ": 
            draw.text((x_pos, -45), char, fill="black", font=eurostile_large)
            x_pos += eurostile_large.getlength(char)

    # Add the single text (e.g., "A")
    x_pos = 920  # Position for the single text "A"
    for char in single_text:
        if char != " ":  # Skip if the character is a space
            draw.text((x_pos, 60), char, fill="black", font=eurostile_medium)
            x_pos += eurostile_small.getlength(char)

    # Draw the black rounded border on top of everything
    border_thickness = 5
    radius = 20  # Radius for rounded corners
    draw.rounded_rectangle(
        [
            border_thickness, 
            border_thickness, 
            width - border_thickness, 
            height - border_thickness
        ],
        radius=radius,
        outline="black",
        width=border_thickness
    )

    # Resize the image to the new dimensions (462x389)
    new_width, new_height = 960, 320
    plate_resized = plate.resize((new_width, new_height))

    # Save the resized license plate image
    plate_resized.save(output_file)
