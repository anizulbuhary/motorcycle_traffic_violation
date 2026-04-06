import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import re

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
    large_text = "".join(class_id_to_char(int(box[7])) for box in large_letters[:3]).ljust(3, "?")

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
    threshold = 0.5  # Adjust this based on your dataset
    return normalized_variation > threshold

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

# ------------------------------------------------------------------------------
# Function to rotate an image by a given angle (in degrees)
# ------------------------------------------------------------------------------
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# ------------------------------------------------------------------------------
# Process one variation: run detection & annotate image.
# Returns:
#   license_plate_text, plate_type, average_confidence, annotated_image
# ------------------------------------------------------------------------------
def process_variation(image, model, target_width, target_height):
    # Run model prediction on the image
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()

    # Identify bounding boxes for class 36 (which we want to ignore)
    class_36_boxes = [
        (x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in detections if int(cls) == 36
    ]

    def is_inside_class_36(x, y):
        """Check if a coordinate is inside any class 36 bounding box."""
        for x1, y1, x2, y2 in class_36_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    # Filter detections:
    filtered_detections = [
        (x1, y1, x2, y2, conf, cls)
        for x1, y1, x2, y2, conf, cls in detections
        if int(cls) != 36  # ignore class 36
           and not is_too_close_to_corner(x1, y1, x2, y2, target_width, target_height)
           and not is_inside_class_36((x1 + x2) / 2, (y1 + y2) / 2)
           and not touches_border(x1, y1, x2, y2, target_width, target_height)
    ]

    # If no valid detections, return empty reading and 0 confidence.
    if not filtered_detections:
        return "", "unknown", 0, image

    # Determine key vertical coordinates from filtered detections
    top_left_y = min(box[1] for box in filtered_detections)
    bottom_right_y = max(box[3] for box in filtered_detections)

    # Mark key points on the image for debugging/visualization
    cv2.circle(image, (int(target_width // 2), int(top_left_y)), 5, (0, 0, 255), -1)
    cv2.circle(image, (int(target_width // 2), int(bottom_right_y)), 5, (255, 0, 0), -1)
    mid_y = (top_left_y + bottom_right_y) / 2
    cv2.line(image, (0, int(mid_y)), (target_width, int(mid_y)), (0, 255, 255), 2)

    # Decide on single- or double-line license plate based on your function
    if is_double_line_license_plate(filtered_detections):
        license_plate_text = extract_double_line_license_plate_text(filtered_detections, target_height, target_width)
        plate_type = "double"
    else:
        license_plate_text = extract_single_line_license_plate_text(filtered_detections, target_height, target_width)
        plate_type = "single"

    # Draw each detection and record confidences for averaging.
    confidences = []
    for (x1, y1, x2, y2, conf, cls) in filtered_detections:
        confidences.append(conf)
        class_name = model.names[int(cls)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, class_name, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (int(x1), int(y2) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Draw a red circle at the center of the detection
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

    avg_conf = np.mean(confidences) if confidences else 0

    # Annotate license plate text and plate type on the image.
    cv2.putText(image, license_plate_text, (10, target_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, plate_type, (target_width - 150, target_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return license_plate_text, plate_type, avg_conf, image

def license_plate_img_to_text(image_paths, model, output_dir=None):
    """
    Process a list of image paths containing license plates and return the license plate readings.
    
    For each image, this function:
      1. Opens and resizes the image (to a fixed target width of 1000 pixels).
      2. Processes the original image using the provided YOLO model (via process_variation)
         to obtain:
         - The license plate text
         - The plate type (e.g., "single" or "double")
         - The average confidence of the detections
         - An annotated image (with drawn boxes/markers)
      3. Saves the annotated image to output_dir if provided.
    
    Parameters:
      image_paths (list): List of file paths to the images.
      model: YOLO model instance used for LP reading.
      output_dir (str): Optional directory to save annotated images. If None, images are not saved.
    
    Returns:
      results (list): A list of dictionaries, one per processed image, containing:
          - image_name: The filename of the image.
          - lp_text: The license plate text reading.
          - plate_type: The plate type as returned by process_variation.
          - avg_conf: The average confidence for the detection.
    """
    # If an output directory is provided, ensure it exists.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compile the regex pattern for a valid license plate (you can adjust it as needed)
    regex_pattern = re.compile(r'^[A-Z]{3}-\d{3,4}(-\d{2})?(-[A-Z])?$')
    
    results = []
    target_width = 1000  # Fixed target width for all images
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            continue

        print(f"Processing image: {image_name}")
        try:
            # Open the image with PIL and resize it.
            pil_image = Image.open(image_path)
            original_width, original_height = pil_image.size
            target_height = int(original_height * (target_width / original_width))
            resized_pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
            original_image = cv2.cvtColor(np.array(resized_pil_image), cv2.COLOR_RGB2BGR)
            
            # Process the original image using process_variation.
            lp_text, plate_type, avg_conf, annotated_image = process_variation(
                original_image.copy(), model, target_width, target_height
            )
            
            if output_dir:
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, annotated_image)
                print(f"Saved annotated image to: {output_path}")
            
            print(f"LP Text: {lp_text}, Plate Type: {plate_type}, Avg Conf: {avg_conf:.2f}")
            results.append({
                'image_name': image_name,
                'lp_text': lp_text,
                'plate_type': plate_type,
                'avg_conf': avg_conf
            })
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            results.append({
                'image_name': image_name,
                'lp_text': "can_not_read",
                'plate_type': None,
                'avg_conf': 0
            })
    
    return results

# ------------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from ultralytics import YOLO

    # Load your YOLO model (adjust the path as needed)
    model = YOLO(r'E:\syn_data\yolo_updated\yolo11n_training_03\weights\last.pt')
    
    # Define input directory containing LP images.
    input_dir = r'E:\syn_data\test_lps'
    # Optionally, define an output directory to save annotated images.
    output_dir = r'E:\syn_data\annotated_lps_text_n'
    
    # Get a list of image paths.
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir)
                   if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Call the function.
    lp_results = license_plate_img_to_text(image_paths, model, output_dir=output_dir)
    
    # Print the results.
    for res in lp_results:
        print(f"Image: {res['image_name']} => LP Text: {res['lp_text']}, "
              f"Plate Type: {res['plate_type']}, Avg Conf: {res['avg_conf']:.2f}")