# step - 03

import cv2
import numpy as np
import pandas as pd
import os
import tempfile
from ultralytics import YOLO
from .read import license_plate_img_to_text  # Import function for LP reading
import ast

def safe_eval(value):
    """Safely evaluates a string representation of a list, or returns None if invalid."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return value

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is expected in the format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def process_tracking_results(input_df, input_csv, video_source, violation_model_path, lp_model_path, lp_char_model_path, output_csv):
    """
    Processes a video to detect motorcycle violations and license plates using YOLO models.

    Parameters:
    - input_csv: Path to the input CSV file containing tracking data.
    - video_source: Path to the input video file.
    - violation_model_path: Path to the motorcycle violation detection YOLO model.
    - lp_model_path: Path to the license plate detection YOLO model.
    - lp_char_model_path: Path to the license plate character recognition YOLO model.
    - output_csv: Path where the final results CSV will be saved.

    Returns:
    - new_df: DataFrame containing the processed tracking results.
    """

    # Load models
    violation_model = YOLO(violation_model_path)
    lp_model = YOLO(lp_model_path)  # NEW LP DETECTION MODEL
    lp_char_model = YOLO(lp_char_model_path)

    # Load CSV (or use the provided DataFrame)
    df = input_df

    # Open video file
    cap = cv2.VideoCapture(video_source)

    # Prepare results list
    results_list = []

    # Process each frame from CSV
    for _, row in df.iterrows():
        frame_id = int(row["frame"])
        track_id = int(row["track_id"])

        # Extract motorcycle bounding box (global coordinates)
        motorcycle_bbox = safe_eval(row["motorcycle_bbox"])  # Convert string to list
        if motorcycle_bbox is None or len(motorcycle_bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, motorcycle_bbox)

        # Set video position to required frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame not found

        # Get frame dimensions from the read frame
        frame_height, frame_width = frame.shape[:2]

        # --- Step 1: Violation Detection ---
        # Try three different expansion settings sequentially.
        expansion_settings = [
            (0.35, 1.5, 0.35),   # Initial expansion
            (0.2, 1.0, 0.2),     # First alternative
            (0.20, 0.75, 0.20)   # Second alternative
        ]
        selected_violation_data = None

        for (w_factor, h_above_factor, h_below_factor) in expansion_settings:
            width_increase = int((x2 - x1) * w_factor)
            height_increase_above = int((y2 - y1) * h_above_factor)
            height_increase_below = int((y2 - y1) * h_below_factor)

            new_x1 = max(0, x1 - width_increase)
            new_x2 = min(frame_width, x2 + width_increase)
            new_y1 = max(0, y1 - height_increase_above)
            new_y2 = min(frame_height, y2 + height_increase_below)

            # Crop the expanded region
            motorcycle_crop = frame[new_y1:new_y2, new_x1:new_x2]

            # Run violation detection on the cropped region
            violation_results = violation_model.predict(motorcycle_crop, conf=0.3, iou=0.5)
            best_violation = None
            best_iou = 0.0
            best_candidate_bbox = None

            if len(violation_results) > 0 and len(violation_results[0].boxes) > 0:
                for box in violation_results[0].boxes:
                    # Get box coordinates in cropped coordinates
                    cropped_coords = box.xyxy[0].cpu().numpy()
                    cropped_vx1, cropped_vy1, cropped_vx2, cropped_vy2 = cropped_coords
                    # Convert to global coordinates by adding crop offsets
                    vx1 = int(cropped_vx1 + new_x1)
                    vy1 = int(cropped_vy1 + new_y1)
                    vx2 = int(cropped_vx2 + new_x1)
                    vy2 = int(cropped_vy2 + new_y1)
                    candidate_bbox = [vx1, vy1, vx2, vy2]
                    # Compute IoU between candidate violation bbox and the original motorcycle_bbox
                    iou = compute_iou(candidate_bbox, [x1, y1, x2, y2])
                    if iou > best_iou:
                        best_iou = iou
                        best_violation = box
                        best_candidate_bbox = candidate_bbox

                # Only if there is some overlap (iou > 0) consider it a valid violation detection.
                if best_iou > 0:
                    selected_violation_data = {
                        "violation_class_id": int(best_violation.cls.item()),
                        "violation_conf": float(best_violation.conf.item()),
                        "violation_class": violation_model.names[int(best_violation.cls.item())],
                        "violation_bbox": best_candidate_bbox,
                        # Also store the crop offsets (for use in LP detection)
                        "crop_offsets": (new_x1, new_y1)
                    }
                    break  # Use the first expansion that provides an overlapping violation

        # If no valid violation is found, set values to None.
        if selected_violation_data is not None:
            violation_class = selected_violation_data["violation_class"]
            violation_class_id = selected_violation_data["violation_class_id"]
            violation_bbox = selected_violation_data["violation_bbox"]
            violation_conf = selected_violation_data["violation_conf"]
            crop_offsets = selected_violation_data["crop_offsets"]
        else:
            violation_class = None
            violation_class_id = None
            violation_bbox = None
            violation_conf = None
            # If no violation was detected, use the crop from the last expansion attempt.
            # (This will be the crop computed with the last settings in expansion_settings.)
            width_increase = int((x2 - x1) * expansion_settings[-1][0])
            height_increase_above = int((y2 - y1) * expansion_settings[-1][1])
            height_increase_below = int((y2 - y1) * expansion_settings[-1][2])
            crop_offsets = (
                max(0, x1 - width_increase),
                max(0, y1 - height_increase_above)
            )
            # Also update new_x1, new_y1 for LP detection:
            new_x1 = max(0, x1 - width_increase)
            new_y1 = max(0, y1 - height_increase_above)

        # --- Step 2: License Plate Detection ---
        # Use the same expanded region ("motorcycle_crop") from the last attempted expansion.
        # (If a violation was detected, this is the crop from that expansion;
        #  otherwise, it is the crop from the final (third) expansion attempt.)
        motorcycle_crop = frame[new_y1:new_y2, new_x1:new_x2]

        lp_results = lp_model.predict(motorcycle_crop, conf=0.3, iou=0.5)
        best_lp_bbox, new_lp_text, new_lp_avg_conf = None, "", 0.0

        if len(lp_results) > 0 and len(lp_results[0].boxes) > 0:
            # Compute motorcycle center relative to the crop
            motorcycle_center_x = ((x2 + x1) // 2) - new_x1
            motorcycle_center_y = ((y2 + y1) // 2) - new_y1

            # Find the LP closest to the motorcycle center (in crop coordinates)
            min_distance = float("inf")
            for i in range(len(lp_results[0].boxes)):
                lp_x1, lp_y1, lp_x2, lp_y2 = lp_results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                lp_center_x = (lp_x1 + lp_x2) // 2
                lp_center_y = (lp_y1 + lp_y2) // 2
                distance = np.sqrt((motorcycle_center_x - lp_center_x) ** 2 + (motorcycle_center_y - lp_center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_lp_bbox = (lp_x1, lp_y1, lp_x2, lp_y2)

            if best_lp_bbox:
                lp_x1, lp_y1, lp_x2, lp_y2 = best_lp_bbox
                lp_crop = frame[new_y1 + lp_y1 : new_y1 + lp_y2, new_x1 + lp_x1 : new_x1 + lp_x2]

                if lp_crop.shape[0] > 5 and lp_crop.shape[1] > 5:
                    temp_lp_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
                    cv2.imwrite(temp_lp_path, lp_crop)

                    if os.path.exists(temp_lp_path) and os.path.getsize(temp_lp_path) > 0:
                        try:
                            lp_result = license_plate_img_to_text([temp_lp_path], lp_char_model, output_dir=None)[0]
                            new_lp_text = lp_result.get('lp_text', '')
                            new_lp_avg_conf = float(lp_result.get('avg_conf', 0))
                        except Exception as e:
                            print(f"Error in LP recognition: {e}")
                            new_lp_text, new_lp_avg_conf = "", 0.0

                    os.remove(temp_lp_path)

                # Convert LP bbox to global coordinates
                global_lp_x1 = new_x1 + lp_x1
                global_lp_y1 = new_y1 + lp_y1
                global_lp_x2 = new_x1 + lp_x2
                global_lp_y2 = new_y1 + lp_y2
                best_lp_bbox = [global_lp_x1, global_lp_y1, global_lp_x2, global_lp_y2]

        # Append new data to row
        row["violation_class"] = violation_class
        row["violation_class_id"] = violation_class_id
        row["violation_bbox"] = violation_bbox
        row["violation_conf"] = violation_conf
        row["new_lp_bbox"] = best_lp_bbox
        row["new_lp_text"] = new_lp_text
        row["new_lp_avg_conf"] = new_lp_avg_conf
        results_list.append(row)

    # Release resources
    cap.release()

    # Convert results to DataFrame and save CSV
    new_df = pd.DataFrame(results_list)
    new_df.to_csv(output_csv, index=False)

    print(f"Final results saved to {output_csv}")
    return new_df



# Example usage:
# processed_data = process_tracking_results("best_tracking_results_hd3.csv", "E:/syn_data/track/video/hd3.mp4", 
#                                           "E:/syn_data/track/yolo_models/motorcycle_violation_detection_n10__normal.pt", 
#                                           "E:/syn_data/track/yolo_models/license_plate_detection_m.pt", 
#                                           "E:/syn_data/track/yolo_models/license_plate_char_detetction_m.pt", 
#                                           "final_tracking_results_hd3.csv")
