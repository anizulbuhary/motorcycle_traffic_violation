import cv2
import pandas as pd
import numpy as np
import os
import tempfile
from collections import defaultdict
from ultralytics import YOLO
from .read import license_plate_img_to_text  # Import function for license plate reading
from moviepy import VideoFileClip

# Helper function to compute Intersection over Union (IoU)
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

# Conversion function using MoviePy
def convert_to_browser_friendly(input_path, output_path):
    """
    Converts a video file to a browser-friendly format using MoviePy.
    The output video is encoded with H.264 for video and AAC for audio.
    """
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        print("Conversion error:", e)

def process_motorcycle_tracking(motorcycle_model_path, lp_model_path, lp_char_model_path,
                                video_source, output_video_dir, output_csv_path, output_video_name):
    """
    Process the video for motorcycle tracking and license plate detection.
    Saves a processed video and outputs a CSV of detection results.
    Then, converts the video to a browser-friendly format.
    """
    # -------------------------------
    # Configuration & Model Loading
    # -------------------------------
    # Load YOLO models
    motorcycle_model = YOLO(motorcycle_model_path)
    lp_model = YOLO(lp_model_path)
    lp_char_model = YOLO(lp_char_model_path)

    # Video source
    cap = cv2.VideoCapture(video_source)

    # -------------------------------
    # Output Video Setup
    # -------------------------------
    os.makedirs(output_video_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_video_dir, output_video_name)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # -------------------------------
    # Data Storage
    # -------------------------------
    results_list = []  # To store detection and tracking results
    track_history = defaultdict(list)  # To store trajectories for visualization
    frame_number = 0

    # -------------------------------
    # Main Loop: Process Video Frames
    # -------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # --- Step 1: Motorcycle Detection & Tracking ---
        results = motorcycle_model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.3,
            iou=0.5
        )
        
        # If no detection, simply write the frame and continue.
        if len(results) == 0:
            cv2.imshow("Frame", frame)
            video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Get detection data from the first result
        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy()         # Bounding boxes [x1, y1, x2, y2]
        # Handle missing tracking IDs
        if det.boxes.id is None:
            track_ids = []
        else:
            track_ids = det.boxes.id.int().cpu().tolist()  # Tracking IDs
        classes = det.boxes.cls.cpu().numpy()          # Class IDs
        confidences = det.boxes.conf.cpu().numpy()       # Confidence scores

        # -------------------------------
        # Separation of Detections
        # -------------------------------
        motorcycle_indices = []
        other_vehicle_indices = []
        for i, cls in enumerate(classes):
            cls_int = int(cls)
            if cls_int == 3:
                motorcycle_indices.append(i)
            elif cls_int in [2, 5, 7]:
                other_vehicle_indices.append(i)

        # Process each motorcycle detection that does not have a high overlap with other vehicle detections.
        for i in motorcycle_indices:
            moto_box = boxes[i]
            ignore_detection = False
            for j in other_vehicle_indices:
                other_box = boxes[j]
                if compute_iou(moto_box, other_box) > 0.25:
                    ignore_detection = True
                    break
            if ignore_detection:
                continue

            # Ensure there is a track ID for this detection
            if i >= len(track_ids):
                continue
            track_id = track_ids[i]
            x1, y1, x2, y2 = moto_box.astype(int)
            motorcycle_conf = float(confidences[i])

            # Draw the motorcycle bounding box and label on the frame.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Motorcycle {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Update track history for visualization.
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            pts = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            # --- Step 2: Crop Motorcycle & Run License Plate Detection ---
            motorcycle_crop = frame[y1:y2, x1:x2]
            lp_results = lp_model.predict(motorcycle_crop, conf=0.3, iou=0.5)
            
            try:
                lp_data = lp_results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
            except Exception as e:
                lp_data = []

            if len(lp_data) > 0:
                motorcycle_center_x = (x2 - x1) // 2
                motorcycle_center_y = (y2 - y1) // 2

                best_lp = None
                min_distance = float("inf")
                for lp_det in lp_data:
                    lp_bbox = lp_det[:4]  # LP bounding box in motorcycle crop coordinates
                    lp_conf = float(lp_det[4])
                    lp_x1, lp_y1, lp_x2, lp_y2 = lp_bbox.astype(int)
                    lp_center_x = (lp_x1 + lp_x2) // 2
                    lp_center_y = (lp_y1 + lp_y2) // 2
                    distance = np.sqrt((motorcycle_center_x - lp_center_x) ** 2 +
                                       (motorcycle_center_y - lp_center_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_lp = (lp_x1, lp_y1, lp_x2, lp_y2, lp_conf)

                if best_lp:
                    lp_x1, lp_y1, lp_x2, lp_y2, lp_conf = best_lp
                    lp_crop = motorcycle_crop[lp_y1:lp_y2, lp_x1:lp_x2]

                    # --- Step 3: Obtain License Plate Text ---
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        temp_lp_path = tmp.name
                    cv2.imwrite(temp_lp_path, lp_crop)
                    lp_result = license_plate_img_to_text([temp_lp_path], lp_char_model, output_dir=None)[0]
                    os.remove(temp_lp_path)
                    
                    lp_text = lp_result.get('lp_text', '')
                    plate_type = lp_result.get('plate_type', '')
                    avg_conf_lp = float(lp_result.get('avg_conf', 0))

                    # Global LP coordinates relative to the original frame.
                    global_lp_x1 = x1 + lp_x1
                    global_lp_y1 = y1 + lp_y1
                    global_lp_x2 = x1 + lp_x2
                    global_lp_y2 = y1 + lp_y2

                    # --- Overlap Check ---
                    ignore_due_to_overlap = False
                    for j in range(len(boxes)):
                        if j == i:
                            continue
                        if int(classes[j]) not in [3, 2, 5, 7]:
                            continue
                        other_box = boxes[j].astype(int)
                        ox1, oy1, ox2, oy2 = other_box
                        iou_val = compute_iou([x1, y1, x2, y2], other_box)
                        if iou_val <= 0.25:
                            inter_x1 = max(x1, ox1)
                            inter_y1 = max(y1, oy1)
                            inter_x2 = min(x2, ox2)
                            inter_y2 = min(y2, oy2)
                            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                                if (global_lp_x1 >= inter_x1 and global_lp_y1 >= inter_y1 and
                                    global_lp_x2 <= inter_x2 and global_lp_y2 <= inter_y2):
                                    ignore_due_to_overlap = True
                                    break

                    if ignore_due_to_overlap:
                        continue

                    # Save detection result.
                    results_list.append({
                        "frame": frame_number,
                        "track_id": track_id,
                        "motorcycle_bbox": moto_box.tolist(),
                        "motorcycle_conf": motorcycle_conf,
                        "lp_bbox": [global_lp_x1, global_lp_y1, global_lp_x2, global_lp_y2],
                        "lp_conf": lp_conf,
                        "lp_text": lp_text,
                        "plate_type": plate_type,
                        "lp_avg_conf": avg_conf_lp
                    })

        cv2.imshow("Frame", frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # -------------------------------
    # Convert the Processed Video to a Browser-Friendly Format
    # -------------------------------
    # Define a new file name for the converted video.
    base, ext = os.path.splitext(output_video_name)
    converted_video_name = f"{base}_converted.mp4"
    converted_video_path = os.path.join(output_video_dir, converted_video_name)
    convert_to_browser_friendly(output_video_path, converted_video_path)
    print(f"Converted video saved to {converted_video_path}")

    return df



# # Example usage:
# df = process_motorcycle_tracking(
#     motorcycle_model_path="E:/syn_data/track/yolo_models/motorcycle_detection.pt",
#     lp_model_path="E:/syn_data/track/yolo_models/license_plate_detection_n.pt",
#     lp_char_model_path="E:/syn_data/track/yolo_models/license_plate_char_detection_nl.pt",
#     video_source="E:/syn_data/track/video/hd3.mp4",
#     output_video_dir="E:/syn_data/track/output_video"
# )
