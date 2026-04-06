import os
import pandas as pd
from django.conf import settings
import time

# Import your processing functions
from .step1_track import process_motorcycle_tracking
from .step2_best_frame import extract_best_tracking_results
from .step3_detection_frame import process_tracking_results
from .step4_filter import filter_violation_results
from .step5_best_lp import process_best_license_plate
from .step6_best_detection_frame import select_best_detections
from .step7_violation_frame import filter_violations
from .step8_save_violation_img import process_violation_data

def main_pipeline(video_path):
    """
    Runs the complete processing pipeline on the given video.
    Returns a message if no violations are found.
    """

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Set up output directories inside MEDIA_ROOT
    csv_dir = os.path.join(settings.MEDIA_ROOT, 'csv', video_id)
    output_video_dir = os.path.join(settings.MEDIA_ROOT, 'output_videos')
    # Generate a unique suffix using the current timestamp
    unique = str(int(time.time()))
    violation_images_dir = os.path.join(settings.MEDIA_ROOT, 'violation_images', f"{video_id}_{unique}")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(violation_images_dir, exist_ok=True)

    # Model Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_models_dir = os.path.join(base_dir, 'yolo_track_models')

    motorcycle_model_path = os.path.join(yolo_models_dir, "motorcycle_detection.pt")
    lp_model_path = os.path.join(yolo_models_dir, "license_plate_detection_n.pt")
    lp_char_model_path = os.path.join(yolo_models_dir, "license_plate_char_detection_nl.pt")
    violation_model_path = os.path.join(yolo_models_dir, "motorcycle_violation_detection_n10__normal.pt")
    lp_model_detection_path = os.path.join(yolo_models_dir, "license_plate_detection_m.pt")
    lp_char_detection_path = os.path.join(yolo_models_dir, "license_plate_char_detetction_m.pt")

    output_video_name = f"{video_id}_out.mp4"

    # Define CSV filenames based on the video identifier
    tracking_results_csv = os.path.join(csv_dir, f"tracking_results_{video_id}.csv")
    best_tracking_csv = os.path.join(csv_dir, f"best_tracking_results_{video_id}.csv")
    final_tracking_csv = os.path.join(csv_dir, f"final_tracking_results_{video_id}.csv")
    filtered_detection_csv = os.path.join(csv_dir, f"filtered_detection_results_{video_id}.csv")
    best_lp_csv = os.path.join(csv_dir, f"filtered_detection_results_with_best_lp_{video_id}.csv")
    best_detection_csv = os.path.join(csv_dir, f"filtered_detection_results_with_best_detection_{video_id}.csv")
    violation_results_csv = os.path.join(csv_dir, f"violation_results_{video_id}.csv")

    # Step 1: Motorcycle Tracking
    print("----- Step 1: Motorcycle Tracking -----")
    step1_df = process_motorcycle_tracking(
        motorcycle_model_path=motorcycle_model_path,
        lp_model_path=lp_model_path,
        lp_char_model_path=lp_char_model_path,
        video_source=video_path,
        output_video_dir=output_video_dir,
        output_csv_path=tracking_results_csv,
        output_video_name=output_video_name
    )
    if step1_df.empty:
        print(f"No motorcycle tracking data found for {video_id}.")
        return "No violations can be detected from this video."

    # Step 2: Extract Best Tracking Results
    print("----- Step 2: Extract Best Tracking Results -----")
    step2_df = extract_best_tracking_results(step1_df, tracking_results_csv, best_tracking_csv, is_df=True)
    if step2_df.empty:
        print(f"No valid tracking results found for {video_id}.")
        return "No violations can be detected from this video."

    # Step 3: Process Tracking Results (Violation & LP Detection)
    print("----- Step 3: Process Tracking Results -----")
    step3_df = process_tracking_results(
        step2_df,
        best_tracking_csv,
        video_path,
        violation_model_path,
        lp_model_detection_path,
        lp_char_detection_path,
        final_tracking_csv
    )
    if step3_df.empty:
        print(f"No violations or license plates detected for {video_id}.")
        return "No violations can be detected from this video."

    # Step 4: Filter Violation Results
    print("----- Step 4: Filter Violation Results -----")
    step4_df = filter_violation_results(step3_df, final_tracking_csv, filtered_detection_csv)
    if step4_df.empty:
        print(f"No filtered violations found for {video_id}.")
        return "No violations can be detected from this video."

    # Step 5: Process Best License Plate
    print("----- Step 5: Process Best License Plate -----")
    step5_df = process_best_license_plate(step4_df, filtered_detection_csv, best_lp_csv)
    if step5_df.empty:
        print(f"No license plate detected for {video_id}.")
        return "No violations can be detected from this video."

    # Step 6: Select Best Detections
    print("----- Step 6: Select Best Detections -----")
    step6_df = select_best_detections(step5_df, best_lp_csv, best_detection_csv)
    if step6_df.empty:
        print(f"No valid detections found for {video_id}.")
        return "No violations can be detected from this video."

    # Step 7: Filter Violations
    print("----- Step 7: Filter Violations -----")
    step7_df = filter_violations(step6_df, best_detection_csv, violation_results_csv)
    if step7_df.empty:
        print(f"No violations found after filtering for {video_id}.")
        return "No violations can be detected from this video."

    # Step 8: Save Violation Images
    print("----- Step 8: Save Violation Images -----")
    step8_df = process_violation_data(step7_df, violation_results_csv, video_path, violation_images_dir)
    if step7_df.empty:
        print(f"No violations found after filtering for {video_id}.")
        return "No violations can be detected from this video."

    print("Pipeline Completed Successfully!")
    
    # Return the path to the violation images directory
    return violation_images_dir, step8_df
