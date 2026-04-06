# step 08
import cv2
import pandas as pd
import os
import re
import numpy as np


import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image  # For upscaling using Pillow's Lanczos filter

def process_violation_data(input_df, csv_file, video_path, output_folder):
    """
    Processes a video and extracts two types of images:
      1. An annotated violation image cropped from the frame, with the license plate (LP) intersection highlighted.
      2. A cropped and upscaled license plate image extracted from the frame based on its bounding box.
      
    The LP cropped image is upscaled using Pillow's Lanczos filter to a width of 500 pixels while maintaining the aspect ratio.
    
    Parameters:
      - input_df: A pandas DataFrame containing the violation details.
      - csv_file: Path to the CSV file (for backward compatibility; not used here).
      - video_path: Path to the input video file.
      - output_folder: Directory where extracted images will be saved.
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)
    
    # Use the provided DataFrame.
    df = input_df.copy()
    
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    
    def parse_bbox(bbox_val):
        """
        Parses a bounding box value and returns a list of four integers.
        - If bbox_val is already list-like (list, tuple, or np.ndarray) with 4 elements, it is returned as a list.
        - If bbox_val is a string, it cleans and parses it (removing any np.int64 wrappers).
        - Returns None if the value is invalid.
        """
        if isinstance(bbox_val, (list, tuple, np.ndarray)):
            arr = np.array(bbox_val)
            if arr.size == 4:
                return arr.tolist()
            return None
        
        if pd.isna(bbox_val) or not isinstance(bbox_val, str):
            return None
        
        cleaned_str = re.sub(r"np\.int64\(|\)", "", bbox_val)
        numbers = re.findall(r"\d+", cleaned_str)
        if len(numbers) == 4:
            return [int(num) for num in numbers]
        else:
            print(f"⚠️ Still Invalid BBox format after cleaning: {bbox_val}")
            return None

    for index, row in df.iterrows():
        frame_number = row["frame"]
        track_id = row["track_id"]
        
        # Set the video to the correct frame (frames are zero-indexed).
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_number} for track_id {track_id}.")
            continue
        
        # Parse and clip the violation bounding box (global coordinates).
        violation_bbox = parse_bbox(row["violation_bbox"])
        if not violation_bbox:
            print(f"⚠️ Skipping track {track_id}, frame {frame_number}: Invalid violation_bbox")
            continue
        v_x1, v_y1, v_x2, v_y2 = violation_bbox
        
        h_frame, w_frame, _ = frame.shape
        v_x1, v_y1 = max(0, v_x1), max(0, v_y1)
        v_x2, v_y2 = min(w_frame, v_x2), min(h_frame, v_y2)
        
        # Crop the violation region from the full frame.
        cropped_violation = frame[v_y1:v_y2, v_x1:v_x2].copy()
        crop_height = v_y2 - v_y1
        crop_width  = v_x2 - v_x1
        
        # Parse the license plate (LP) bounding box and text.
        best_lp_bbox = parse_bbox(row["best_lp_bbox"])
        best_lp_text = str(row["best_lp_text"]) if pd.notna(row["best_lp_text"]) else ""
        
        intersect_coords = None  # Coordinates of the LP region that intersects with the violation crop.
        
        if best_lp_bbox:
            lp_x1, lp_y1, lp_x2, lp_y2 = best_lp_bbox
            print(f"🔍 Processing Track {track_id}, Frame {frame_number}")
            print(f"   LP BBox (Frame): {best_lp_bbox}")
            print(f"   Violation BBox (Frame): {violation_bbox}")
            
            # Compute the intersection between the LP and violation bounding boxes.
            inter_x1 = max(lp_x1, v_x1)
            inter_y1 = max(lp_y1, v_y1)
            inter_x2 = min(lp_x2, v_x2)
            inter_y2 = min(lp_y2, v_y2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                # Adjust coordinates relative to the cropped violation image.
                adj_lp_x1 = inter_x1 - v_x1
                adj_lp_y1 = inter_y1 - v_y1
                adj_lp_x2 = inter_x2 - v_x1
                adj_lp_y2 = inter_y2 - v_y1
                intersect_coords = [adj_lp_x1, adj_lp_y1, adj_lp_x2, adj_lp_y2]
                print(f"✅ Intersection BBox (Crop): {intersect_coords}")
            else:
                print(f"❌ No valid intersection for LP bbox on track {track_id}, frame {frame_number}")
        
        # Resize the cropped violation image to a fixed target height while preserving the aspect ratio.
        target_height = 2500  # Modify as needed.
        scale = target_height / crop_height
        target_width = int(crop_width * scale)
        resized_cropped = cv2.resize(cropped_violation, (target_width, target_height))
        
        # Annotate the LP intersection on the resized image using fixed font scale.
        if intersect_coords is not None:
            r_adj_lp_x1 = int(intersect_coords[0] * scale)
            r_adj_lp_y1 = int(intersect_coords[1] * scale)
            r_adj_lp_x2 = int(intersect_coords[2] * scale)
            r_adj_lp_y2 = int(intersect_coords[3] * scale)
            
            cv2.rectangle(resized_cropped, (r_adj_lp_x1, r_adj_lp_y1),
                          (r_adj_lp_x2, r_adj_lp_y2), (0, 255, 0), 2)
            # Uncomment below to overlay the LP text on the violation crop if desired.
            # if best_lp_text.strip():
            #     text_position = (r_adj_lp_x1, max(30, r_adj_lp_y1 - 10))
            #     cv2.putText(resized_cropped, best_lp_text, text_position,
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
        # (Optional) Draw violation class text on the resized image.
        # violation_class = str(row["violation_class"])
        # cv2.putText(resized_cropped, violation_class, (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        
        # Save the annotated violation image.
        output_filename = os.path.join(output_folder, f"track_{track_id}_frame_{frame_number}.jpg")
        cv2.imwrite(output_filename, resized_cropped)
        print(f"✅ Saved violation image: {output_filename}")
        
        # ----- Save and Upscale the License Plate (LP) Cropped Image -----
        if best_lp_bbox:
            # Clip the LP bounding box to the frame boundaries.
            lp_x1 = max(0, best_lp_bbox[0])
            lp_y1 = max(0, best_lp_bbox[1])
            lp_x2 = min(w_frame, best_lp_bbox[2])
            lp_y2 = min(h_frame, best_lp_bbox[3])
            
            if lp_x2 > lp_x1 and lp_y2 > lp_y1:
                lp_crop = frame[lp_y1:lp_y2, lp_x1:lp_x2].copy()
                
                # Convert the LP crop from BGR to RGB for Pillow.
                lp_crop_rgb = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2RGB)
                lp_pil = Image.fromarray(lp_crop_rgb)
                
                # Upscale the LP crop to a width of 500px while preserving aspect ratio using Lanczos filter.
                new_width = 500
                orig_width, orig_height = lp_pil.size
                scale_factor = new_width / float(orig_width)
                new_height = int(orig_height * scale_factor)
                lp_upscaled = lp_pil.resize((new_width, new_height), Image.LANCZOS)
                
                lp_output_filename = os.path.join(output_folder, f"track_{track_id}_frame_{frame_number}_lp.jpg")
                lp_upscaled.save(lp_output_filename)
                print(f"✅ Saved upscaled LP cropped image: {lp_output_filename}")
            else:
                print(f"⚠️ Skipping LP crop for track {track_id}, frame {frame_number}: Invalid LP bbox after clipping.")
    
    cap.release()
    print(f"✅ All images saved in: {output_folder}")
    
    return df


# Example usage:
# process_violation_data(input_df, csv_file, video_path, output_folder)

# Example usage:
# process_violation_data("violation_results_hd3.csv", r"E:\syn_data\track\video\hd3.mp4", "hd3_violation_images")
