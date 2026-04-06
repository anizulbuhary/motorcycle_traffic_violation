# step 04
import pandas as pd
import ast
import re
import numpy as np

def filter_violation_results(input_df, input_csv, output_csv):
    """
    Filters the CSV file to retain only the rows where the license plate bounding box 
    overlaps with the violation bounding box.

    Parameters:
    - input_df: A pandas DataFrame (already loaded from the input CSV).
    - input_csv: Path to the input CSV file. (Not used for reading since input_df is provided)
    - output_csv: Path to save the filtered CSV file.

    Returns:
    - filtered_df: The filtered pandas DataFrame.
    """
    
    df = input_df

    # Helper function to clean and parse bounding box strings
    def clean_bbox_string(bbox_str):
        if not isinstance(bbox_str, str):
            return None  # If it's not a string, return None
        # Remove any 'np.int64' wrapping from the string
        cleaned_str = re.sub(r"np\.int64\(|\)", "", bbox_str)
        try:
            return ast.literal_eval(cleaned_str)
        except Exception as e:
            print(f"Error parsing bbox: {bbox_str} -> {e}")
            return None

    # Helper function to check if a bounding box is valid (has 4 numbers)
    def is_valid_bbox(bbox):
        if isinstance(bbox, (list, tuple)):
            return len(bbox) == 4
        if isinstance(bbox, np.ndarray):
            return bbox.size == 4
        if isinstance(bbox, str):
            return True  # We'll try to parse it later
        return False

    # Function to check if two rectangles overlap
    def rectangles_overlap(vx1, vy1, vx2, vy2, lp_x1, lp_y1, lp_x2, lp_y2):
        inter_width = min(vx2, lp_x2) - max(vx1, lp_x1)
        inter_height = min(vy2, lp_y2) - max(vy1, lp_y1)
        return inter_width > 0 and inter_height > 0

    # Function to check if lp_bbox overlaps with violation_bbox
    def is_lp_within_violation(row):
        vb = row["violation_bbox"]
        lpb = row["new_lp_bbox"]
        # Check validity using our helper function instead of pd.notna
        if is_valid_bbox(vb) and is_valid_bbox(lpb):
            # If values are strings, clean and parse them; otherwise, assume they are already lists/arrays.
            violation_bbox = clean_bbox_string(vb) if isinstance(vb, str) else vb
            lp_bbox = clean_bbox_string(lpb) if isinstance(lpb, str) else lpb

            if violation_bbox is None or lp_bbox is None:
                return False

            try:
                vx1, vy1, vx2, vy2 = map(int, violation_bbox)
                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_bbox)
                return rectangles_overlap(vx1, vy1, vx2, vy2, lp_x1, lp_y1, lp_x2, lp_y2)
            except Exception as e:
                print(f"Error processing row {row.get('frame', 'unknown')}: {e}")
                return False

        return False

    # Filter rows where LP bbox overlaps with violation bbox
    filtered_df = df[df.apply(is_lp_within_violation, axis=1)]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered violation results saved to {output_csv}")
    return filtered_df

# Example usage:
# filtered_data = filter_violation_results("final_tracking_results_hd3.csv", "filtered_detection_results_hd3.csv")
