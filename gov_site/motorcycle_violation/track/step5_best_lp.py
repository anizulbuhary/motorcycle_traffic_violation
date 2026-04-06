# step 05
import pandas as pd
import ast
import re
import numpy as np

def safe_parse_bbox(value):
    """
    Safely parses a bounding box value.
    - If the value is a string, it cleans any 'np.int64(...)' wrapping and uses ast.literal_eval.
    - If the value is already a list, tuple, or a numpy array of length 4, it returns it.
    - Otherwise, returns an empty string.
    """
    if isinstance(value, str):
        try:
            # Remove any 'np.int64(' and trailing ')' from the string
            cleaned_str = re.sub(r"np\.int64\(|\)", "", value)
            bbox = ast.literal_eval(cleaned_str)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return bbox
            else:
                return ""
        except Exception as e:
            print(f"Error parsing bounding box string: {value} -> {e}")
            return ""
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        return value
    elif isinstance(value, np.ndarray):
        arr = value.tolist()
        if isinstance(arr, list) and len(arr) == 4:
            return arr
        else:
            return ""
    else:
        return ""

def process_best_license_plate(input_df, input_csv, output_csv):
    """
    Processes a CSV file (provided as a DataFrame) to determine the best license plate text, 
    confidence, and bounding box for each row based on:
      1. The least number of '?' characters.
      2. The highest confidence score (if '?' count is the same).
      3. Removing rows where both license plate texts are empty.

    The function adds the following columns:
      - "best_lp_text"
      - "best_lp_avg_conf"
      - "best_lp_bbox"
      
    Parameters:
    - input_df: A pandas DataFrame already loaded from the input CSV.
    - input_csv: Path to the input CSV file (not used for loading in this version).
    - output_csv: Path to save the updated CSV file.

    Returns:
    - df: The processed pandas DataFrame with the additional columns.
    """
    
    # Use the provided DataFrame directly
    df = input_df.copy()

    # Define function to choose the best license plate info for each row
    def choose_best_lp(row):
        # Extract texts and confidence values (if present)
        lp_text = str(row["lp_text"]) if pd.notna(row["lp_text"]) else ""
        new_lp_text = str(row["new_lp_text"]) if pd.notna(row["new_lp_text"]) else ""

        lp_avg_conf = row["lp_avg_conf"] if pd.notna(row["lp_avg_conf"]) else 0.0
        new_lp_avg_conf = row["new_lp_avg_conf"] if pd.notna(row["new_lp_avg_conf"]) else 0.0

        # Safely parse bounding boxes (using our helper function)
        lp_bbox = safe_parse_bbox(row["lp_bbox"])
        new_lp_bbox = safe_parse_bbox(row["new_lp_bbox"])

        # Case 1: If both texts are empty, return None so the row can be dropped later
        if not lp_text and not new_lp_text:
            return None

        # Case 2: If one text is empty, choose the non-empty one along with its confidence and bbox
        if not lp_text:
            return new_lp_text, new_lp_avg_conf, new_lp_bbox
        if not new_lp_text:
            return lp_text, lp_avg_conf, lp_bbox

        # Count the number of '?' characters in each text
        lp_question_marks = lp_text.count("?")
        new_lp_question_marks = new_lp_text.count("?")

        # Choose based on fewer '?' characters
        if lp_question_marks < new_lp_question_marks:
            best_lp_text = lp_text
            best_lp_avg_conf = lp_avg_conf
            best_lp_bbox = lp_bbox
        elif new_lp_question_marks < lp_question_marks:
            best_lp_text = new_lp_text
            best_lp_avg_conf = new_lp_avg_conf
            best_lp_bbox = new_lp_bbox
        else:
            # If both have the same number of '?' characters, choose the one with higher confidence
            if lp_avg_conf >= new_lp_avg_conf:
                best_lp_text = lp_text
                best_lp_avg_conf = lp_avg_conf
                best_lp_bbox = lp_bbox
            else:
                best_lp_text = new_lp_text
                best_lp_avg_conf = new_lp_avg_conf
                best_lp_bbox = new_lp_bbox

        return best_lp_text, best_lp_avg_conf, best_lp_bbox

    # Apply the function to each row and assign the result to new columns.
    # The lambda wraps the tuple into a pandas Series.
    df[["best_lp_text", "best_lp_avg_conf", "best_lp_bbox"]] = df.apply(lambda row: pd.Series(choose_best_lp(row)), axis=1)

    # Drop rows where best_lp_text is None (i.e., both lp_text values were empty)
    df = df.dropna(subset=["best_lp_text"])

    # Save the updated CSV file
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved as: {output_csv}")
    return df


# Example usage:
# processed_data = process_best_license_plate("filtered_detection_results_hd3.csv", "filtered_detection_results_with_best_lp_hd3.csv")
