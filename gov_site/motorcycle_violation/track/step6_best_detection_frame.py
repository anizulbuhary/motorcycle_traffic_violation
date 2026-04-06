# step 06
import pandas as pd

def select_best_detections(input_df, input_csv, output_csv):
    """
    Processes a CSV file to select the best detection per track_id based on:
    1. The most frequent detection class.
    2. The least number of '?' in best_lp_text.
    3. The highest best_lp_avg_conf (if there's still a tie).

    Parameters:
    - input_csv: Path to the input CSV file.
    - output_csv: Path to save the filtered CSV file.

    Returns:
    - best_detections: The filtered pandas DataFrame.
    """
    
    # Load CSV file
    df = input_df

    # Ensure required columns exist
    required_columns = {"track_id", "violation_class", "violation_class_id", "best_lp_text", "best_lp_avg_conf"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    # Group by track_id and count occurrences of each detection class
    detection_counts = df.groupby(["track_id", "violation_class", "violation_class_id"]).size().reset_index(name="count")

    # Merge the count data back to the original DataFrame
    df = df.merge(detection_counts, on=["track_id", "violation_class", "violation_class_id"], how="left")

    # Function to choose the best detection class for each track_id
    def select_best_detection(track_group):
        # Find the most frequent detection class
        max_count = track_group["count"].max()
        best_classes = track_group[track_group["count"] == max_count]

        # If only one best class, return that row as a DataFrame
        if len(best_classes) == 1:
            return best_classes.iloc[[0]]  # Ensures a DataFrame is returned

        # Step 1: Choose based on least number of '?' in best_lp_text
        best_classes = best_classes.copy()  # Prevent SettingWithCopyWarning
        best_classes["best_lp_text"] = best_classes["best_lp_text"].astype(str).fillna("")
        best_classes["question_marks"] = best_classes["best_lp_text"].apply(lambda x: x.count("?"))

        min_question_marks = best_classes["question_marks"].min()
        best_classes = best_classes[best_classes["question_marks"] == min_question_marks]

        # Step 2: If there's still a tie, choose based on highest best_lp_avg_conf
        best_detection = best_classes.loc[[best_classes["best_lp_avg_conf"].idxmax()]]  # Ensures DataFrame

        return best_detection

    # Apply function to ensure one row per track_id
    best_detections = df.groupby("track_id", group_keys=False).apply(select_best_detection).reset_index(drop=True)

    # Fix issue: Ensure all original columns are retained
    if isinstance(best_detections, pd.Series):  # Handle unexpected Series return
        best_detections = best_detections.to_frame().T

    # Ensure all original columns exist
    for col in df.columns:
        if col not in best_detections.columns:
            best_detections[col] = None  # Add missing columns with default None

    # Maintain original column order
    best_detections = best_detections[df.columns]

    # Save updated CSV with only one row per track_id
    best_detections.to_csv(output_csv, index=False)

    print(f"Updated CSV saved as: {output_csv}")
    return best_detections  # Return the filtered DataFrame


# Example usage:
# best_data = select_best_detections("filtered_detection_results_with_best_lp_hd3.csv", "filtered_detection_results_with_best_detection_hd3.csv")
