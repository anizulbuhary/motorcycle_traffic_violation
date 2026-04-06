# step 02

import pandas as pd
import re

def extract_best_tracking_results(input_df, input_csv, output_csv, is_df):
    """
    Extracts the best frames for each track_id based on license plate quality and confidence.
    
    Parameters:
    - input_csv: Path to the input CSV file containing tracking data.
    - output_csv: Path where the best tracking results CSV will be saved.
    
    Returns:
    - best_frames_df: The filtered pandas DataFrame containing the best frames.
    """
    
    # Load CSV file
    if is_df:
        df = input_df.copy()
    else:
        df = pd.read_csv(input_csv)

    # Compile the regex pattern for valid license plates
    regex_pattern = re.compile(r'^[A-Z]{3}-\d{3,4}(-\d{2})?(-[A-Z])?$')

    # Dictionary to store best frames for each track_id
    best_frames = {}

    # Iterate through unique track IDs
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]

        # Filter rows where license plate text is detected and not empty or unknown
        detected_plates = track_data[track_data['lp_text'].notna() & 
                                     (track_data['lp_text'] != '') & 
                                     (track_data['plate_type'] != 'unknown')]

        if detected_plates.empty:
            continue  # No valid LP text detected for this track_id

        # Create a copy before modifying
        detected_plates = detected_plates.copy()

        # Rank LP texts based on similarity to regex pattern (count of '?' characters)
        detected_plates.loc[:, 'lp_quality'] = detected_plates['lp_text'].apply(lambda x: x.count('?'))

        # Find the best-matching LP text (lowest count of '?')
        min_quality = detected_plates['lp_quality'].min()
        best_matches = detected_plates[detected_plates['lp_quality'] == min_quality]

        # If multiple frames have the same LP text, choose the one with the highest lp_avg_conf
        best_matches = best_matches.sort_values(by=['lp_text', 'lp_avg_conf'], ascending=[True, False])
        best_matches = best_matches.groupby('lp_text').head(1)  # Keep highest confidence per LP text

        # If more than 3 best frames exist for this track ID, take the top 3 with highest lp_avg_conf
        best_frames[track_id] = best_matches.nlargest(3, 'lp_avg_conf')

    # Concatenate all best frames into a new DataFrame
    best_frames_df = pd.concat(best_frames.values(), ignore_index=True)

    # Drop the temporary column used for ranking
    best_frames_df = best_frames_df.drop(columns=['lp_quality'])

    # Save to a new CSV file
    best_frames_df.to_csv(output_csv, index=False)

    print(f"Best frames saved to {output_csv}")
    return best_frames_df  # Return the filtered DataFrame


# Example usage:
# best_data = extract_best_tracking_results("tracking_results_hd3.csv", "best_tracking_results_hd3.csv")
