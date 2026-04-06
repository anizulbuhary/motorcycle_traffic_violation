# step 07
import pandas as pd

def filter_violations(input_df, input_csv, output_csv):
    """
    Filters out specific violation classes ('DHelmet', 'DHelmetP1Helmet') from the input CSV file 
    and saves the result to a new CSV file.

    Parameters:
    - input_csv: Path to the input CSV file.
    - output_csv: Path where the filtered CSV file will be saved.

    Returns:
    - filtered_df: The filtered pandas DataFrame.
    """
    
    # Read the CSV into a DataFrame
    df = input_df

    # Filter out rows where violation_class is 'DHelmet' or 'DHelmetP1Helmet'
    filtered_df = df[~df["violation_class"].isin(["DHelmet", "DHelmetP1Helmet"])]

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered CSV saved as: {output_csv}")
    return filtered_df  # Return the filtered DataFrame


# Example usage:
# filtered_data = filter_violations("filtered_detection_results_with_best_detection_hd3.csv", "violation_results_hd3.csv")
