import os
import pandas as pd
import glob

# Configuration variables
input_directory = "D:/EECE 490 project/Entire Dataset_cleaned"  # Directory containing processed CSV files 
output_directory = "D:/EECE 490 project/RF_Dataset"  # Directory where processed files will be saved
file_pattern = "processed_*.csv"  # Pattern to match CSV files

# Define only the headers we want to keep
headers_to_keep = [
    "pkSeqID", "proto", "saddr", "sport", "daddr", "dport", "seq", "stddev", 
    "min", "mean", "drate", "state", "max", "attack", "category", "subcategory"
]

# Headers that need to be added (these don't exist in original dataset)
headers_to_add = [
    "N_IN_Conn_P_SrcIP", 
    "N_IN_Conn_P_DstIP",
    "state_number"
]

def process_csv_file(input_file, output_file):
    """
    Process a single CSV file to keep only specified headers and add new ones
    """
    print(f"Processing file: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check which headers from our keep list actually exist in the file
        existing_headers = [h for h in headers_to_keep if h in df.columns]
        missing_headers = [h for h in headers_to_keep if h not in df.columns]
        
        if missing_headers:
            print(f"Warning: The following headers are missing from the input file: {', '.join(missing_headers)}")
        
        # Select only the columns we want to keep
        df_selected = df[existing_headers].copy()
        
        # State to number conversion (if 'state' exists)
        if 'state' in df_selected.columns:
            # Create a mapping of state strings to numbers
            state_mapping = {state: i for i, state in enumerate(df_selected['state'].unique())}
            df_selected['state_number'] = df_selected['state'].map(state_mapping)
            print(f"Created 'state_number' with {len(state_mapping)} unique states")
        else:
            # If state doesn't exist, add an empty state_number column
            df_selected['state_number'] = None
        
        # Calculate the connection count per source IP
        if 'saddr' in df_selected.columns:
            src_ip_counts = df_selected['saddr'].value_counts().to_dict()
            df_selected['N_IN_Conn_P_SrcIP'] = df_selected['saddr'].map(src_ip_counts)
            print(f"Added 'N_IN_Conn_P_SrcIP' feature")
        else:
            df_selected['N_IN_Conn_P_SrcIP'] = None
        
        # Calculate the connection count per destination IP
        if 'daddr' in df_selected.columns:
            dst_ip_counts = df_selected['daddr'].value_counts().to_dict()
            df_selected['N_IN_Conn_P_DstIP'] = df_selected['daddr'].map(dst_ip_counts)
            print(f"Added 'N_IN_Conn_P_DstIP' feature")
        else:
            df_selected['N_IN_Conn_P_DstIP'] = None
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the processed dataframe
        df_selected.to_csv(output_file, index=False)
        print(f"Processed file saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False


def process_all_files():
    """
    Process all processed CSV files in the input directory
    """
    # Make sure input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    # Get list of CSV files
    input_pattern = os.path.join(input_directory, file_pattern)
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{input_pattern}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Track progress
    successful = 0
    failed = 0
    
    # Process each file
    for input_file in csv_files:
        # Generate output filename
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_directory, f"rf_{file_name}")
        
        # Process the file
        if process_csv_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    print(f"CSV processing completed. Successfully processed: {successful}, Failed: {failed}")


if __name__ == "__main__":
    print("Starting RF preprocessing for DDoS detection...")
    process_all_files()
    print("Preprocessing completed.")