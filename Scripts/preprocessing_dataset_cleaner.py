import os
import pandas as pd
import glob
import numpy as np

# Configuration variables
input_directory = "D:/EECE 490 project/Entire Dataset"  # Directory containing CSV files without headers
output_directory = "D:/EECE 490 project/Entire Dataset_cleaned"  # Directory where processed files will be saved
file_pattern = "*.csv"  # Pattern to match CSV files

# Define the headers based on the example provided
headers = [
    "pkSeqID", "stime", "flgs", "proto", "saddr", "sport", "daddr", "dport", 
    "pkts", "bytes", "state", "ltime", "seq", "dur", "mean", "stddev", "smac", 
    "dmac", "sum", "min", "max", "soui", "doui", "sco", "dco", "spkts", 
    "dpkts", "sbytes", "dbytes", "rate", "srate", "drate", "attack", 
    "category", "subcategory"
]

def process_csv_file(input_file, output_file):
    """
    Process a single CSV file:
    1. Add headers if missing
    2. Remove empty columns
    """
    print(f"Processing file: {input_file}")
    
    try:
        # Read the CSV file without headers, handle mixed types with low_memory=False
        df = pd.read_csv(input_file, header=None, low_memory=False)
        
        # If the number of columns matches the headers, assign them
        if df.shape[1] <= len(headers):
            # Assign headers to the dataframe (only as many as there are columns)
            df.columns = headers[:df.shape[1]]
        else:
            print(f"Warning: File {input_file} has more columns ({df.shape[1]}) than defined headers ({len(headers)})")
            # Assign headers to known columns and number the rest
            column_names = headers.copy()
            for i in range(len(headers), df.shape[1]):
                column_names.append(f"extra_col_{i}")
            df.columns = column_names

        # Identify and remove empty columns more efficiently
        # This approach checks for empty values without converting entire DataFrame to strings
        empty_cols = []
        for col in df.columns:
            # Check if column is numeric and all values are NaN
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().all():
                empty_cols.append(col)
            # For string columns, check if all values are empty or just whitespace
            elif df[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).all():
                empty_cols.append(col)
        
        # Remove empty columns
        if empty_cols:
            df = df.drop(columns=empty_cols)
            print(f"Removed {len(empty_cols)} empty columns: {', '.join(empty_cols)}")
        else:
            print("No empty columns found.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the processed dataframe
        df.to_csv(output_file, index=False)
        print(f"Processed file saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False


def process_all_files():
    """
    Process all CSV files in the input directory
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
        output_file = os.path.join(output_directory, f"processed_{file_name}")
        
        # Process the file
        if process_csv_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    print(f"CSV processing completed. Successfully processed: {successful}, Failed: {failed}")


def process_large_files():
    """
    Alternative method for very large files using chunking
    """
    input_pattern = os.path.join(input_directory, file_pattern)
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{input_pattern}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to process with chunking method.")
    
    for input_file in csv_files:
        print(f"Processing large file: {input_file}")
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_directory, f"processed_{file_name}")
        
        try:
            # First, determine the number of columns in the file
            with open(input_file, 'r') as f:
                first_line = f.readline().strip()
                num_columns = len(first_line.split(','))
            
            # Assign headers
            column_names = headers[:num_columns] if num_columns <= len(headers) else headers + [f"extra_col_{i}" for i in range(len(headers), num_columns)]
            
            # Process in chunks
            chunk_size = 100000  # Adjust based on your system's memory
            empty_cols = set(column_names)  # Start assuming all columns are empty
            
            # First pass: identify non-empty columns
            for chunk in pd.read_csv(input_file, header=None, names=column_names, chunksize=chunk_size, low_memory=False):
                for col in list(empty_cols):  # Use list to avoid modifying set during iteration
                    # Check if the column has any non-empty values in this chunk
                    if not chunk[col].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')).all():
                        empty_cols.remove(col)
                
                # If we've found non-empty values for all columns, stop checking
                if not empty_cols:
                    break
            
            # Second pass: read and write in chunks, excluding empty columns
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write header first
            non_empty_cols = [col for col in column_names if col not in empty_cols]
            with open(output_file, 'w') as f:
                f.write(','.join(non_empty_cols) + '\n')
            
            # Process and append each chunk
            for chunk in pd.read_csv(input_file, header=None, names=column_names, chunksize=chunk_size, low_memory=False):
                # Remove empty columns
                chunk = chunk.drop(columns=list(empty_cols))
                # Append to output file without header
                chunk.to_csv(output_file, mode='a', header=False, index=False)
            
            print(f"Removed {len(empty_cols)} empty columns: {', '.join(empty_cols)}")
            print(f"Processed file saved to: {output_file}")
            
        except Exception as e:
            print(f"Error processing large file {input_file}: {str(e)}")


if __name__ == "__main__":
    print("Starting CSV processing...")
    
    # Ask user which method to use
    print("\nChoose processing method:")
    print("1. Standard processing (for normal-sized files)")
    print("2. Chunked processing (for very large files)")
    
    try:
        choice = input("Enter 1 or 2 (default is 1): ").strip()
        if choice == "2":
            process_large_files()
        else:
            process_all_files()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    
    print("CSV processing completed.")