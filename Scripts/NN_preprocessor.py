import os
import pandas as pd
import glob
import csv

# Configuration variables
input_directory = "D:/EECE 490 project/Neural Network/Entire Dataset_cleaned"  # Directory containing processed CSV files
output_directory = "D:/EECE 490 project/Neural Network/NN_Ready_Dataset"  # Directory for neural network ready data
file_pattern = "processed_*.csv"  # Pattern to match CSV files

# Define the headers to keep (for neural network model)
headers_to_keep = [
    "pkSeqID", "proto", "saddr", "sport", "daddr", "dport", 
    "seq", "stddev", "min", "mean", "state", "max", 
    "drate", "attack", "category", "subcategory"
]

# Additional headers mentioned that aren't in the original data
# We'll check if they exist in the dataset
additional_headers = [
    "N_IN_Conn_P_SrcIP", "state_number", "N_IN_Conn_P_DstIP"
]

def process_csv_file(input_file, output_file):
    """
    Process a single CSV file:
    1. Keep only the specified headers
    2. Handle any encoding or parsing issues
    """
    print(f"Processing file: {input_file}")
    
    try:
        # Read the CSV file with headers
        df = pd.read_csv(input_file, low_memory=False)
        
        # Get the list of available columns in the dataframe
        available_columns = df.columns.tolist()
        
        # Determine which columns to keep (intersection of desired headers and available columns)
        columns_to_keep = [col for col in headers_to_keep + additional_headers if col in available_columns]
        
        # Report on any headers that weren't found
        missing_headers = [col for col in headers_to_keep + additional_headers if col not in available_columns]
        if missing_headers:
            print(f"  Warning: The following requested headers were not found: {', '.join(missing_headers)}")
        
        # Select only the columns we want to keep
        if columns_to_keep:
            df_filtered = df[columns_to_keep]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save the processed dataframe
            df_filtered.to_csv(output_file, index=False)
            print(f"  Kept {len(columns_to_keep)} columns: {', '.join(columns_to_keep)}")
            print(f"  Processed file saved to: {output_file}")
            return True
        else:
            print(f"  Error: None of the requested headers were found in {input_file}")
            return False
        
    except Exception as e:
        print(f"  Error processing file {input_file}: {str(e)}")
        return False


def process_large_csv_file(input_file, output_file):
    """
    Process a large CSV file using chunking to reduce memory usage:
    1. Keep only the specified headers
    2. Handle any encoding or parsing issues
    """
    print(f"Processing large file: {input_file}")
    
    try:
        # First, read the header to identify available columns
        with open(input_file, 'r') as f:
            header_line = f.readline().strip()
            available_columns = header_line.split(',')
        
        # Determine which columns to keep
        columns_to_keep = [col for col in headers_to_keep + additional_headers if col in available_columns]
        
        # Report on any headers that weren't found
        missing_headers = [col for col in headers_to_keep + additional_headers if col not in available_columns]
        if missing_headers:
            print(f"  Warning: The following requested headers were not found: {', '.join(missing_headers)}")
        
        if not columns_to_keep:
            print(f"  Error: None of the requested headers were found in {input_file}")
            return False
        
        # Get column indices for the columns we want to keep
        column_indices = [available_columns.index(col) for col in columns_to_keep]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Open output file and write the header
        with open(output_file, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(columns_to_keep)
            
            # Process the input file in chunks
            chunk_size = 100000  # Adjust based on available memory
            chunk_reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)
            
            row_count = 0
            for chunk in chunk_reader:
                # Filter the chunk to only include the columns we want
                chunk_filtered = chunk[columns_to_keep]
                
                # Append to the output file without header
                chunk_filtered.to_csv(out_f, header=False, index=False, mode='a')
                
                row_count += len(chunk)
                print(f"  Processed {row_count:,} rows...")
        
        print(f"  Kept {len(columns_to_keep)} columns: {', '.join(columns_to_keep)}")
        print(f"  Processed file saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"  Error processing large file {input_file}: {str(e)}")
        return False


def main():
    print("Starting IoT DDoS prevention data preprocessing...")
    
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
    
    # Ask user which processing method to use
    print("\nChoose processing method:")
    print("1. Standard processing (for normal-sized files)")
    print("2. Chunked processing (for very large files)")
    
    try:
        choice = input("Enter 1 or 2 (default is 2 for large files): ").strip()
        if choice == "1":
            processing_function = process_csv_file
        else:
            processing_function = process_large_csv_file
        
        # Track progress
        successful = 0
        failed = 0
        
        # Process each file
        for input_file in csv_files:
            # Generate output filename
            file_name = os.path.basename(input_file).replace("processed_", "nn_")
            output_file = os.path.join(output_directory, file_name)
            
            # Process the file
            if processing_function(input_file, output_file):
                successful += 1
            else:
                failed += 1
        
        print(f"\nPreprocessing completed:")
        print(f"  Successfully processed: {successful} files")
        print(f"  Failed: {failed} files")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    
    print("\nIoT DDoS prevention data preprocessing completed.")


if __name__ == "__main__":
    main()