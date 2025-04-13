import os
import csv

# Configuration - Only specify the file numbers
file_numbers_to_process = [7, 9]  # Just specify the numbers

# Paths
input_directory = "D:/EECE 490 project/Entire Dataset"
output_directory = "D:/EECE 490 project/Entire Dataset_cleaned"
file_base_name = "UNSW_2018_IoT_Botnet_Dataset_"

# Define the headers
headers = [
    "pkSeqID", "stime", "flgs", "proto", "saddr", "sport", "daddr", "dport", 
    "pkts", "bytes", "state", "ltime", "seq", "dur", "mean", "stddev", "smac", 
    "dmac", "sum", "min", "max", "soui", "doui", "sco", "dco", "spkts", 
    "dpkts", "sbytes", "dbytes", "rate", "srate", "drate", "attack", 
    "category", "subcategory"
]

# Columns to remove (based on previous results)
columns_to_remove = ["smac", "dmac", "soui", "doui", "sco", "dco"]


def repair_csv_file(input_file, output_file):
    """
    Repair and process CSV files with quote/parsing errors
    """
    print(f"Repairing file: {input_file}")
    
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # First, we'll try to identify the number of columns
        with open(input_file, 'r', errors='replace') as f:
            # Read the first 10 lines to determine column count
            sample_lines = [f.readline() for _ in range(10)]
            delimiter = ','
            num_columns = max(len(line.split(delimiter)) for line in sample_lines)
        
        # Prepare column names (will truncate if needed)
        column_names = headers[:num_columns] if num_columns <= len(headers) else headers + [f"extra_col_{i}" for i in range(len(headers), num_columns)]
        
        # Now open the repaired output file and write the header
        with open(output_file, 'w', newline='') as out_file:
            writer = csv.writer(out_file)
            
            # Write header, excluding columns we know are empty
            filtered_columns = [col for col in column_names if col not in columns_to_remove]
            writer.writerow(filtered_columns)
            
            # Process the input file line by line
            with open(input_file, 'r', errors='replace') as in_file:
                reader = csv.reader((line.replace('\0', '') for line in in_file), 
                                   quoting=csv.QUOTE_NONE, escapechar='\\')
                
                line_count = 0
                for row in reader:
                    line_count += 1
                    
                    # Skip if row is too short
                    if len(row) < 5:  # Arbitrary minimum column threshold
                        continue
                        
                    # Filter out columns we want to remove
                    if len(row) <= len(column_names):
                        # Map columns to their names and filter
                        row_dict = {column_names[i]: row[i] if i < len(row) else '' 
                                   for i in range(min(len(column_names), len(row)))}
                        filtered_row = [row_dict[col] for col in filtered_columns 
                                      if col in row_dict]
                        writer.writerow(filtered_row)
                    
                    # Progress reporting
                    if line_count % 100000 == 0:
                        print(f"  Processed {line_count:,} lines...")
        
        print(f"Repair completed. Processed file saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error repairing file {input_file}: {str(e)}")
        return False


def main():
    print(f"Starting CSV repair for files {file_numbers_to_process}...")
    
    for file_number in file_numbers_to_process:
        file_name = f"{file_base_name}{file_number}.csv"
        input_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, f"processed_{file_name}")
        
        if os.path.exists(input_file):
            repair_csv_file(input_file, output_file)
        else:
            print(f"File not found: {input_file}")
    
    print("CSV repair completed.")


if __name__ == "__main__":
    main()