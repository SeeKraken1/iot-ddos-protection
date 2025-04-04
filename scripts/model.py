import os
import csv
import pickle
import random
import time
import re
from collections import Counter, defaultdict
from datetime import datetime

# Try to import sklearn - if not available, print helpful message
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. Run 'pip install scikit-learn' first.")
    SKLEARN_AVAILABLE = False

# Configuration variables
data_directory = "D:/EECE 490 project/RF_Dataset"  # Directory with preprocessed data
output_directory = "D:/EECE 490 project/Model"  # Directory to save the model
model_filename = "iot_ddos_rf_model.pkl"  # Filename for the saved model

# Lightweight feature set for IoT devices
# These are the most important features for detecting DDoS attacks while keeping model size small
IOT_FEATURES = [
    "proto", "sport", "dport", "seq", "stddev", "min", "mean", "drate", 
    "state_number", "N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP"
]

# Target column
TARGET = "attack"

# Limit sample size to reduce memory usage on IoT devices (adjust based on your device's capabilities)
# This samples rows from each file to build a balanced, representative dataset
MAX_SAMPLES_PER_FILE = 50000  # Adjust based on available memory
MAX_TOTAL_SAMPLES = 500000    # Maximum total samples to use for training

# Fraction of data to use for testing (e.g., 0.2 = 20% for testing, 80% for training)
TEST_SIZE = 0.2

# Model parameters optimized for IoT devices (reduced complexity, less memory usage)
RF_PARAMS = {
    'n_estimators': 10,      # Fewer trees to reduce memory footprint
    'max_depth': 10,         # Limit tree depth to prevent overfitting and reduce memory
    'min_samples_split': 10, # Require more samples to split to create smaller trees
    'min_samples_leaf': 5,   # Require more samples in leaves to reduce tree complexity
    'n_jobs': -1,            # Use all available cores for training
    'random_state': 42,      # For reproducible results
    'class_weight': 'balanced', # Handle imbalanced datasets (more normal than attack traffic)
    'verbose': 1,            # Show progress during training
    'warm_start': False      # Building the forest at once uses less memory than adding trees
}

def is_numeric(value):
    """Check if a value is numeric (can be converted to int or float)"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def is_hex(value):
    """Check if a value is a hexadecimal string"""
    if not isinstance(value, str):
        return False
    
    # Check if it's a hex string (0x followed by hex digits)
    return bool(re.match(r'^0x[0-9a-fA-F]+$', value))

def clean_value(value):
    """
    Clean and normalize values for better encoding
    """
    if value is None:
        return value
    
    # Convert to string for consistent handling
    str_value = str(value).strip().lower()
    
    # Try to convert to number if possible
    if is_numeric(str_value):
        try:
            # Try integer first, then float
            return int(str_value)
        except ValueError:
            return float(str_value)
    
    # Handle hex values by converting to integers
    if is_hex(str_value):
        try:
            return int(str_value, 16)
        except ValueError:
            pass
    
    # For everything else, return as is
    return str_value

def load_sampled_data(file_paths, max_samples_per_file=MAX_SAMPLES_PER_FILE, max_total=MAX_TOTAL_SAMPLES):
    """
    Load and sample data from multiple CSV files to reduce memory usage
    Returns X (features) and y (target) as lists
    """
    print(f"Loading data with max {max_samples_per_file} samples per file, {max_total} samples total")
    
    X_raw = []  # Features before encoding
    y = []      # Target
    
    # For encoding categorical features
    value_mappings = {feature: {} for feature in IOT_FEATURES}
    feature_types = {feature: 'unknown' for feature in IOT_FEATURES}
    
    feature_indices = {}  # Will store the column indices for our features
    total_samples = 0
    
    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}...")
        
        with open(file_path, 'r', errors='replace') as f:
            # Read header to get column indices
            header = next(csv.reader([f.readline()]))
            
            # Find indices of features and target
            if not feature_indices:
                feature_indices = {
                    feature: header.index(feature) if feature in header else -1
                    for feature in IOT_FEATURES
                }
                target_idx = header.index(TARGET) if TARGET in header else -1
                
                # Check if we have the target column
                if target_idx == -1:
                    print(f"Warning: Target column '{TARGET}' not found in {file_path}")
                    continue
                
                # Check which features are available
                missing_features = [f for f, idx in feature_indices.items() if idx == -1]
                if missing_features:
                    print(f"Warning: Missing features in {file_path}: {missing_features}")
            
            # Count lines in file to determine sampling interval
            f.seek(0)
            next(f)  # Skip header
            line_count = sum(1 for _ in f)
            
            # Calculate how many rows to sample
            if max_samples_per_file >= line_count:
                sampling_interval = 1  # Take every row
                samples_to_take = line_count
            else:
                sampling_interval = max(1, line_count // max_samples_per_file)
                samples_to_take = max_samples_per_file
            
            print(f"  File has {line_count} rows, sampling ~{samples_to_take} rows (interval: {sampling_interval})")
            
            # Return to beginning and skip header
            f.seek(0)
            next(f)
            
            # Sample the data
            reader = csv.reader(f)
            row_num = 0
            file_samples = 0
            
            # Convert feature_indices.values() to a list for max() operation
            feature_idxs = list(feature_indices.values())
            max_idx = max(feature_idxs + [target_idx])
            
            for row in reader:
                row_num += 1
                
                # Sample based on interval
                if row_num % sampling_interval != 0:
                    continue
                
                # Extract features and target
                if len(row) <= max_idx:
                    continue  # Skip if row is too short
                
                # Get target value first - only include row if it has a valid attack value
                if target_idx >= 0 and target_idx < len(row):
                    target_val = clean_value(row[target_idx])
                    # Skip if target is not numeric
                    if not isinstance(target_val, (int, float)):
                        continue
                else:
                    continue  # Skip rows without target value
                
                # Extract features
                features = []
                for i, (feature, idx) in enumerate(feature_indices.items()):
                    if idx >= 0 and idx < len(row):
                        # Clean the value
                        value = clean_value(row[idx])
                        
                        # Track value types for later encoding
                        if value is not None:
                            current_type = feature_types[feature]
                            if current_type == 'unknown':
                                if isinstance(value, (int, float)):
                                    feature_types[feature] = 'numeric'
                                else:
                                    feature_types[feature] = 'categorical'
                            elif current_type == 'numeric' and not isinstance(value, (int, float)):
                                # If we find a non-numeric value in what we thought was numeric, 
                                # mark as mixed/categorical
                                feature_types[feature] = 'categorical'
                            
                            # For non-numeric values, build mappings
                            if not isinstance(value, (int, float)) and str(value) not in value_mappings[feature]:
                                value_mappings[feature][str(value)] = len(value_mappings[feature])
                        
                        features.append(value)
                    else:
                        features.append(None)  # Missing feature
                
                # Add to dataset
                X_raw.append(features)
                y.append(target_val)
                
                file_samples += 1
                if file_samples >= max_samples_per_file:
                    break
        
        total_samples += file_samples
        print(f"  Added {file_samples} samples from this file. Total samples: {total_samples}")
        
        if total_samples >= max_total:
            print(f"Reached maximum total samples ({max_total}). Stopping data collection.")
            break
    
    print(f"Data loading complete. Total samples collected: {len(X_raw)}")
    
    # Print feature types and value mappings
    print("\nFeature types and mappings:")
    categorical_features = []
    for feature, ftype in feature_types.items():
        print(f"  {feature}: {ftype}")
        if ftype == 'categorical':
            categorical_features.append(feature)
            print(f"    Values: {value_mappings[feature]}")
    
    # Now encode all non-numeric features
    print("\nEncoding non-numeric features...")
    X = []
    
    for row in X_raw:
        encoded_row = []
        for i, value in enumerate(row):
            feature = IOT_FEATURES[i]
            
            if feature_types[feature] == 'categorical':
                # Categorical feature - encode using mapping
                if value is None:
                    encoded_value = -1  # Default for missing
                elif isinstance(value, (int, float)):
                    encoded_value = value  # Already numeric
                else:
                    encoded_value = value_mappings[feature].get(str(value), -1)
                encoded_row.append(encoded_value)
            else:
                # Numeric feature - use as is or default
                if value is None:
                    encoded_row.append(0)  # Default for missing
                elif isinstance(value, (int, float)):
                    encoded_row.append(value)
                else:
                    # Try to convert to number as fallback
                    try:
                        encoded_row.append(float(value))
                    except (ValueError, TypeError):
                        encoded_row.append(0)  # Default for non-convertible
        X.append(encoded_row)
    
    print("Encoding complete. All features converted to numeric values.")
    
    # Check for class balance
    y_counter = Counter(y)
    print(f"\nClass distribution: {dict(y_counter)}")
    
    # For binary classification, convert to 0/1
    if len(y_counter) == 2:
        # Check if we need to convert to binary (0/1)
        values = list(y_counter.keys())
        if not (0 in values and 1 in values):
            print("Converting target to binary format (0/1)")
            # Map the first value to 0, the second to 1
            mapping = {values[0]: 0, values[1]: 1}
            y = [mapping[val] for val in y]
            print(f"Mapping: {mapping}")
            print(f"New class distribution: {dict(Counter(y))}")
    
    # Save value mappings to use during inference
    mapping_file = os.path.join(output_directory, "feature_mappings.pkl")
    os.makedirs(output_directory, exist_ok=True)
    
    with open(mapping_file, 'wb') as f:
        pickle.dump({
            'value_mappings': value_mappings,
            'feature_types': feature_types
        }, f)
    
    print(f"Feature mappings saved to: {mapping_file}")
    
    return X, y, value_mappings, feature_types

def train_model(X, y, params=RF_PARAMS):
    """Train a Random Forest model with the given parameters"""
    start_time = time.time()
    print(f"Starting model training with parameters: {params}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    
    print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples")
    
    # Train the model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # More detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate precision, recall and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f"\nWeighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = sorted(zip(IOT_FEATURES, importances), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importance:")
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.4f}")
    
    return model, X_test, y_test

def save_model(model, output_path):
    """Save the trained model to disk"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Get model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel saved to: {output_path}")
    print(f"Model size: {model_size_mb:.2f} MB")

def generate_deployment_code(output_dir, value_mappings, feature_types):
    """Generate lightweight inference code for IoT deployment"""
    code_path = os.path.join(output_dir, "iot_ddos_detector.py")
    
    code = """# IoT DDoS Attack Detector
# Generated on {date}
# Lightweight inference code for IoT devices

import pickle
import time
import re

# Load the model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Feature types from training
FEATURE_TYPES = {feature_types}

# Value mappings from training
VALUE_MAPPINGS = {value_mappings}

# Helper function to check if a value is numeric
def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Helper function to check if a value is hex
def is_hex(value):
    if not isinstance(value, str):
        return False
    
    # Check if it's a hex string (0x followed by hex digits)
    return bool(re.match(r'^0x[0-9a-fA-F]+$', value))

# Clean and normalize values
def clean_value(value):
    if value is None:
        return value
    
    # Convert to string for consistent handling
    str_value = str(value).strip().lower()
    
    # Try to convert to number if possible
    if is_numeric(str_value):
        try:
            # Try integer first, then float
            return int(str_value)
        except ValueError:
            return float(str_value)
    
    # Handle hex values by converting to integers
    if is_hex(str_value):
        try:
            return int(str_value, 16)
        except ValueError:
            pass
    
    # For everything else, return as is
    return str_value

# Process input for prediction
def preprocess_input(data, feature_names):
    features = []
    for feature in feature_names:
        value = data.get(feature)
        
        # Clean the value
        value = clean_value(value)
        
        # Encode based on feature type
        if FEATURE_TYPES.get(feature) == 'categorical':
            # Categorical feature
            if value is None:
                features.append(-1)  # Default for missing
            elif isinstance(value, (int, float)):
                features.append(value)
            else:
                # Look up in mapping
                mapping = VALUE_MAPPINGS.get(feature, {{}})
                features.append(mapping.get(str(value), -1))
        else:
            # Numeric feature
            if value is None:
                features.append(0)  # Default for missing
            elif isinstance(value, (int, float)):
                features.append(value)
            else:
                # Try to convert to number
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    features.append(0)
    
    return features

# Run inference
def detect_attack(model, data, feature_names):
    # Preprocess the input
    features = preprocess_input(data, feature_names)
    
    # Get prediction
    start_time = time.time()
    prediction = model.predict([features])[0]
    inference_time = time.time() - start_time
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba([features])[0][1]  # Probability of class 1
    
    return {{
        'is_attack': bool(prediction),
        'probability': probability,
        'inference_time_ms': inference_time * 1000
    }}

# Example usage
if __name__ == "__main__":
    # Feature names used by the model
    FEATURE_NAMES = {feature_list}
    
    # Load the model
    try:
        model = load_model('iot_ddos_rf_model.pkl')
        print("Model loaded successfully")
        
        # Example data point (replace with real-time data from your network)
        sample_data = {{
            'proto': 'tcp',
            'sport': 80,
            'dport': 443,
            'seq': 1234,
            'stddev': 0.5,
            'min': 10,
            'mean': 50,
            'drate': 0.1,
            'state_number': 1,
            'N_IN_Conn_P_SrcIP': 10,
            'N_IN_Conn_P_DstIP': 5
        }}
        
        # Run detection
        result = detect_attack(model, sample_data, FEATURE_NAMES)
        
        # Print results
        print(f"Attack detected: {{result['is_attack']}}")
        if result['probability'] is not None:
            print(f"Attack probability: {{result['probability']:.2f}}")
        print(f"Inference time: {{result['inference_time_ms']:.2f}} ms")
        
    except Exception as e:
        print(f"Error: {{e}}")
""".format(
    date=datetime.now().strftime("%Y-%m-%d"), 
    value_mappings=repr(value_mappings),
    feature_types=repr(feature_types),
    feature_list=repr(IOT_FEATURES)
)
    
    with open(code_path, 'w') as f:
        f.write(code)
    
    print(f"\nDeployment code generated: {code_path}")
    print("This lightweight code can be used to run inference on IoT devices.")

def main():
    if not SKLEARN_AVAILABLE:
        return
    
    print("IoT DDoS Random Forest Trainer")
    print("==============================")
    
    # Check if data directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Data directory not found: {data_directory}")
        return
    
    # Get all csv files in data directory
    data_files = [
        os.path.join(data_directory, f) 
        for f in os.listdir(data_directory) 
        if f.endswith('.csv')
    ]
    
    if not data_files:
        print(f"No CSV files found in {data_directory}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Ask user if they want to limit the number of files to use
    use_all_files = input(f"Use all {len(data_files)} files? (y/n, default y): ").strip().lower() != 'n'
    
    if not use_all_files:
        num_files = min(5, len(data_files))
        try:
            user_input = input(f"How many files to use (1-{len(data_files)}, default 5): ").strip()
            if user_input:
                num_files = int(user_input)
                num_files = max(1, min(num_files, len(data_files)))
        except ValueError:
            print(f"Invalid input. Using {num_files} files.")
        
        # Randomly select files to ensure good representation
        random.seed(42)  # For reproducibility
        data_files = random.sample(data_files, num_files)
    
    print(f"Using {len(data_files)} files for training")
    
    # Load data with automatic encoding
    X, y, value_mappings, feature_types = load_sampled_data(data_files)
    
    if len(X) == 0:
        print("No data loaded. Check your data files and target column.")
        return
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model
    output_path = os.path.join(output_directory, model_filename)
    save_model(model, output_path)
    
    # Generate deployment code
    generate_deployment_code(output_directory, value_mappings, feature_types)
    
    print("\nProcess completed successfully.")

if __name__ == "__main__":
    main()