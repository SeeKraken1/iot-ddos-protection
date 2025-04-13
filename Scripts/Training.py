import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc  # Garbage collector

# Configuration
REAL_DATA_DIR = "D:/EECE 490 project/Neural Network/NN_Ready_Dataset"  # Real dataset
SYNTHETIC_DATA_DIR = "D:/EECE 490 project/Neural Network/Synthetic_Dataset"  # Synthetic dataset
MODEL_DIR = "D:/EECE 490 project/Neural Network/Models"
LOGS_DIR = "D:/EECE 490 project/Neural Network/Logs"
RESULTS_DIR = "D:/EECE 490 project/Neural Network/Results"
CHUNK_SIZE = 100000  # Process 100,000 rows at a time
TRAIN_RATIO = 0.8  # Use 80% of files for training
SYNTHETIC_TRAIN_RATIO = 0.7  # Use 70% of synthetic files for training
LOW_MEMORY = True  # Use low_memory=True when reading CSVs to handle mixed types

# Create necessary directories
for directory in [MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Timestamp for unique model/log naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def split_train_test(real_data_dir, synthetic_data_dir, real_ratio=0.8, synthetic_ratio=0.7):
    """Split both real and synthetic files into training and testing sets"""
    import random
    
    # Get all real CSV files
    real_files = [os.path.join(real_data_dir, f) for f in os.listdir(real_data_dir) if f.endswith('.csv')]
    
    # Get all synthetic CSV files
    synthetic_files = []
    if os.path.exists(synthetic_data_dir):
        synthetic_files = [os.path.join(synthetic_data_dir, f) for f in os.listdir(synthetic_data_dir) if f.endswith('.csv')]
    
    # Shuffle files to ensure random distribution
    random.seed(42)  # For reproducibility
    random.shuffle(real_files)
    random.shuffle(synthetic_files)
    
    # Calculate split points
    real_split_idx = int(len(real_files) * real_ratio)
    synthetic_split_idx = int(len(synthetic_files) * synthetic_ratio)
    
    # Split into training and testing sets
    real_train_files = real_files[:real_split_idx]
    real_test_files = real_files[real_split_idx:]
    
    synthetic_train_files = synthetic_files[:synthetic_split_idx] if synthetic_files else []
    synthetic_test_files = synthetic_files[synthetic_split_idx:] if synthetic_files else []
    
    # Combine for training and testing
    train_files = real_train_files + synthetic_train_files
    test_files = real_test_files + synthetic_test_files
    
    # Print summary
    print(f"Real files: {len(real_files)} total")
    print(f"  - Training: {len(real_train_files)} ({real_ratio*100:.0f}%)")
    print(f"  - Testing: {len(real_test_files)} ({(1-real_ratio)*100:.0f}%)")
    
    print(f"Synthetic files: {len(synthetic_files)} total")
    print(f"  - Training: {len(synthetic_train_files)} ({synthetic_ratio*100:.0f}%)")
    print(f"  - Testing: {len(synthetic_test_files)} ({(1-synthetic_ratio)*100:.0f}%)")
    
    print(f"Combined training files: {len(train_files)}")
    print(f"Combined testing files: {len(test_files)}")
    
    return train_files, test_files, real_test_files, synthetic_test_files

def count_rows_in_files(files):
    """Estimate total rows in CSV files to determine sample size"""
    # For speed, just estimate rather than counting exact rows
    # Assume each file has roughly the same number of rows
    avg_rows_per_file = 1000000  # Based on your previous log showing ~1M rows per file
    total_rows = len(files) * avg_rows_per_file
    
    return total_rows

def extract_ip_features(row):
    """Extract features from IP addresses in a single row"""
    features = {}
    
    # Process source IP
    try:
        if pd.isna(row['saddr']) or row['saddr'] == '':
            features.update({
                'sip1': 0, 'sip2': 0, 'sip3': 0, 'sip4': 0
            })
        else:
            octets = str(row['saddr']).split('.')
            if len(octets) == 4:
                features.update({
                    'sip1': int(octets[0]),
                    'sip2': int(octets[1]),
                    'sip3': int(octets[2]),
                    'sip4': int(octets[3])
                })
            else:
                features.update({
                    'sip1': 0, 'sip2': 0, 'sip3': 0, 'sip4': 0
                })
    except:
        features.update({
            'sip1': 0, 'sip2': 0, 'sip3': 0, 'sip4': 0
        })
    
    # Process destination IP
    try:
        if pd.isna(row['daddr']) or row['daddr'] == '':
            features.update({
                'dip1': 0, 'dip2': 0, 'dip3': 0, 'dip4': 0
            })
        else:
            octets = str(row['daddr']).split('.')
            if len(octets) == 4:
                features.update({
                    'dip1': int(octets[0]),
                    'dip2': int(octets[1]),
                    'dip3': int(octets[2]),
                    'dip4': int(octets[3])
                })
            else:
                features.update({
                    'dip1': 0, 'dip2': 0, 'dip3': 0, 'dip4': 0
                })
    except:
        features.update({
            'dip1': 0, 'dip2': 0, 'dip3': 0, 'dip4': 0
        })
    
    return features

def preprocess_chunk(chunk):
    """Preprocess a single chunk of data"""
    # Make a copy
    chunk_processed = chunk.copy()
    
    # Process IP addresses one row at a time (more memory efficient)
    ip_features = []
    for _, row in chunk_processed.iterrows():
        ip_features.append(extract_ip_features(row))
    
    # Convert IP features to DataFrame and join with original
    ip_df = pd.DataFrame(ip_features)
    chunk_processed = pd.concat([chunk_processed, ip_df], axis=1)
    
    # Drop original IP columns
    chunk_processed.drop(['saddr', 'daddr'], axis=1, inplace=True)
    
    # Convert categorical columns to strings to ensure uniformity
    if 'proto' in chunk_processed.columns:
        chunk_processed['proto'] = chunk_processed['proto'].astype(str)
    
    if 'state' in chunk_processed.columns:
        chunk_processed['state'] = chunk_processed['state'].astype(str)
    
    # Convert numeric columns to appropriate types, handling mixed types
    numeric_columns = ['sport', 'dport', 'seq', 'stddev', 'min', 'mean', 'max', 'drate']
    for col in numeric_columns:
        if col in chunk_processed.columns:
            # Convert to numeric, errors='coerce' will convert invalid strings to NaN
            chunk_processed[col] = pd.to_numeric(chunk_processed[col], errors='coerce')
            # Fill NaN values with 0
            chunk_processed[col] = chunk_processed[col].fillna(0)
    
    # Add computed features (to help with generalization)
    # Port category features (common vs uncommon ports)
    chunk_processed['sport_is_well_known'] = (chunk_processed['sport'] < 1024).astype(int)
    chunk_processed['dport_is_well_known'] = (chunk_processed['dport'] < 1024).astype(int)
    
    # Ratio features if possible
    if 'min' in chunk_processed.columns and 'max' in chunk_processed.columns:
        # Avoid division by zero
        chunk_processed['min_max_ratio'] = chunk_processed['min'] / chunk_processed['max'].replace(0, 1)
    
    # Handle missing values
    chunk_processed = chunk_processed.fillna(0)
    
    # Make sure attack is encoded properly
    if 'attack' in chunk_processed.columns:
        chunk_processed['attack'] = chunk_processed['attack'].astype(int)
    
    return chunk_processed

def process_data_in_chunks(files_to_process, sample_size=1000000):
    """Process data in chunks and create a smaller representative sample"""
    print(f"Processing data in chunks with target sample size: {sample_size:,}")
    
    # Calculate sampling ratio based on estimated total rows and desired sample size
    total_rows = count_rows_in_files(files_to_process)
    sampling_ratio = min(1.0, sample_size / total_rows) if total_rows > 0 else 1.0
    
    print(f"Estimated total rows: {total_rows:,}")
    print(f"Using sampling ratio: {sampling_ratio:.4f}")
    
    # Lists to store sampled chunks
    normal_samples = []
    attack_samples = []
    
    # Define specific dtypes for columns with mixed types
    dtype_dict = {
        'sport': 'str',  # Handle as string initially, will convert numerically later
        'dport': 'str',  # Handle as string initially, will convert numerically later
    }
    
    for file in files_to_process:
        print(f"Processing file: {os.path.basename(file)}...")
        
        try:
            # Read and process file in chunks with explicit dtype
            chunk_reader = pd.read_csv(file, 
                                      chunksize=CHUNK_SIZE, 
                                      dtype=dtype_dict, 
                                      low_memory=False)  # Explicitly set low_memory=False
            
            for i, chunk in enumerate(chunk_reader):
                print(f"  Processing chunk {i+1}...")
                
                try:
                    # Sample rows from this chunk based on sampling ratio
                    chunk_sample = chunk.sample(frac=sampling_ratio, random_state=42)
                    
                    if len(chunk_sample) > 0:
                        # Preprocess the sampled chunk
                        processed_chunk = preprocess_chunk(chunk_sample)
                        
                        # Split into normal and attack traffic for balanced sampling
                        if 'attack' in processed_chunk.columns:
                            normal_chunk = processed_chunk[processed_chunk['attack'] == 0]
                            attack_chunk = processed_chunk[processed_chunk['attack'] == 1]
                            
                            if len(normal_chunk) > 0:
                                normal_samples.append(normal_chunk)
                            if len(attack_chunk) > 0:
                                attack_samples.append(attack_chunk)
                        else:
                            # If no attack column, just add the whole chunk
                            normal_samples.append(processed_chunk)
                except Exception as e:
                    print(f"  Error processing chunk {i+1}: {str(e)}")
                    continue
                
                # Force garbage collection to free memory
                gc.collect()
                
                # Check if we have enough samples
                normal_count = sum(len(df) for df in normal_samples)
                attack_count = sum(len(df) for df in attack_samples)
                
                if normal_count >= sample_size // 2 and attack_count >= sample_size // 2:
                    print(f"  Reached target sample size. Normal: {normal_count}, Attack: {attack_count}")
                    break
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Combine samples
    print("Combining samples...")
    
    if not normal_samples and not attack_samples:
        raise ValueError("No valid samples were collected from any files")
    
    normal_df = pd.concat(normal_samples, ignore_index=True) if normal_samples else pd.DataFrame()
    attack_df = pd.concat(attack_samples, ignore_index=True) if attack_samples else pd.DataFrame()
    
    print(f"Collected samples - Normal: {len(normal_df):,}, Attack: {len(attack_df):,}")
    
    # Balance classes if needed
    if len(normal_df) > 0 and len(attack_df) > 0:
        # Take at most 50% of the sample size from each class
        max_per_class = sample_size // 2
        
        if len(normal_df) > max_per_class:
            normal_df = normal_df.sample(max_per_class, random_state=42)
        
        if len(attack_df) > max_per_class:
            attack_df = attack_df.sample(max_per_class, random_state=42)
    
    # Combine balanced samples
    combined_df = pd.concat([normal_df, attack_df], ignore_index=True) if len(attack_df) > 0 else normal_df
    
    # Shuffle the final dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset shape: {combined_df.shape}")
    return combined_df

def prepare_features_and_target(df, target_col='attack'):
    """Prepare features and target for model training"""
    print("Preparing features and target...")
    
    # Define categorical and numerical columns
    categorical_cols = ['proto', 'state']
    
    # ID columns and target/label columns to exclude from features
    exclude_cols = ['pkSeqID', 'category', 'subcategory']
    if target_col in df.columns:
        exclude_cols.append(target_col)
    
    # Get all numerical columns (everything except categorical and excluded)
    numerical_cols = [col for col in df.columns 
                     if col not in categorical_cols + exclude_cols]
    
    # Check if we have categorical columns in the data
    existing_categorical = [col for col in categorical_cols if col in df.columns]
    
    # Create preprocessor based on available columns
    transformers = []
    
    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))
    
    if existing_categorical:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), existing_categorical))
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Extract features and target
    X = df.drop(exclude_cols, axis=1)
    y = df[target_col] if target_col in df.columns else None
    
    # Split the data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Fit and transform the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform the test data
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names for reference
        feature_names = []
        if numerical_cols:
            feature_names.extend(numerical_cols)
        if existing_categorical and len(preprocessor.transformers_) > 1:
            feature_names.extend(
                preprocessor.named_transformers_['cat'].get_feature_names_out(existing_categorical)
            )
        
        return X_train_processed, X_test_processed, y_train, y_test, feature_names, preprocessor, X_train
    else:
        # For prediction only (no labels)
        X_processed = preprocessor.fit_transform(X)
        feature_names = []
        if numerical_cols:
            feature_names.extend(numerical_cols)
        if existing_categorical and len(preprocessor.transformers_) > 1:
            feature_names.extend(
                preprocessor.named_transformers_['cat'].get_feature_names_out(existing_categorical)
            )
        return X_processed, None, None, None, feature_names, preprocessor, X

def build_neural_network(input_dim):
    """Build a neural network model optimized for DDoS detection on IoT devices"""
    print(f"Building lightweight neural network with input dimension: {input_dim}")
    
    # Using a more compact architecture suitable for IoT devices
    model = Sequential([
        # First layer - reduced size for IoT deployment
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second layer - even more compact
        Dense(16, activation='relu'),
        BatchNormalization(),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Use binary crossentropy for binary classification (attack/no attack)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Print model summary to see parameter count
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128):
    """Train the neural network model with expanded epochs"""
    print(f"Training model with {epochs} epochs and batch size {batch_size}...")
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience to allow more exploration
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"ddos_model_{TIMESTAMP}.h5"),
        monitor='val_loss',
        save_best_only=True
    )
    
    # Learning rate scheduler to improve convergence
    def lr_schedule(epoch):
        initial_lr = 0.001
        if epoch > 40:
            return initial_lr * 0.1
        elif epoch > 20:
            return initial_lr * 0.5
        else:
            return initial_lr
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,  # Reduced batch size for better convergence
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print results"""
    print("Evaluating model...")
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{TIMESTAMP}.png"))
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{TIMESTAMP}.png"))
    
    return accuracy, report, conf_matrix

def evaluate_on_files(model, preprocessor, files, name="Test", sample_limit=100000):
    """Evaluate the model on a set of files"""
    print(f"\nEvaluating model on {name} data ({len(files)} files)...")
    
    if not files:
        print(f"No {name} files to evaluate.")
        return None
    
    # Process a subset of each file
    all_predictions = []
    all_true_labels = []
    samples_processed = 0
    
    # Define specific dtypes for columns with mixed types
    dtype_dict = {
        'sport': 'str',
        'dport': 'str',
    }
    
    for file in files:
        try:
            print(f"Processing {name} file: {os.path.basename(file)}...")
            
            # Read file in chunks
            chunk_reader = pd.read_csv(file, chunksize=CHUNK_SIZE, dtype=dtype_dict, low_memory=False)
            
            for chunk in chunk_reader:
                # Process the chunk
                processed_chunk = preprocess_chunk(chunk)
                
                if 'attack' in processed_chunk.columns:
                    # Extract features and true labels
                    X = processed_chunk.drop(['pkSeqID', 'category', 'subcategory', 'attack'], axis=1, errors='ignore')
                    y = processed_chunk['attack']
                    
                    # Transform features
                    X_transformed = preprocessor.transform(X)
                    
                    # Make predictions
                    y_pred_prob = model.predict(X_transformed, verbose=0)
                    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                    
                    # Store results
                    all_predictions.extend(y_pred)
                    all_true_labels.extend(y)
                    
                    # Update counter
                    samples_processed += len(y)
                    print(f"  Processed {samples_processed:,} samples...")
                    
                    # Check if we've processed enough samples
                    if samples_processed >= sample_limit:
                        break
                        
                else:
                    print(f"  Skipping file {os.path.basename(file)} - no 'attack' column found")
                
                # Break the file loop if we've processed enough samples
                if samples_processed >= sample_limit:
                    break
                    
        except Exception as e:
            print(f"Error processing {name} file {file}: {str(e)}")
            continue
            
        # Break the file loop if we've processed enough samples
        if samples_processed >= sample_limit:
            break
    
    # Calculate metrics if we have predictions
    if all_predictions and all_true_labels:
        accuracy = accuracy_score(all_true_labels, all_predictions)
        report = classification_report(all_true_labels, all_predictions)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        
        # Print results
        print(f"\n{name} Data Evaluation Results:")
        print(f"Total samples evaluated: {len(all_predictions):,}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name} Data')
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name.lower()}_{TIMESTAMP}.png"))
        
        return accuracy, report, conf_matrix
    else:
        print(f"No valid predictions could be made on {name} data.")
        return None

def plot_training_history(history):
    """Plot training and validation metrics"""
    print("Plotting training history...")
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"training_history_{TIMESTAMP}.png"))

def save_model_and_preprocessor(model, preprocessor, feature_names, X_train_data=None):
    """Save the model and preprocessing pipeline for later use, optimized for IoT deployment"""
    print("Saving model and preprocessor...")
    
    # Save the model in HDF5 format
    model_path = os.path.join(MODEL_DIR, f"ddos_model_final_{TIMESTAMP}.h5")
    model.save(model_path)
    
    try:
        # Save model in TensorFlow Lite format for IoT devices
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Use a simpler quantization approach that doesn't require representative dataset
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = os.path.join(MODEL_DIR, f"ddos_model_final_{TIMESTAMP}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        # Try advanced quantization if simpler approach worked
        if X_train_data is not None and hasattr(X_train_data, '__len__') and len(X_train_data) > 0:
            print("Applying int8 quantization with representative dataset...")
            advanced_converter = tf.lite.TFLiteConverter.from_keras_model(model)
            advanced_converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Define the representative dataset generator using the provided data
            def representative_dataset():
                # Convert sparse matrix to dense if needed
                if hasattr(X_train_data, 'todense'):
                    X_dense = X_train_data.todense()
                    for i in range(min(100, X_dense.shape[0])):
                        sample = np.array(X_dense[i:i+1]).astype(np.float32)
                        yield [sample]
                else:
                    for i in range(min(100, len(X_train_data))):
                        sample = np.array(X_train_data[i:i+1]).astype(np.float32)
                        yield [sample]
            
            # Set the representative dataset
            advanced_converter.representative_dataset = representative_dataset
            
            # Try with int8 quantization
            try:
                advanced_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                advanced_tflite_model = advanced_converter.convert()
                
                # Save the advanced TFLite model
                advanced_tflite_path = os.path.join(MODEL_DIR, f"ddos_model_final_int8_{TIMESTAMP}.tflite")
                with open(advanced_tflite_path, 'wb') as f:
                    f.write(advanced_tflite_model)
                    
                print(f"Int8 quantized model saved to: {advanced_tflite_path}")
                tflite_path = advanced_tflite_path  # Use the advanced path for size calculation
            except Exception as e:
                print(f"Int8 quantization failed: {str(e)}, using default quantization instead")
    except Exception as e:
        print(f"TFLite conversion error: {str(e)}")
        print("Saving original model only")
        tflite_path = None
    
    # Save preprocessor using joblib
    import joblib
    preprocessor_path = os.path.join(MODEL_DIR, f"preprocessor_{TIMESTAMP}.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save feature names
    with open(os.path.join(MODEL_DIR, f"feature_names_{TIMESTAMP}.txt"), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    # Calculate and print model sizes
    h5_size = os.path.getsize(model_path) / 1024
    print(f"Model saved to: {model_path} (Size: {h5_size:.1f} KB)")
    
    if tflite_path and os.path.exists(tflite_path):
        tflite_size = os.path.getsize(tflite_path) / 1024
        print(f"TFLite model saved to: {tflite_path} (Size: {tflite_size:.1f} KB)")
        print(f"Size reduction: {100 - (tflite_size/h5_size*100):.1f}%")
    
    print(f"Preprocessor saved to: {preprocessor_path}")

def main():
    """Main function to run the entire pipeline"""
    print("Starting IoT DDoS Neural Network Training Pipeline")
    
    # 1. Split both real and synthetic files into training and testing sets
    train_files, test_files, real_test_files, synthetic_test_files = split_train_test(
        REAL_DATA_DIR, 
        SYNTHETIC_DATA_DIR, 
        real_ratio=TRAIN_RATIO, 
        synthetic_ratio=SYNTHETIC_TRAIN_RATIO
    )
    
    # 2. Process training data in chunks to create a manageable dataset
    df_processed = process_data_in_chunks(train_files, sample_size=1000000)
    
    # 3. Prepare features and target
    X_train, X_val, y_train, y_val, feature_names, preprocessor, X_train_orig = prepare_features_and_target(df_processed)
    
    # 4. Build the model
    model = build_neural_network(X_train.shape[1])
    
    # 5. Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 6. Evaluate the model on validation data
    print("\nEvaluating on validation data (split from training data):")
    evaluate_model(model, X_val, y_val)
    
    # 7. Plot training history
    plot_training_history(history)
    
    # 8. Evaluate on all test files
    if test_files:
        evaluate_on_files(model, preprocessor, test_files, name="Combined Test")
    
    # 9. Evaluate separately on real test files
    if real_test_files:
        evaluate_on_files(model, preprocessor, real_test_files, name="Real Test")
    
    # 10. Evaluate separately on synthetic test files
    if synthetic_test_files:
        evaluate_on_files(model, preprocessor, synthetic_test_files, name="Synthetic Test")
    
    # 11. Save model and preprocessor
    save_model_and_preprocessor(model, preprocessor, feature_names, X_train)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()