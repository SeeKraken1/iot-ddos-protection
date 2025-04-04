import os
import shutil
import argparse

def create_api_structure(base_dir, model_path, mappings_path):
    """Create the API structure for the IoT DDoS detection service"""
    
    # Create directories
    directories = [
        os.path.join(base_dir, "api"),
        os.path.join(base_dir, "model")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Copy model files
    if os.path.exists(model_path):
        shutil.copy(model_path, os.path.join(base_dir, "model", "iot_ddos_rf_model.pkl"))
        print(f"Copied model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
    
    if os.path.exists(mappings_path):
        shutil.copy(mappings_path, os.path.join(base_dir, "model", "feature_mappings.pkl"))
        print(f"Copied mappings from {mappings_path}")
    else:
        print(f"Warning: Mappings file not found at {mappings_path}")
    
    # Create requirements.txt
    requirements_content = """scikit-learn==1.0.2
flask==2.0.1
numpy==1.22.0
pandas==1.3.5
gunicorn==20.1.0
"""
    
    with open(os.path.join(base_dir, "requirements.txt"), 'w') as f:
        f.write(requirements_content)
    print("Created requirements.txt")
    
    # Create API application file
    app_py_content = """from flask import Flask, request, jsonify
import pickle
import time
import os

app = Flask(__name__)

# Load model and mappings
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/iot_ddos_rf_model.pkl')
MAPPINGS_PATH = os.environ.get('MAPPINGS_PATH', 'model/feature_mappings.pkl')

print(f"Loading model from {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(f"Loading mappings from {MAPPINGS_PATH}")
with open(MAPPINGS_PATH, 'rb') as f:
    mappings = pickle.load(f)
    feature_mappings = mappings.get('value_mappings', {})
    feature_types = mappings.get('feature_types', {})

# Define feature names
FEATURE_NAMES = [
    "proto", "sport", "dport", "seq", "stddev", "min", "mean", "drate", 
    "state_number", "N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP"
]

def preprocess_input(data):
    \"\"\"Process input data based on feature mappings and types\"\"\"
    features = []
    for feature in FEATURE_NAMES:
        value = data.get(feature)
        
        # Clean the value if it's a string
        if isinstance(value, str):
            value = value.strip().lower()
        
        # Convert hex to int if possible
        if isinstance(value, str) and value.startswith('0x'):
            try:
                value = int(value, 16)
            except ValueError:
                pass
                
        # Handle based on feature type
        if feature_types.get(feature) == 'categorical':
            # Categorical feature
            mapping = feature_mappings.get(feature, {})
            if isinstance(value, str) and value in mapping:
                features.append(mapping[value])
            elif isinstance(value, (int, float)):
                features.append(value)
            else:
                features.append(-1)  # Default for unknown
        else:
            # Numeric feature
            if value is None:
                features.append(0)
            elif isinstance(value, (int, float)):
                features.append(value)
            else:
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    features.append(0)
    
    return features

@app.route('/api/detect', methods=['POST'])
def detect_attack():
    \"\"\"Detect DDoS attacks in network traffic\"\"\"
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Preprocess and predict
    features = preprocess_input(data)
    
    start_time = time.time()
    prediction = model.predict([features])[0]
    inference_time = time.time() - start_time
    
    # Get probability if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba([features])[0][1]
    
    return jsonify({
        'is_attack': bool(prediction),
        'probability': float(probability) if probability is not None else None,
        'inference_time_ms': inference_time * 1000,
        'model_version': '1.0.0'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    \"\"\"Health check endpoint\"\"\"
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_size_bytes': os.path.getsize(MODEL_PATH)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
"""
    
    with open(os.path.join(base_dir, "api", "app.py"), 'w') as f:
        f.write(app_py_content)
    print("Created API application (api/app.py)")
    
    print("\nAPI structure created successfully!")
    print(f"\nTo test the API locally:")
    print(f"1. Navigate to {base_dir}")
    print(f"2. Run: pip install -r requirements.txt")
    print(f"3. Run: python api/app.py")
    print(f"4. Access the API at: http://localhost:5000/api/health")

def main():
    parser = argparse.ArgumentParser(description='Create API structure for IoT DDoS detection')
    parser.add_argument('--output', default='D:/EECE 490 project/API', help='Output directory for the project')
    parser.add_argument('--model', default='D:/EECE 490 project/Model/iot_ddos_rf_model.pkl', help='Path to trained model file')
    parser.add_argument('--mappings', default='D:/EECE 490 project/Model/feature_mappings.pkl', help='Path to feature mappings file')
    
    args = parser.parse_args()
    
    create_api_structure(args.output, args.model, args.mappings)

if __name__ == "__main__":
    main()