from flask import Flask, request, jsonify
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
    """Process input data based on feature mappings and types"""
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
    """Detect DDoS attacks in network traffic"""
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_size_bytes': os.path.getsize(MODEL_PATH)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
