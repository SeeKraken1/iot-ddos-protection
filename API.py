from flask import Flask, request, jsonify, redirect
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from flask_swagger_ui import get_swaggerui_blueprint
import os

app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = './models/ddos_model_final_int8_20250412_205328.tflite'
PREPROCESSOR_PATH = './models/preprocessor_20250412_205328.joblib'

# Create models directory if it doesn't exist
os.makedirs('./models', exist_ok=True)

# Initialize model and preprocessor loading
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

# Helper functions
def extract_ip_features(ip_address):
    """Extract numerical features from IP address"""
    try:
        if pd.isna(ip_address) or ip_address == '':
            return [0, 0, 0, 0]
        
        octets = str(ip_address).split('.')
        if len(octets) == 4:
            return [int(o) for o in octets]
        else:
            return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]

def preprocess_input(df):
    """Basic preprocessing of input data"""
    # Convert numerical features if needed
    numeric_columns = ['sport', 'dport', 'seq', 'stddev', 'min', 'mean', 'max', 'drate']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Fill missing values with 0
    df = df.fillna(0)
    
    return df

def extract_features(df):
    """Extract features for model prediction"""
    processed_df = df.copy()
    
    # Process IP addresses if they exist
    if 'saddr' in processed_df.columns:
        sip_features = [extract_ip_features(ip) for ip in processed_df['saddr']]
        processed_df['sip1'] = [f[0] for f in sip_features]
        processed_df['sip2'] = [f[1] for f in sip_features]
        processed_df['sip3'] = [f[2] for f in sip_features]
        processed_df['sip4'] = [f[3] for f in sip_features]
        processed_df.drop('saddr', axis=1, inplace=True)
    
    if 'daddr' in processed_df.columns:
        dip_features = [extract_ip_features(ip) for ip in processed_df['daddr']]
        processed_df['dip1'] = [f[0] for f in dip_features]
        processed_df['dip2'] = [f[1] for f in dip_features]
        processed_df['dip3'] = [f[2] for f in dip_features]
        processed_df['dip4'] = [f[3] for f in dip_features]
        processed_df.drop('daddr', axis=1, inplace=True)
    
    # Add computed features
    if 'sport' in processed_df.columns:
        processed_df['sport_is_well_known'] = (processed_df['sport'] < 1024).astype(int)
    
    if 'dport' in processed_df.columns:
        processed_df['dport_is_well_known'] = (processed_df['dport'] < 1024).astype(int)
    
    # Add ratio features if possible
    if 'min' in processed_df.columns and 'max' in processed_df.columns:
        processed_df['min_max_ratio'] = processed_df['min'] / processed_df['max'].replace(0, 1)
    
    # Convert categorical columns to strings
    if 'proto' in processed_df.columns:
        processed_df['proto'] = processed_df['proto'].astype(str)
    
    if 'state' in processed_df.columns:
        processed_df['state'] = processed_df['state'].astype(str)
        
    return processed_df

# Setup Swagger UI
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "DDoS Detection API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Create a static folder if it doesn't exist
os.makedirs('./static', exist_ok=True)

# Root route
@app.route('/', methods=['GET'])
def index():
    """Root route that provides basic UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DDoS Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 30px; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .card { background: #f8f9fa; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .btn { display: inline-block; background: #3498db; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; }
            code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            pre { background: #e9ecef; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DDoS Detection API</h1>
            <div class="card">
                <h2>Status: Online</h2>
                <p>The DDoS Detection Neural Network API is running and ready to accept requests.</p>
                <a href="/api/docs" class="btn">View API Documentation</a>
            </div>
            <div class="card">
                <h2>API Endpoints:</h2>
                <ul>
                    <li><code>GET /health</code> - Check API health status</li>
                    <li><code>POST /predict</code> - Predict if network traffic is a DDoS attack</li>
                    <li><code>GET /api/docs</code> - Interactive API documentation</li>
                </ul>
            </div>
            <div class="card">
                <h2>Example Usage:</h2>
                <pre><code>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "proto": "tcp",
    "saddr": "192.168.1.1",
    "sport": 80,
    "daddr": "192.168.1.2",
    "dport": 443,
    "seq": 12345,
    "stddev": 0.05,
    "min": 0.01,
    "mean": 0.5,
    "max": 1.0,
    "state": "CON",
    "drate": 0.002
  }'</code></pre>
            </div>
        </div>
    </body>
    </html>
    """

# API routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model_loaded})

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please check the server logs."}), 500
        
    try:
        # Get request data
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess input data
        processed_data = preprocess_input(df)
        
        # Extract features
        features = extract_features(processed_data)
        
        # Transform using preprocessor
        features_transformed = preprocessor.transform(features)
        
        # Make prediction using TFLite
        input_data = features_transformed.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], [input_data])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Convert prediction to binary result
        is_attack = bool(prediction[0][0] > 0.5)
        
        return jsonify({
            "prediction": "attack" if is_attack else "normal",
            "confidence": float(prediction[0][0]),
            "input_received": data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/static/swagger.json')
def swagger_json():
    return jsonify({
        "swagger": "2.0",
        "info": {
            "title": "DDoS Detection API",
            "description": "API for detecting DDoS attacks using machine learning",
            "version": "1.0.0"
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Predict if network traffic is a DDoS attack",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "description": "Network traffic data",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "proto": {"type": "string"},
                                    "saddr": {"type": "string"},
                                    "sport": {"type": "number"},
                                    "daddr": {"type": "string"},
                                    "dport": {"type": "number"},
                                    "seq": {"type": "number"},
                                    "stddev": {"type": "number"},
                                    "min": {"type": "number"},
                                    "mean": {"type": "number"},
                                    "max": {"type": "number"},
                                    "state": {"type": "string"},
                                    "drate": {"type": "number"}
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "prediction": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "input_received": {"type": "object"}
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "produces": ["application/json"],
                    "responses": {
                        "200": {
                            "description": "API health status",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "model_loaded": {"type": "boolean"}
                                }
                            }
                        }
                    }
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)