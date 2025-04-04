
# IoT DDoS Detection Project

## Overview
This repository implements a system for detecting Distributed Denial-of-Service (DDoS) attacks in IoT environments using data from the **UNSW BoT-IoT 2018** dataset. The system employs a machine learning model to classify network traffic as benign or malicious. **This repository focuses on**:
1. **Data Preprocessing** (cleaning and structuring raw UNSW BoT-IoT 2018 data),
2. **Model Training** (Random Forest classifier for DDoS detection),
3. **Model Evaluation** (analyzing performance metrics and model size),
4. **Dockerization** (building a Docker image for easy deployment),
5. **REST API** (Flask-based inference service).

## Repository Structure

```
iot-ddos-protection/
├── api/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   ├── app.py
│   └── requirements.txt
├── model/
│   ├── model_results/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── inference_time.png
│   │   ├── model_size.png
│   │   └── roc_curve.png
│   ├── iot_ddos_rf_model.pkl
│   ├── categorical_mappings.pkl
│   └── feature_mappings.pkl
├── scripts/
│   ├── preprocessing.py
│   ├── repair.py
│   ├── RF_preprocessor.py
│   └── model.py
└── README.md
```

## Model Evaluation
The model evaluation results, including confusion matrix, ROC curve, feature importance, inference time, and model size comparison, are stored in the `model_results/` directory. These visualizations help in analyzing the model's accuracy, efficiency, and performance in IoT environments.

## Usage Instructions
### Data Preprocessing
Use the **preprocessing.py** and **repair.py** scripts to clean and preprocess the raw UNSW BoT-IoT data.
```
cd scripts
python preprocessing.py
python repair.py
python RF_preprocessor.py
```

### Model Training
Run the training script to build the Random Forest model.
```
cd scripts
python model.py
```

### API Deployment (Docker)
Build and run the Docker container for the inference API.
```
cd api/docker
docker build -t iot-ddos-api .
docker run -p 5000:5000 iot-ddos-api
docker-compose up --build
```

## API Endpoints
- **/api/health**: Returns model status and size.
- **/api/detect**: Detects DDoS attacks from incoming JSON traffic data.

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
- **UNSW BoT-IoT 2018**: We thank the creators for enabling research on IoT security.
