# IoT DDoS Detection Project

## Overview
This repository implements a system for detecting Distributed Denial-of-Service (DDoS) attacks in IoT environments using data from the **UNSW BoT-IoT 2018** dataset. The system employs a machine learning model to classify network traffic as benign or malicious. **This repository focuses on**:
1. **Data Preprocessing** (cleaning and structuring raw UNSW BoT-IoT 2018 data),
2. **Model Training** (Random Forest classifier for DDoS detection),
3. **Dockerization** (building a Docker image for easy deployment),
4. **REST API** (Flask-based inference service).

## Repository Structure

```
my-iot-ddos-project/
├── api/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   ├── app.py
│   └── requirements.txt
├── data/
├── docs/
│   └── Project_Announcement.pdf
├── model/
│   ├── iot_ddos_rf_model.pkl
│   └── feature_mappings.pkl
├── scripts/
│   ├── preprocessing.py
│   ├── repair.py
│   ├── RF_preprocessor.py
│   └── model.py
└── README.md
```

- **api/**  
  Contains the Flask application (`app.py`) for inference, plus Docker files in `docker/`.
- **data/**  
  (Optional) May contain references or partial subsets of the UNSW BoT-IoT 2018 dataset.
- **docs/**  
  Includes supplementary documentation such as the `Project_Announcement.pdf`.
- **model/**  
  Holds the trained Random Forest model (`iot_ddos_rf_model.pkl`) and feature mappings (`feature_mappings.pkl`).
- **scripts/**  
  Contains the Python scripts for data preprocessing and training:
  - **preprocessing.py**: Cleans & formats raw UNSW BoT-IoT CSV data.  
  - **repair.py**: Handles quote/parsing fixes for corrupted CSV files.  
  - **RF_preprocessor.py**: Feature selection/engineering steps for Random Forest training.  
  - **model.py**: Trains the Random Forest classifier, saving results to the `model/` folder.

## Data Preprocessing
We use the **UNSW BoT-IoT 2018 dataset** to train and evaluate our DDoS detection approach.

1. **preprocessing.py** & **repair.py**  
   - Automate CSV cleanup: removing empty columns, assigning headers, handling malformed rows, etc.

2. **RF_preprocessor.py**  
   - Refines or adds features (e.g., `state_number`, `N_IN_Conn_P_SrcIP`, `N_IN_Conn_P_DstIP`) to improve detection performance.

**Usage** (example commands):
```bash
cd scripts
python preprocessing.py
python repair.py
python RF_preprocessor.py
```

## Model Training
- **model.py** loads the processed data, extracts features, and trains a Random Forest classifier optimized for IoT devices.  
- The trained model (`iot_ddos_rf_model.pkl`) and supporting `feature_mappings.pkl` are saved in the `model/` directory.

**Usage**:
```bash
cd scripts
python model.py
```

## Docker & API
The `api/` folder contains:
- **`app.py`** — A Flask server with two endpoints:
  - **`/api/health`**: Basic health check (returns model status and size).  
  - **`/api/detect`**: Classifies incoming network traffic JSON as benign or malicious.

### Building & Running (Docker)
1. **Build the image**:
   ```bash
   cd api/docker
   docker build -t iot-ddos-api .
   ```
2. **Run the container**:
   ```bash
   docker run -p 5000:5000 iot-ddos-api
   ```
3. **Use docker-compose**:
   ```bash
   cd api/docker
   docker-compose up --build
   ```

## Notes & Considerations
- **UNSW BoT-IoT 2018 Dataset**: Ensure you have permission to use and redistribute the dataset.  
- **Data Volume**: Consider storing large datasets externally and linking in the README.  
- **Model Parameters**: Adjust Random Forest parameters in `model.py` to optimize for your environment.

## License
*(Insert your chosen license here)*

## Acknowledgments
- **UNSW BoT-IoT 2018**: We thank the creators for enabling research on IoT security.
