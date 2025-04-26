import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import altair as alt

API_URL = "http://localhost:5000/predict"  # Flask API endpoint: the backend API is running locally on our machine
SAMPLE_POINTS = 10               # time steps to simulate
CALL_DELAY = 0.1                           # API call delay

st.set_page_config(page_title="DDoS Detection Dashboard", layout="wide")

# menu in sidebar
with st.sidebar:
    st.title("Actions:")
    selected_page = st.radio("", (
        "Dashboard Visualization",
        "Submit Custom Traffic",
        "Model Performance Metrics",
        "Detection Methodology"
    ))
    st.markdown("---")
    st.caption("IoT DDoS Protection System")

# dashboard visualization 
if selected_page == "Dashboard Visualization":
    st.title("🛡️ DDoS Detection Dashboard")
    st.markdown("Simulating network traffic and highlighting DDoS detection results.")

    def generate_sample_data(n):
        rng = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='T')
        data = pd.DataFrame({
            "timestamp": rng,
            "saddr": [f"192.168.1.{i%255}" for i in range(n)],
            "daddr": [f"10.0.0.{i%255}" for i in range(n)],
            "sport": np.random.choice([80, 443, 8080], size=n),
            "dport": np.random.choice([80, 443, 8080], size=n),
            "proto": ["tcp"] * n,
            "state": ["CON"] * n,
            "seq": np.arange(n) * 100,
            "stddev": np.random.rand(n) * 0.1,
            "min": np.random.rand(n),
            "mean": np.random.rand(n),
            "max": np.random.rand(n) + 0.5,
            "drate": np.random.rand(n) * 0.01,
            "packets": np.random.poisson(lam=200, size=n),
        }).set_index("timestamp")

        attack_indices = np.random.choice(n, size=int(0.15 * n), replace=False)
        data.loc[data.index[attack_indices], "packets"] *= 3  # spike packets
        data["is_attack"] = False
        data.loc[data.index[attack_indices], "is_attack"] = True
        return data

    df = generate_sample_data(SAMPLE_POINTS)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Run Detection"):
            with st.spinner("Running detection..."):
                st.session_state.predictions = []
                for _, row in df.iterrows():
                    payload = row.drop("is_attack").to_dict()
                    try:
                        resp = requests.post(API_URL, json=payload)
                        status = resp.json().get("prediction", "normal")
                    except Exception:
                        status = "normal"
                    st.session_state.predictions.append(status == "attack")
                    time.sleep(CALL_DELAY)

    if "predictions" in st.session_state and len(st.session_state.predictions) == len(df):
        df["is_attack"] = st.session_state.predictions

    with col2:
        st.subheader("Traffic Flow")
        st.line_chart(df[["packets", "drate"]])

    st.subheader("Attack Highlights")
    attack_chart = (
        alt.Chart(df.reset_index())
        .mark_circle(size=80)
        .encode(
            x=alt.X("timestamp:T", title="Time", axis=alt.Axis(format='%H:%M')),
            y=alt.Y("packets:Q", title="Packet Count"),
            color=alt.condition(
                alt.datum.is_attack,
                alt.value("red"),
                alt.value("steelblue")
            ),
            tooltip=["timestamp:T", "packets", "drate", "is_attack"]
        )
        .properties(width=800, height=300)
    )
    st.altair_chart(attack_chart, use_container_width=True)

    st.download_button(
        "⬇️ Download Results as CSV",
        df.reset_index().to_csv(index=False).encode(),
        file_name="ddos_detection_results.csv",
        mime="text/csv"
    )

#add new data and predict if attack
elif selected_page == "Submit Custom Traffic":
    st.title("Submit Custom Network Traffic Data")
    st.markdown("Upload your CSV/JSON file or enter feature data manually.")

    option = st.radio("Choose input method:", ("Upload File", "Manual Input"))

    if option == "Upload File":
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    data = pd.read_csv(uploaded)
                else:
                    data = pd.read_json(uploaded)
                st.dataframe(data.head())
                if st.button("Submit for Detection"):
                    results = []
                    for _, row in data.iterrows():
                        try:
                            res = requests.post(API_URL, json=row.to_dict())
                            results.append(res.json().get("prediction"))
                        except Exception:
                            results.append("error")
                    st.success(f"Predictions: {results}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    else:
        with st.form("manual_input_form"):
            feature_cols = ["saddr", "daddr", "sport", "dport", "proto", "state", "seq", "stddev", "min", "mean", "max", "drate"]
            values = {col: st.text_input(f"{col}") for col in feature_cols}
            submit_btn = st.form_submit_button("Submit")
        if submit_btn:
            try:
                response = requests.post(API_URL, json=values)
                prediction = response.json().get("prediction")
                st.success(f"Attack Detected: {'Yes' if prediction == 'attack' else 'No'}")
            except Exception as e:
                st.error(f"API call failed: {e}")

# check the model metrics 
elif selected_page == "Model Performance Metrics":
    st.title("Model Performance Metrics")
    perf_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Standard Model": [0.9999, 1.00, 0.9999, 0.9999],
        "Optimized Model": [0.99, 1.00, 0.9999, 0.9999]
    } )
    st.table(perf_df.set_index("Metric"))
    perf_melt = perf_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    perf_chart = (
        alt.Chart(perf_melt)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y("Score:Q", title="Score"),
            color="Model:N",
            column="Model:N"
        )
        .properties(height=300)
    )
    st.altair_chart(perf_chart, use_container_width=True)

# info about our detection methodology
elif selected_page == "Detection Methodology":
    st.title("Detection Methodology")

    st.markdown(
        """
        ### 🛡️ IoT DDoS Protection System

        A lightweight neural network-based solution for detecting Distributed Denial of Service (DDoS) attacks on Internet of Things (IoT) devices. Designed for efficiency in resource-constrained environments, this system processes network traffic data and deploys a trained model through a containerized API.

        
        ### Project Objectives

        - Clean and preprocess raw traffic data from real-world and synthetic sources  
        - Engineer relevant network features to represent DDoS behavior  
        - Train a compact neural network for accurate and efficient classification  
        - Optimize the model for edge devices using TensorFlow Lite  
        - Enable integration via a portable, containerized API

        
        ### Data Processing Pipeline

        1. **Raw Data Cleaning**  
           - Processes CSV files from the UNSW_2018_IoT_Botnet_Dataset  
           - Adds headers, removes empty columns

        2. **Feature Extraction & Transformation**  
           - Protocol types  
           - Port numbers  
           - Packet sequence data  
           - Statistical features (mean, stddev, min, max)  
           - IP address components  
           - Traffic rate metrics

        
        ### Neural Network Architecture

        Optimized for IoT devices, the model uses a lightweight design:

        - **Input Layer**: 37 input features  
        - **Dense Layer** (32 units, ReLU)  
        - **Batch Normalization + Dropout (20%)**  
        - **Dense Layer** (16 units, ReLU)  
        - **Batch Normalization**  
        - **Output Layer**: 1 unit with Sigmoid activation  

        **Total Parameters**: 1,953  
        - Trainable: 1,857  
        - Non-trainable: 96

    
        ### Training Strategy

        A hybrid dataset was used to improve robustness:

        - **Real traffic**: 80% used for training  
        - **Synthetic traffic**: 70% used for training  
        - Remaining data reserved for validation and testing  
        - Covers both known and novel DDoS attack patterns

        
        ### Model Variants

        1. **Standard Model (.h5)**  
           - Full precision  
           - ~15KB  
           - Suitable for server-side deployment  

        2. **Optimized Model (.tflite)**  
           - Int8 quantized  
           - ~4KB (73% smaller)  
           - Minimal memory and compute usage  
           - Ideal for deployment on edge and IoT devices  
           - Maintains accuracy with reduced inference time
        
        """
    )
