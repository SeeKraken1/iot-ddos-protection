FROM tensorflow/tensorflow:2.8.0

# Set working directory
WORKDIR /app

# Upgrade pip and install retry-capable pip tools
RUN pip install --upgrade pip && \
    pip install pip-tools

# Install packages one at a time with increased timeout and retries
RUN pip install --no-cache-dir --timeout=120 --retries=10 flask==2.0.1 && \
    pip install --no-cache-dir --timeout=120 --retries=10 werkzeug==2.0.2 && \
    pip install --no-cache-dir --timeout=300 --retries=10 pandas==1.3.3 && \
    pip install --no-cache-dir --timeout=300 --retries=10 joblib==1.1.0 && \
    pip install --no-cache-dir --timeout=120 --retries=10 flask-swagger-ui==3.36.0 && \
    pip install --no-cache-dir --timeout=120 --retries=10 gunicorn==20.1.0 && \
    pip install --no-cache-dir --timeout=300 --retries=10 scikit-learn==1.0.1

# Create necessary directories
RUN mkdir -p /app/models /app/static

# Copy model files
COPY ./Models/ddos_model_final_int8_20250412_205328.tflite /app/models/
COPY ./Models/preprocessor_20250412_205328.joblib /app/models/

# Copy application code
COPY API.py /app/api.py

# Make port 5000 available
EXPOSE 5000

# Use python for direct execution
CMD ["python", "/app/api.py"]
