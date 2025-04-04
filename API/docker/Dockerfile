FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and API code
COPY model/ model/
COPY api/ api/
 
# Environment variables
ENV MODEL_PATH="/app/model/iot_ddos_rf_model.pkl"
ENV MAPPINGS_PATH="/app/model/feature_mappings.pkl"
ENV PORT=5000

# Expose the API port
EXPOSE 5000

# Run the application
CMD ["python", "api/app.py"]
