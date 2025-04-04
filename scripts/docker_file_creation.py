import os
import argparse

def create_docker_files(base_dir):
    """Create Docker-related files for deploying the IoT DDoS detection API"""
    
    # Create docker directory
    docker_dir = os.path.join(base_dir, "docker")
    os.makedirs(docker_dir, exist_ok=True)
    print(f"Created directory: {docker_dir}")
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.9-slim

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
"""
    
    with open(os.path.join(base_dir, "Dockerfile"), 'w') as f:
        f.write(dockerfile_content)
    print("Created Dockerfile")
    
    # Create docker-compose.yml
    docker_compose_content = """version: '3'

services:
  iot-ddos-api:
    build: .
    ports:
      - "5000:5000"
    restart: always
    environment:
      - PORT=5000
      - MODEL_PATH=/app/model/iot_ddos_rf_model.pkl
      - MAPPINGS_PATH=/app/model/feature_mappings.pkl
    volumes:
      - ./logs:/app/logs
"""
    
    with open(os.path.join(base_dir, "docker-compose.yml"), 'w') as f:
        f.write(docker_compose_content)
    print("Created docker-compose.yml")
    
    # Create .dockerignore
    dockerignore_content = """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
venv/
ENV/
.git
.gitignore
README.md
.DS_Store
logs/
"""
    
    with open(os.path.join(base_dir, ".dockerignore"), 'w') as f:
        f.write(dockerignore_content)
    print("Created .dockerignore")
create_docker_files("D:/EECE 490 project/API")       