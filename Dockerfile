FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install missing system libraries
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create specific directories
RUN mkdir -p /app/data /app/mlruns

# Copy application files (volume mounted during dev usually, but good for standalone)
COPY src/ /app/src/
COPY run_pipeline.py /app/

CMD ["python", "run_pipeline.py"]
