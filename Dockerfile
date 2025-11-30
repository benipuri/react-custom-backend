# FINAL WORKING DOCKERFILE — RunPod Load Balancer + Auto-download SAM model
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies + wget for downloading model
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your code and small model
COPY app.py .

# Download official SAM ViT-B checkpoint (~300 MB) — this runs every build (15 seconds)
RUN wget -O sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# RunPod Load Balancer expects port 80
# EXPOSE 80

# Start Uvicorn on port 80 — this is the key fix
CMD ["python3", "app.py"]
