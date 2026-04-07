FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

# Download YOLO11m-seg model on build (cached in Docker layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"

CMD ["python", "-u", "processor.py"]
