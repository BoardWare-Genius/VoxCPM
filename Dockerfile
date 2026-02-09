FROM python:3.10.12-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Create app directory
WORKDIR /app
COPY api_concurrent.py requirements.txt ./
RUN pip install -r requirements.txt
ENV VOXCPM_MODEL_ID="/models/VoxCPM1.5/" \
    VOXCPM_CPU_WORKERS="2" \
    VOXCPM_UVICORN_WORKERS="1" \
    MAX_GPU_CONCURRENT="1"

EXPOSE 5000
CMD [ "python", "./api_concurrent.py" ]

