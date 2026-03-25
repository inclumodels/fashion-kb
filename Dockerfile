FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Pin numpy first to avoid conflicts
RUN pip install --no-cache-dir "numpy==1.26.4"

# Install CPU-only torch (much smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create data directory for LanceDB
RUN mkdir -p /data/lancedb

EXPOSE 8000

CMD ["python", "main.py"]
