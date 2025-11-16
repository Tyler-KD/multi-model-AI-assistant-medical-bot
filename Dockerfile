# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git .

# Create necessary directories for data persistence and uploads
RUN mkdir -p data/processed data/qdrantdb uploads/backend uploads/skin_lesion_output

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir unstructured[pdf]

# Expose the port that FastAPI will run on
EXPOSE 8000

# Volumes for data persistence
VOLUME ["/app/data/processed", "/app/data/qdrantdb", "/app/uploads"]

# Define healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Create entry point script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]