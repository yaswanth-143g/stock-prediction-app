# Production Dockerfile for NSE Stock Price Prediction & Quantitative Portfolio Platform
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=5000

# Install system dependencies & compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    libgomp1 \
    pkg-config \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir gunicorn eventlet gevent && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Create necessary runtime directories
RUN mkdir -p /app/logs /app/data /app/models /app/frontend/static/exports

# Expose HTTP port
EXPOSE 5000

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh 2>/dev/null || true

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Launch entrypoint
CMD ["/bin/bash", "/app/docker-entrypoint.sh"]
