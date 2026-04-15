# Extend official Airflow image
FROM apache/airflow:2.9.3-python3.11

# Switch to root to install system packages
USER root

# Install system dependencies required by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --retries 20 --timeout 120 -r /tmp/requirements.txt
