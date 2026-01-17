#!/bin/bash

# Get absolute path to config folder
CONFIG_PATH="$(pwd)/infrastructure/airflow/config"

# Airflow container UID (from docker-compose.yml)
AIRFLOW_UID=50000

# Set ownership and permissions
sudo chown -R $AIRFLOW_UID:0 "$CONFIG_PATH"
sudo chmod -R 775 "$CONFIG_PATH"

# Start Docker Compose
docker compose up -d
