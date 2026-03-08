#!/bin/bash

PATHS=(
"$(pwd)/config"
"$(pwd)/logs"
"$(pwd)/data"
"$(pwd)/src"
)

# Airflow container UID (from docker-compose.yml)
AIRFLOW_UID=50000

for DIR in "${PATHS[@]}"; do
  if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
  fi
  # Set ownership and permissions
  sudo chown -R $AIRFLOW_UID:0 "$DIR"
  sudo chmod -R 777 "$DIR"
done

# Start Docker Compose
docker compose up -d
