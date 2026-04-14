FROM python:3.12-slim

LABEL maintainer="PDG Shock Hemorrágico"
LABEL description="Config Editor - Streamlit interface for pipeline_config.yaml"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Configurar Streamlit
RUN mkdir -p /root/.streamlit && \
    echo "[server]\n\
headless = true\n\
port = 8501\n\
address = \"0.0.0.0\"\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
serverAddress = \"localhost\"\n\
serverPort = 8501\n\
\n\
[theme]\n\
base = \"light\"" > /root/.streamlit/config.toml

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Comando de inicio
CMD ["streamlit", "run", "app.py"]
