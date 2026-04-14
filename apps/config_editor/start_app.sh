#!/bin/bash
# Script para iniciar el editor de configuración

echo "Iniciando Editor de Configuración del Pipeline..."
echo "=================================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "../../config/pipeline_config.yaml" ]; then
    echo "Error: No se encuentra pipeline_config.yaml"
    echo "Ejecute este script desde apps/config_editor/"
    exit 1
fi

# Instalar dependencias si es necesario
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Instalando dependencias..."
    pip install -r requirements.txt
fi

echo "Listo!"
echo ""
echo "Abriendo aplicación en el navegador..."
echo "URL: http://localhost:8501"
echo ""
echo "Presione Ctrl+C para detener la aplicación"
echo ""

# Iniciar Streamlit
streamlit run app.py
