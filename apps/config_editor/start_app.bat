@echo off
REM Script para iniciar el editor de configuración en Windows

echo Iniciando Editor de Configuración del Pipeline...
echo ==================================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "..\..\config\pipeline_config.yaml" (
    echo Error: No se encuentra pipeline_config.yaml
    echo Ejecute este script desde apps\config_editor\
    pause
    exit /b 1
)

REM Instalar dependencias si es necesario
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Instalando dependencias...
    pip install -r requirements.txt
)

echo Listo!
echo.
echo Abriendo aplicación en el navegador...
echo URL: http://localhost:8501
echo.
echo Presione Ctrl+C para detener la aplicación
echo.

REM Iniciar Streamlit
streamlit run app.py
