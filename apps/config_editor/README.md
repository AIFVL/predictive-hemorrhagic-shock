# Editor de Configuración del Pipeline

Interfaz visual construida con Streamlit para editar el archivo `pipeline_config.yaml` de forma segura y validada.

## Características

- Validación en tiempo real de todos los parámetros
- Interfaz intuitiva organizada por secciones
- Backup automático antes de guardar
- Visualización clara de todas las opciones
- Alertas y ayudas contextuales para usuarios no expertos
- Recarga en vivo de la configuración

## Uso

El Config Editor se levanta automáticamente como parte del stack del proyecto cuando ejecutas:

```bash
# Desde el directorio raíz del proyecto
docker compose up -d
```

La aplicación estará disponible en **http://localhost:8501**

## Comandos Útiles

Todos los comandos se ejecutan desde el directorio raíz del proyecto:

```bash
# Ver logs del config editor
docker compose logs -f config-editor

# Reiniciar solo el config editor
docker compose restart config-editor

# Detener todo el stack
docker compose down

# Reconstruir el config editor (después de cambios en el código)
docker compose build config-editor
docker compose up -d
```

## Integración con el Proyecto

El Config Editor forma parte del docker-compose principal del proyecto y comparte:

- **Red Docker**: Puede comunicarse con otros servicios si es necesario
- **Volúmenes**: Accede directamente al archivo `config/pipeline_config.yaml`
- **Gestión unificada**: Se levanta y detiene junto con Airflow y PostgreSQL

### Servicios del Stack Completo

Cuando ejecutas `docker compose up -d` se levantan:

1. **postgres** (puerto 5432) - Base de datos de Airflow
2. **airflow-webserver** (puerto 8080) - Interfaz web de Airflow
3. **airflow-scheduler** - Planificador de DAGs
4. **config-editor** (puerto 8501) - Editor de configuración

## Estructura de la Interfaz

### Secciones Disponibles

1. **Configuración General**
   - Versiones del pipeline y dataset
   - Semilla aleatoria
   - Configuración de visualización

2. **Características (Features)**
   - Variable objetivo
   - Features numéricas y binarias
   - Features agregadas (habilitación/deshabilitación)
   - Features categóricas

3. **Limpieza y Validación**
   - Columnas a excluir
   - Rangos válidos para edad y hemoglobina
   - Reglas de validación

4. **Partición de Datos**
   - Tamaño del conjunto de prueba
   - Opciones de shuffle y estratificación

5. **Validación Cruzada**
   - Número de folds
   - Escalado de features
   - Paralelización

6. **Modelos**
   - Habilitación/deshabilitación de modelos
   - Configuración de hiperparámetros principales
   - Recall objetivo por modelo

7. **Búsqueda de Hiperparámetros**
   - Activación de búsqueda automática
   - Configuración de iteraciones y CV

8. **Optimización de Umbral**
   - Rangos de búsqueda
   - Umbrales para comparación en test
   - Permutaciones para test estadístico

9. **Poda de Features**
   - Habilitación de poda automática
   - Umbral mínimo de prevalencia

10. **Logging**
    - Nivel de detalle de logs
    - Directorio de salida

## Validaciones Implementadas

La aplicación incluye validaciones para:

- Formato de versiones (v1, v2, etc.)
- Rangos de edad (0-150 años)
- Rangos de hemoglobina (0-30 g/dL)
- Tamaño de test set (0-1)
- Número de folds de CV (2-20)
- Umbrales de clasificación (0-1)
- Recall objetivo (0.5-1.0)
- Coherencia entre parámetros relacionados

## Seguridad

- Se crea un backup automático (`.yaml.backup`) antes de cada guardado
- Las validaciones previenen configuraciones inválidas
- No se permite guardar si hay errores de validación
- El archivo de configuración se monta con permisos de lectura/escritura

## Flujo de Trabajo Recomendado

1. Asegúrate de que el stack está corriendo: `docker compose ps`
2. Abrir `http://localhost:8501` en el navegador
3. Navegar por las secciones usando el menú lateral
4. Modificar los parámetros deseados
5. Verificar que no haya errores (marcadores rojos)
6. Guardar con el botón "Guardar Configuración"
7. En Airflow (`http://localhost:8080`), ejecutar el DAG con la nueva configuración

## Troubleshooting

Para problemas comunes y sus soluciones, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Problemas más frecuentes:

- **Config Editor no inicia**: Verificar logs con `docker compose logs config-editor`
- **Puerto 8501 en uso**: Verificar que no haya otro servicio usando el puerto
- **Cambios no se guardan**: Verificar permisos del archivo `config/pipeline_config.yaml`
- **Aplicación lenta**: Aumentar recursos de Docker en configuración

Ver documentación completa de troubleshooting para más detalles.

## Configuración de Streamlit

El contenedor Docker viene pre-configurado con:

```toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
```

Esta configuración:
- Previene errores CORS
- Habilita protección XSRF
- Desactiva telemetría
- Optimiza para ejecución en contenedor

## Tecnologías

- Streamlit 1.32+: Framework de UI
- PyYAML 6.0+: Parsing y serialización de YAML
- Python 3.12: Lenguaje base
- Docker: Containerización y despliegue
