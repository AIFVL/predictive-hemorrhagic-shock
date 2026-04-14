# Editor de Configuración del Pipeline

Interfaz visual construida con Streamlit para editar el archivo `pipeline_config.yaml` de forma segura y validada.

## Características

- Validación en tiempo real de todos los parámetros
- Interfaz intuitiva organizada por secciones
- Backup automático antes de guardar
- Visualización clara de todas las opciones
- Alertas y ayudas contextuales para usuarios no expertos
- Recarga en vivo de la configuración

## Instalación

Asegúrese de tener instalado Streamlit:

```bash
pip install streamlit pyyaml
```

O usando el requirements del proyecto principal.

## Uso

Desde el directorio raíz del proyecto, ejecute:

```bash
streamlit run apps/config_editor/app.py
```

La aplicación se abrirá en su navegador en `http://localhost:8501`

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

## Flujo de Trabajo Recomendado

1. Abrir la aplicación
2. Navegar por las secciones usando el menú lateral
3. Modificar los parámetros deseados
4. Verificar que no haya errores (marcadores rojos)
5. Guardar con el botón "Guardar Configuración"
6. Ejecutar el DAG en Airflow con la nueva configuración

## Notas Importantes

**Para usuarios no expertos:**

- Los campos con ayuda (icono de información) contienen información útil al pasar el mouse
- Los valores recomendados están indicados en las descripciones
- La configuración por defecto es un buen punto de partida
- No modifique parámetros avanzados sin consultar la documentación

**Consideraciones técnicas:**

- La búsqueda de hiperparámetros aumenta significativamente el tiempo de entrenamiento
- Un recall alto (>0.85) es importante en contexto médico pero puede reducir precisión
- La poda de features elimina variables con muy baja prevalencia
- El escalado de features es recomendado para regresión logística

## Tecnologías

- Streamlit: Framework de UI
- PyYAML: Parsing y serialización de YAML
- Python 3.12+: Lenguaje base
