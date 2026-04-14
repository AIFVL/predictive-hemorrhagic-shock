# Guía de Usuario - Editor de Configuración

## Introducción

Esta aplicación permite configurar el pipeline de predicción de shock hemorrágico sin necesidad de editar manualmente archivos de código. Está diseñada para ser usada por personas sin experiencia técnica.

## Inicio Rápido

### 1. Abrir la Aplicación

**En Windows:**
- Haz doble clic en `start_app.bat`
- Se abrirá automáticamente en tu navegador

**En Linux/Mac:**
- Abre una terminal
- Navega a la carpeta: `cd apps/config_editor`
- Ejecuta: `./start_app.sh`

### 2. Navegación

En el lado izquierdo verás un menú con las diferentes secciones:

- General: Configuración básica del proyecto
- Características: Variables que usa el modelo
- Limpieza: Validación de datos
- Partición: Cómo dividir los datos
- Validación Cruzada: Configuración de entrenamiento
- Modelos: Activar/desactivar modelos y ajustar parámetros
- Búsqueda HP: Búsqueda automática de mejores parámetros
- Optimización Umbral: Ajuste de sensibilidad del modelo
- Poda Features: Eliminación automática de variables poco útiles
- Logging: Nivel de detalle en los logs

### 3. Modificar Configuración

1. Selecciona una sección del menú
2. Modifica los valores que desees
3. Si hay un error, aparecerá un mensaje en rojo
4. Corrige los errores antes de guardar

### 4. Guardar Cambios

1. Haz clic en "Guardar Configuración" en el menú lateral
2. Verás un mensaje de éxito si todo salió bien
3. Se crea automáticamente un backup del archivo anterior

## Explicación de Secciones Principales

### Configuración General

**Qué es:** Parámetros básicos del proyecto.

**Campos importantes:**
- **Versión del Pipeline**: Identifica esta ejecución (ejemplo: v1, v2, v3)
  - Cambia este valor si quieres separar los resultados de diferentes experimentos
- **Versión del Dataset**: Identifica qué datos usar (ejemplo: v1)
  - Cambia solo si tienes diferentes versiones de los datos
- **Semilla Aleatoria**: Número para reproducir resultados (deja en 42 si no sabes)

### Características (Features)

**Qué es:** Las variables que el modelo usará para hacer predicciones.

**Tipos de variables:**
- **Numéricas**: Valores continuos como edad, hemoglobina
- **Binarias**: Valores de 0 o 1 (presente/ausente) como diabetes, hipertensión
- **Agregadas**: Variables calculadas combinando otras (puedes activarlas/desactivarlas)
- **Categóricas**: Variables discretizadas por umbrales (puedes activarlas/desactivarlas)

**Recomendaciones:**
- No modifiques las variables numéricas y binarias a menos que cambies los datos de entrada
- Experimenta activando/desactivando features agregadas y categóricas
- Más features no siempre significa mejor modelo

### Limpieza y Validación

**Qué es:** Reglas para verificar que los datos sean correctos.

**Campos importantes:**
- **Edad Mínima/Máxima**: Rango válido de edades (ejemplo: 18-120 años)
- **Hemoglobina Min/Max**: Rango válido de hemoglobina (ejemplo: 3-20 g/dL)

**Recomendaciones:**
- Estos valores deben reflejar lo que es médicamente posible
- Si hay valores fuera de rango, se detectarán como errores

### Partición de Datos

**Qué es:** Cómo dividir los datos entre entrenamiento y prueba.

**Campos importantes:**
- **Proporción del Conjunto de Prueba**: Qué porcentaje usar para probar (recomendado: 0.2 = 20%)
- **Partición Estratificada**: Mantener la misma proporción de casos positivos/negativos (recomendado: activado)

### Validación Cruzada

**Qué es:** Técnica para evaluar el modelo de forma más robusta.

**Campos importantes:**
- **Número de Folds**: Cuántas particiones hacer (recomendado: 10)
  - Más folds = evaluación más precisa pero más lento
- **Escalar Features**: Normalizar variables (recomendado: activado)

### Modelos

**Qué es:** Los algoritmos que aprenderán a predecir shock hemorrágico.

**Modelos disponibles:**
- **LightGBM**: El más potente, recomendado para mejor rendimiento
- **Árbol de Decisión**: Simple e interpretable
- **Regresión Logística**: Clásico, buena línea base
- **Naive Bayes**: Rápido, probabilístico

**Para cada modelo puedes:**
- Activarlo/desactivarlo
- Ajustar el **Recall Objetivo**: Sensibilidad mínima deseada
  - Valores altos (>0.85) son importantes en medicina para no perder casos
  - Puede reducir la precisión (más falsos positivos)

**Parámetros avanzados:**
- Si no tienes experiencia, usa los valores por defecto
- Si quieres experimentar:
  - **N° Estimadores** (LightGBM): Más = mejor pero más lento
  - **Learning Rate**: Más bajo = aprendizaje más cuidadoso
  - **Profundidad Máxima**: Más = modelo más complejo

### Búsqueda de Hiperparámetros

**Qué es:** Búsqueda automática de los mejores parámetros para cada modelo.

**Advertencia:** Esta opción aumenta mucho el tiempo de ejecución (horas en lugar de minutos).

**Cuándo usar:**
- Cuando quieras el mejor modelo posible
- Cuando tengas tiempo y recursos computacionales
- Para experimentos finales

**Cuándo NO usar:**
- Para pruebas rápidas
- Si tienes tiempo limitado
- Si no tienes experiencia (los valores por defecto son buenos)

### Optimización de Umbral

**Qué es:** Ajuste del punto de corte para clasificar como shock positivo/negativo.

**Concepto:**
- El modelo da una probabilidad (0-1)
- Necesitamos un umbral para decidir: positivo o negativo
- Umbral bajo = más sensible (detecta más casos) pero menos preciso
- Umbral alto = más específico (menos falsos positivos) pero puede perder casos

**Recomendaciones:**
- En medicina, es mejor ser sensible (no perder casos)
- Usa los valores por defecto a menos que tengas requisitos específicos

### Poda de Features

**Qué es:** Eliminación automática de variables muy raras.

**Por qué es útil:**
- Variables con muy pocos casos positivos pueden causar problemas
- Simplifica el modelo
- Mejora la generalización

**Recomendaciones:**
- Déjalo activado con el valor por defecto (10)
- Aumenta el valor si tienes muchos datos
- Reduce si tienes pocos datos

### Logging

**Qué es:** Nivel de detalle en los registros de ejecución.

**Niveles:**
- **DEBUG**: Máximo detalle (solo para desarrolladores)
- **INFO**: Balance detalle/claridad (recomendado)
- **WARNING**: Solo advertencias y errores
- **ERROR**: Solo errores

## Preguntas Frecuentes

### Qué pasa si cometo un error?

No te preocupes, la aplicación:
1. Te avisa con mensajes en rojo si algo está mal
2. No te deja guardar si hay errores
3. Crea un backup automático antes de guardar

### Cómo recupero la configuración anterior?

Si guardaste cambios que no querías:
1. Ve a la carpeta `config/`
2. Verás un archivo `pipeline_config.yaml.backup`
3. Ese es tu backup más reciente
4. Cópialo sobre `pipeline_config.yaml`

O en la aplicación:
1. Haz clic en "Recargar desde Archivo"
2. Los cambios no guardados se perderán

### Debo modificar todo?

No, solo modifica lo que necesites:
- Para empezar, solo activa/desactiva modelos
- Experimenta con features agregadas y categóricas
- Los valores por defecto son un buen punto de partida

### Cómo sé si mi configuración es buena?

1. Guarda la configuración
2. Ejecuta el pipeline en Airflow
3. Revisa las métricas de evaluación
4. Compara diferentes versiones

### Qué configuración usar para mi primer experimento?

Recomendaciones para principiantes:
- Versión: `v1`
- Modelos: Solo LightGBM activado
- Búsqueda de hiperparámetros: Desactivada
- Features agregadas: Prueba activar `N_COMORB` y `CARDIO_RISK`
- Resto: Valores por defecto

## Flujo de Trabajo Recomendado

### Experimento Básico
1. Abre la aplicación
2. Cambia la versión a `v2` (para no sobrescribir resultados anteriores)
3. Activa 2-3 features agregadas
4. Guarda
5. Ejecuta el DAG en Airflow
6. Revisa resultados

### Experimento Avanzado
1. Copia la configuración que funcionó mejor
2. Cambia la versión
3. Activa búsqueda de hiperparámetros
4. Aumenta el recall objetivo
5. Guarda y ejecuta
6. Compara con versiones anteriores

## Soporte Técnico

Si encuentras problemas:
1. Revisa que no haya mensajes en rojo en la aplicación
2. Verifica que el archivo `pipeline_config.yaml` existe en `config/`
3. Consulta los logs de Airflow si el DAG falla
4. Contacta al equipo técnico con:
   - Captura de pantalla del error
   - Versión que estabas configurando
   - Pasos que seguiste

## Consejos Finales

**Recomendaciones:**
- Cambia la versión entre experimentos
- Prueba diferentes combinaciones de features
- Anota qué configuración probaste y sus resultados
- Guarda frecuentemente

**Evitar:**
- Modificar parámetros sin entender qué hacen
- Usar la misma versión para diferentes experimentos
- Activar búsqueda de hiperparámetros sin necesidad
- Olvidar guardar antes de cerrar
