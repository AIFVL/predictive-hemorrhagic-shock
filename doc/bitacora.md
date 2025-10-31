# Bitácora de desarrollo del proyecto

**Periodo:** Agosto – Octubre 2025
**Integrantes:** Cristian Botina, Óscar Gómez, Juan Manuel Marín
**Proyecto:** Modelo predictivo de shock hemorrágico en pacientes de cirugía mayor no cardíaca
**Metodología:** CRISP-DM

---

## Semana 1-2 (Inicio del proyecto – Agosto)

**Actividades realizadas:**

* Revisión de los lineamientos del curso y definición general del problema clínico a abordar.
* Discusión sobre el enfoque metodológico y elección del marco CRISP-DM como base.
* Establecimiento del objetivo principal: construir un modelo predictivo de shock hemorrágico a partir de datos clínicos.
* Identificación de las fuentes de datos (Fundación Valle del Lili) y requisitos de privacidad.
* Realización del curso de *Good Clinical Practice (GCP)* para cumplimiento ético.

**Observaciones:**

* En este punto no teníamos aún el dataset, así que el trabajo fue principalmente conceptual y documental.
* Se elaboró el cronograma preliminar y se acordó que la documentación se llevaría en Word y Mendeley.
* Todos completamos el curso GCP y firmamos los consentimientos para manejo de datos clínicos.

---

## Semana 3-4 (Segunda mitad de agosto)

**Actividades:**

* Búsqueda bibliográfica inicial sobre shock hemorrágico y modelos clínicos existentes.
* Organización de referencias en Mendeley y definición de los primeros capítulos del marco teórico.
* Revisión de artículos recientes en PubMed y Scopus.
* Identificación de variables clínicas relevantes (hematocrito, presión arterial, frecuencia cardíaca, etc.).

**Problemas:**

* La información clínica estaba dispersa y algunos artículos eran de acceso restringido.
* Se tomó la decisión de resumir la información en una tabla comparativa con las variables más comunes en estudios similares.

---

## Semana 5-6 (Inicio de septiembre)

**Actividades:**

* Redacción del marco teórico inicial, estructurado en secciones (definición del shock, tipos, fisiopatología, variables clínicas).
* Análisis de metodologías aplicadas (TRIPOD, CRISP-DM, SHAP para interpretabilidad).
* Discusión sobre la arquitectura general del proyecto (pipelines, notebooks, versionamiento).

**Decisiones:**

* Se estableció GitHub como repositorio principal de código.
* Poetry se definió como gestor de dependencias.
* Se dividieron los roles principales: Cristian (investigación clínica y documentación), Óscar (EDA y preprocesamiento), Juan Manuel (modelado e interpretación).

---

## Semana 7-8 (Finales de septiembre)

**Actividades:**

* Se consolidó el anteproyecto con el marco teórico y la metodología.
* Se creó el cronograma detallado con fechas de entregas y actividades por fase.
* Revisión y entrega del primer documento parcial.

**Problemas:**

* Dificultades con la versión del dataset, ya que la entrega de la FVL se retrasó.
* El equipo ajustó las fechas del cronograma para mantener coherencia con la planificación académica.

---

## Semana 9-10 (Inicio de octubre)

**Actividades:**

* Se recibió el dataset y se comenzó la exploración inicial (EDA preliminar).
* Configuración del entorno: instalación de dependencias con Poetry y carga del dataset en un notebook Jupyter.
* Revisión de las variables, detección de valores faltantes y outliers.
* Elaboración de un script base de análisis exploratorio (EDA.ipynb).

**Problemas:**

* Se detectaron variables con valores nulos en más del 50% de los registros.
* Algunas columnas tenían nombres inconsistentes y hubo que renombrarlas para estandarizar el pipeline.

---

## Semana 11 (Segunda semana de octubre)

**Actividades:**

* Limpieza inicial de los datos: eliminación de columnas con varianza cero y normalización de nombres de variables.
* Exploración de correlaciones entre variables numéricas y categóricas.
* Pruebas con estadísticos básicos (chi-cuadrado, Mann-Whitney, t-test).
* Revisión de asociaciones significativas con la variable objetivo.

**Problemas:**

* La ejecución del notebook era lenta por el tamaño del dataset, así que se optó por trabajar con una muestra.
* Se ajustaron algunos gráficos para mejorar legibilidad (matrices de correlación y distribución).

---

## Semana 12 (Tercera semana de octubre)

**Actividades:**

* Implementación de una pipeline reproducible para el preprocesamiento (eliminación de nulos, codificación y escalado).
* Documentación del proceso en un README técnico.
* Revisión de dependencias y ajustes en el archivo `pyproject.toml`.
* Creación de scripts auxiliares para visualización y análisis de variables categóricas.

**Observaciones:**

* El flujo general del análisis quedó definido: carga → limpieza → EDA → feature engineering → modelado.
* El equipo comenzó a preparar las diapositivas para la presentación intermedia.

---

## Semana 13 (Última semana de octubre)

**Actividades:**

* Consolidación del EDA con análisis univariado, bivariado y multivariado.
* Generación de gráficos PCA y matrices de asociación.
* Preparación de la presentación 2 (31 de octubre) con resultados parciales.
* Avances en la documentación de la metodología en el anteproyecto.

**Problemas:**

* Falta de tiempo para ejecutar el test completo con todos los modelos planificados.
* Se priorizó la limpieza y el análisis exploratorio para mostrar resultados estables.

---

## Estado actual (30 de octubre)

**Avances consolidados:**

* Fases 1 y 2 de CRISP-DM completadas (comprensión del problema y comprensión/preparación de los datos).
* Anteproyecto en revisión con la sección metodológica completa.
* EDA funcional en Jupyter con resultados reproducibles.
* Cronograma y bitácora al día.

**Pendiente:**

* Iniciar formalmente la fase de modelado (fase 3 de CRISP-DM).
* Incorporar análisis de interpretabilidad (SHAP) y experimentos con modelos base.
* Ajustar documentación final e integrar métricas de validación.
