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

---

## Semana 14 (Primera semana de noviembre - 4 de noviembre)

**Actividades realizadas:**

* **Identificación de problema crítico en EDA**: Revisión exhaustiva de resultados del análisis exploratorio reveló que solo 1 de 17 variables categóricas (5.9%) mostraba asociación significativa con shock hemorrágico.
* **Análisis de limitaciones**: Se identificó que las variables individuales tienen poder predictivo muy bajo (Cramer's V máximo = 0.18), sugiriendo que el shock hemorrágico es un fenómeno multifactorial que requiere agregación de variables.
* **Diseño de estrategia de agregación**: Se desarrolló metodología de feature engineering basada en agrupaciones fisiopatológicas y evidencia clínica documentada.

**Decisiones técnicas:**

* **Creación de documento metodológico**: Se elaboró `doc/3.5_feature_aggregation.md` con justificación clínica detallada de cada variable agregada, incluyendo referencias bibliográficas a literatura médica (ATLS, Charlson Index, ASA Guidelines).
* **Implementación de módulo FeatureAggregator**: Se desarrolló `src/preprocessing/feature_aggregator.py` con clase que implementa 12 variables agregadas:
  - 6 índices de comorbilidades por sistemas (cardiovascular, respiratorio, metabólico, etc.)
  - 2 categorizaciones de variables continuas (edad, hemoglobina)
  - 3 variables de interacción
  - 1 score de riesgo compuesto

**Variables agregadas creadas:**

1. `CARGA_COMORBILIDADES`: Suma de 13 condiciones crónicas (rango 0-5 en dataset)
2. `RIESGO_CARDIOVASCULAR`: Índice de 4 condiciones cardiovasculares
3. `RIESGO_RESPIRATORIO`: Índice de 3 condiciones respiratorias
4. `RIESGO_METABOLICO`: Índice de 3 condiciones metabólicas (diabetes, obesidad, hipotiroidismo)
5. `INMUNOCOMPROMISO_CANCER`: Presencia de cáncer activo o inmunosupresión
6. `FACTORES_RIESGO_SANGRADO`: Sangrado mayor previo + tabaquismo
7. `CATEGORIA_EDAD`: Estratificación en 4 grupos de riesgo quirúrgico (18-44, 45-64, 65-74, ≥75)
8. `CATEGORIA_HB_PREQX`: Clasificación de anemia según criterios OMS (normal, leve, moderada, severa)
9. `EDAD_X_CARGA_COMORBILIDADES`: Interacción edad × comorbilidades (captura fragilidad)
10. `HB_INVERSA_X_COMORBILIDADES`: Interacción hemoglobina baja × comorbilidades
11. `SANGRADO_X_CV`: Interacción sangrado mayor × riesgo cardiovascular
12. `RISK_SCORE`: Score ponderado de riesgo compuesto

**Pipeline de EDA con agregación:**

* Se creó `src/pipelines/eda_aggregated_pipeline.py` que integra agregación + análisis exploratorio completo.
* Se generó script ejecutable `run_eda_aggregated_pipeline.py` en raíz del proyecto.
* Pipeline ejecutado exitosamente generando dataset agregado con 34 variables (22 originales + 12 agregadas).

**Resultados del EDA con variables agregadas:**

* **Mejora en poder predictivo**: 3 de 9 variables agregadas categóricas (33.3%) resultaron significativas vs 5.9% en dataset original.
* **Variables agregadas significativas**:
  - `SANGRADO_X_CV`: Cramer's V = 0.18, p < 0.001
  - `FACTORES_RIESGO_SANGRADO`: Cramer's V = 0.16, p < 0.001
  - `CATEGORIA_EDAD`: Cramer's V = 0.11, p < 0.001
  - `RISK_SCORE`: Cohen's d = 0.28, p < 0.001
* **Dataset resultante**: 1,324 pacientes × 34 variables guardado en `data/processed/shock_aggregated.csv`

**Hallazgos importantes:**

* La variable `FACTORES_RIESGO_SANGRADO` (que combina sangrado mayor previo + tabaquismo) mostró segunda mayor asociación con shock.
* La categorización de edad en grupos de riesgo quirúrgico capturó mejor la relación no lineal que la edad continua.
* El `RISK_SCORE` compuesto demostró efecto moderado (Cohen's d = 0.28), comparable a EDAD aislada.
* Variables de sistemas específicos (cardiovascular, respiratorio, metabólico) no mostraron significancia individual, sugiriendo que su efecto es capturado mejor por la carga total de comorbilidades.

**Documentación generada:**

* Reporte de EDA agregado: `reports/eda_aggregated/eda_aggregated_summary_report.md`
* Tablas comparativas: `reports/eda_aggregated/tables/comparison_original_vs_aggregated.csv`
* 12 visualizaciones en `reports/eda_aggregated/figures/`
* Resumen de features: `reports/eda_aggregated/tables/feature_aggregation_summary.csv`

**Observaciones metodológicas:**

* La estrategia de agregación siguió principios de CRISP-DM, manteniendo transparencia y reproducibilidad.
* Se priorizó interpretabilidad clínica sobre complejidad: todas las variables agregadas tienen significado claro para médicos.
* Se mantuvieron variables originales en dataset para permitir comparación en fase de modelado.
* La justificación de cada variable se respaldó con literatura médica peer-reviewed (15 referencias bibliográficas).

**Problemas identificados:**

* Aunque hubo mejora en porcentaje de variables significativas (33% vs 6%), el efecto sigue siendo modesto (Cramer's V < 0.20).
* Esto confirma que el shock hemorrágico es evento complejo que probablemente requiera modelos no lineales (random forest, XGBoost) para capturar interacciones.
* La colinealidad entre variables agregadas y originales deberá manejarse en fase de modelado.

**Próximos pasos (semana 15):**

* Validar variables agregadas con experto médico del equipo.
* Iniciar fase de modelado con dataset agregado.
* Comparar rendimiento de modelos usando variables originales vs agregadas.
* Implementar análisis SHAP para interpretabilidad de features.

**Reflexión del equipo:**

La agregación de variables fue crucial para superar la limitación de poder predictivo individual. Aunque los efectos siguen siendo modestos, el aumento de 5.9% a 33.3% en variables significativas es sustancial y proporciona base más sólida para modelado. La documentación exhaustiva asegura trazabilidad y reproducibilidad del proceso.
