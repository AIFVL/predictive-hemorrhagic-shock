# Fase 1: Comprensión del Negocio

## Predicción Temprana de Shock Hemorrágico en Cirugía Mayor No Cardiaca

---

## 1.1 Contexto del Problema

El shock hemorrágico constituye una emergencia médica crítica caracterizada por una falla aguda del sistema circulatorio secundaria a una pérdida rápida del volumen sanguíneo, lo que compromete la perfusión tisular y la oxigenación de órganos vitales. En el contexto quirúrgico, específicamente en procedimientos de cirugía mayor no cardiaca, la detección temprana de esta condición resulta determinante para la supervivencia del paciente y la prevención de complicaciones sistémicas.

Los métodos tradicionales de detección se fundamentan en la evaluación clínica y el uso de índices predictivos como el Índice de Shock (IS) y el Índice de Shock Modificado (MSI). No obstante, estas herramientas presentan limitaciones inherentes: dependencia de la experiencia clínica individual, variabilidad interpretativa entre profesionales, y énfasis en manifestaciones tardías del shock cuando los mecanismos compensatorios fisiológicos han sido superados. Esta detección tardía incrementa significativamente la mortalidad (15.33% en pacientes con shock hemorrágico postoperatorio según la literatura), la necesidad de transfusiones masivas, el desarrollo de falla multiorgánica, y la saturación de recursos hospitalarios críticos.

## 1.2 Objetivos del Proyecto

### 1.2.1 Objetivo General

Desarrollar y validar un modelo de inteligencia artificial que permita la predicción temprana del riesgo de shock hemorrágico en pacientes sometidos a cirugía mayor no cardiaca, integrando datos clínicos preoperatorios e intraoperatorios, con el fin de apoyar la toma de decisiones médicas oportunas y reducir complicaciones asociadas.

### 1.2.2 Objetivos Específicos

1. **Construcción del conjunto de datos estructurado**: Consolidar un dataset a partir de registros clínicos preoperatorios e intraoperatorios de la Fundación Valle del Lili (FVL), mediante procesos rigurosos de limpieza, transformación y anonimización, garantizando la calidad, trazabilidad y conformidad con estándares de protección de datos sensibles.

2. **Definición de criterios diagnósticos**: Establecer criterios clínicos y operativos para el diagnóstico de shock hemorrágico, consensuados con expertos del área de anestesiología y medicina crítica, que permitan la etiquetación precisa de casos y controles en el dataset.

3. **Identificación de variables predictoras**: Determinar las variables clínicas, demográficas y laboratoriales con mayor capacidad discriminativa para la predicción de shock hemorrágico, utilizando métodos estadísticos y de selección de características.

4. **Entrenamiento y validación de modelos**: Entrenar y validar modelos supervisados de aprendizaje automático, evaluando su desempeño mediante métricas clínicamente relevantes (sensibilidad, especificidad, F1-score, AUC-ROC, calibración), con énfasis en la minimización de falsos negativos.

5. **Validación clínica inicial**: Diseñar y ejecutar una validación clínica preliminar mediante retroalimentación de expertos médicos o simulación de escenarios históricos, evaluando la utilidad y aplicabilidad del modelo en contextos reales.

6. **Estrategia de integración**: Proponer un plan de integración del modelo en sistemas de información quirúrgicos, considerando aspectos técnicos (interoperabilidad, latencia, disponibilidad), clínicos (flujo de trabajo, alertas, interpretabilidad) y de usabilidad (interfaz, capacitación).

## 1.3 Interesados del Proyecto

### 1.3.1 Interesados Clínicos

- **Dr. Gustavo Adolfo Cruz Suárez** (Médico Anestesiólogo, FVL): Proveedor de expertise clínico, validación de variables relevantes y evaluación de la pertinencia de resultados desde la perspectiva del cuidado perioperatorio.

- **Dra. Daniela Hincapié** (Médico, Centro de Investigaciones Clínicas, FVL): Facilitadora del acceso a datos clínicos, coordinación de aspectos éticos y regulatorios, validación de protocolos de investigación.

### 1.3.2 Interesados Técnicos

- **Felipe Ocampo Osorio** (Director de Inteligencia Artificial, FVL): Líder técnico en aspectos de infraestructura de IA, arquitectura de datos, y estrategias de despliegue en entornos hospitalarios.

### 1.3.3 Equipo de Desarrollo

- **Cristian Eduardo Botina Carpio** (Ingeniería de Sistemas, Universidad Icesi): Líder del proyecto, responsable de la coordinación general, documentación y desarrollo de componentes de procesamiento de datos.

- **Juan Manuel Marín Angarita** (Ingeniería de Sistemas, Universidad Icesi): Responsable de desarrollo de modelos de machine learning, análisis estadístico y generación de reportes técnicos.

- **Óscar Andrés Gómez Lozano** (Ingeniería de Sistemas, Universidad Icesi): Responsable de infraestructura de código, visualizaciones, y pipeline de procesamiento de datos.

### 1.3.4 Tutores Académicos

- **Ángela Villota Gómez** (Tutora, Universidad Icesi): Supervisión académica del proyecto, validación metodológica y aseguramiento de calidad de entregables.

## 1.4 Criterios de Éxito del Proyecto

### 1.4.1 Criterios Técnicos

1. **Desempeño predictivo**:
   - Sensibilidad mínima: 85% (detección de al menos 85% de casos verdaderos de shock)
   - Especificidad mínima: 70% (reducción de falsas alarmas)
   - AUC-ROC mínima: 0.80 (capacidad discriminativa superior a métodos tradicionales)
   - Calibración adecuada: error de calibración < 0.1

2. **Reproducibilidad**:
   - Código completamente documentado y versionado (Git)
   - Pipeline reproducible en diferentes entornos
   - Documentación de decisiones de preprocesamiento y modelado

3. **Interpretabilidad**:
   - Explicabilidad de predicciones mediante SHAP o métodos equivalentes
   - Identificación de variables más influyentes
   - Justificación clínica de las decisiones del modelo

### 1.4.2 Criterios Clínicos

1. **Utilidad clínica**:
   - Predicción anticipada en al menos 15-30 minutos antes de manifestaciones clínicas evidentes
   - Reducción potencial de mortalidad y morbilidad (a validar en estudios prospectivos)
   - Compatibilidad con flujo de trabajo quirúrgico existente

2. **Validación por expertos**:
   - Retroalimentación positiva de anestesiólogos sobre la pertinencia clínica
   - Validación de la coherencia de las variables seleccionadas con la fisiopatología del shock

### 1.4.3 Criterios Académicos

1. **Documentación formal**:
   - Informe final conforme a estándares académicos (APA, IEEE)
   - Cumplimiento de guías TRIPOD para modelos predictivos clínicos
   - Presentaciones orales y escritas aprobadas por comité académico

2. **Entrega puntual**:
   - Cumplimiento de cronograma establecido
   - Entregables parciales en fechas acordadas
   - Presentaciones de avance completadas exitosamente

## 1.5 Restricciones y Supuestos

### 1.5.1 Restricciones

1. **Temporales**:
   - Proyecto limitado al semestre académico 2025-2
   - Entrega final: 21 de noviembre de 2025
   - No se contempla despliegue en producción dentro del alcance del proyecto

2. **Acceso a datos**:
   - Dataset retrospectivo proporcionado por FVL
   - Datos anonimizados y aprobados por comité de ética
   - No se contemplan datos en tiempo real

3. **Recursos computacionales**:
   - Desarrollo en infraestructura local (Python, Jupyter)
   - Sin acceso a clusters de GPU de alta potencia
   - Limitación en tamaño de dataset según recursos disponibles

4. **Alcance clínico**:
   - No se incluye validación prospectiva en pacientes reales
   - No se incluye integración con sistemas hospitalarios en producción
   - Enfoque en cirugía mayor no cardiaca (exclusión de cirugía cardíaca)

### 1.5.2 Supuestos

1. **Calidad de datos**:
   - Los registros clínicos de FVL son completos y confiables
   - El etiquetado de casos de shock es preciso y basado en criterios clínicos estándar
   - Los datos son representativos de la población objetivo

2. **Validez clínica**:
   - Las variables disponibles en el dataset son suficientes para predicción
   - Los patrones identificados son generalizables a otros contextos similares
   - La detección temprana tiene potencial de impacto en outcomes clínicos

3. **Metodológicos**:
   - Los métodos de validación cruzada son suficientes para evaluación inicial
   - La retroalimentación de expertos complementa adecuadamente la validación estadística
   - Los sesgos pueden ser identificados y mitigados mediante técnicas estándar

## 1.6 Riesgos Identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Calidad insuficiente de datos (valores faltantes >30%) | Media | Alto | Análisis exhaustivo de calidad, múltiples estrategias de imputación |
| Desbalance severo de clases (>10:1) | Alta | Medio | Técnicas de resampling, métricas ajustadas, SMOTE |
| Falta de variables clave en dataset | Baja | Alto | Validación temprana con expertos clínicos |
| Retrasos en acceso a datos o aprobaciones éticas | Media | Alto | Comunicación proactiva con FVL, cronograma con holgura |
| Modelos no interpretables clínicamente | Media | Alto | Énfasis en métodos interpretables, uso de SHAP/LIME |
| Rendimiento insuficiente (AUC < 0.70) | Baja | Alto | Exploración de múltiples algoritmos, feature engineering |

## 1.7 Plan de Proyecto

### 1.7.1 Fases del Proyecto

El proyecto se estructura siguiendo la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining), adaptada para contextos clínicos con incorporación de elementos de la guía TRIPOD:

1. **Fase 1: Comprensión del Negocio** (Semanas 1-4)
   - Definición de objetivos clínicos y técnicos
   - Identificación de interesados y requisitos
   - Elaboración de cronograma y plan de riesgos

2. **Fase 2: Comprensión de los Datos** (Semanas 5-9)
   - Análisis exploratorio de datos (EDA)
   - Evaluación de calidad de datos
   - Diccionario de datos y documentación de variables

3. **Fase 3: Preparación de los Datos** (Semanas 8-10)
   - Limpieza y transformación de datos
   - Manejo de valores faltantes
   - Ingeniería de características (feature engineering)
   - Selección de variables relevantes

4. **Fase 4: Modelado y Experimentación** (Semanas 11-13)
   - Entrenamiento de modelos candidatos (Regresión Logística, Random Forest, LightGBM)
   - Optimización de hiperparámetros
   - Validación cruzada y evaluación de desempeño

5. **Fase 5: Evaluación Clínica** (Semanas 14-15)
   - Retroalimentación de expertos médicos
   - Análisis de interpretabilidad (SHAP)
   - Validación de utilidad clínica

6. **Fase 6: Documentación y Despliegue** (Semanas 16-17)
   - Elaboración de informe final
   - Preparación de presentaciones
   - Documentación de código y modelos
   - Propuesta de integración

### 1.7.2 Cronograma Detallado

El cronograma específico del proyecto se encuentra documentado en el archivo `cronograma.csv`, el cual está alineado con las fechas académicas establecidas para el curso de Ciencia de Datos/IA:

- **Presentación Avance 1**: 12/09/2025 (Contexto, problema, objetivos, marco teórico)
- **Presentación Avance 2**: 17/10/2025 (Fase preliminar de comprensión de datos)
- **Entrega Final**: 21/11/2025 (Proyecto completo)

### 1.7.3 Herramientas y Tecnologías

- **Gestión de proyecto**: Jira, SharePoint, Microsoft Teams
- **Desarrollo**: Python 3.10+, Jupyter Notebook, VS Code
- **Análisis de datos**: Pandas, NumPy, Scikit-learn
- **Visualización**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, LightGBM, XGBoost
- **Interpretabilidad**: SHAP, LIME
- **Control de versiones**: Git, GitHub
- **Documentación**: Markdown, LaTeX, Mendeley

## 1.8 Marco Metodológico

### 1.8.1 CRISP-DM

El proyecto sigue el proceso CRISP-DM, metodología estándar de la industria para proyectos de minería de datos y aprendizaje automático, que garantiza un enfoque estructurado, iterativo y orientado a resultados.

### 1.8.2 TRIPOD

Se adhiere a la guía TRIPOD (Transparent Reporting of a multivariable Prediction model for Individual Prognosis or Diagnosis), que establece estándares de transparencia y calidad para el desarrollo y reporte de modelos predictivos clínicos. Esto incluye:

- Descripción detallada de la población de estudio
- Especificación clara de variables predictoras y outcome
- Documentación de métodos estadísticos y de validación
- Reporte transparente de desempeño del modelo
- Análisis de limitaciones y sesgos potenciales

## 1.9 Consideraciones Éticas

1. **Protección de datos**:
   - Todos los datos han sido anonimizados por FVL
   - No se manejan identificadores personales (nombres, cédulas, historias clínicas)
   - Cumplimiento con normativas de protección de datos (Ley 1581 de 2012)

2. **Consentimiento informado**:
   - Firma de consentimiento informado por parte del equipo de desarrollo
   - Curso de protección de datos completado
   - Aprobación por comité de ética de FVL

3. **Uso responsable de IA**:
   - El modelo es una herramienta de apoyo, no reemplazo del criterio médico
   - Énfasis en interpretabilidad y explicabilidad
   - Documentación de limitaciones y advertencias de uso

4. **Sesgo y equidad**:
   - Análisis de sesgos potenciales en datos y modelos
   - Evaluación de desempeño en subgrupos demográficos
   - Mitigación de disparidades en predicción

---

**Fecha de elaboración**: Octubre 2025  
**Versión**: 1.0  
**Autores**: Cristian Botina, Juan Manuel Marín, Óscar Gómez  
**Revisado por**: Ángela Villota Gómez
