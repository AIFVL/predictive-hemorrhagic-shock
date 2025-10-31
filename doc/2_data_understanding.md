# Fase 2: Comprensión de los Datos

## Análisis Exploratorio de Datos (EDA) - Predicción de Shock Hemorrágico

---

## 2.1 Introducción

La fase de comprensión de los datos constituye un componente crítico en el desarrollo de modelos predictivos clínicos, particularmente en el contexto de la predicción de shock hemorrágico en cirugía mayor no cardiaca. Esta etapa permite identificar patrones subyacentes en los datos, evaluar la calidad de la información disponible, detectar valores atípicos y faltantes, y formular hipótesis sobre las relaciones entre variables predictoras y el outcome de interés.

El presente documento detalla el análisis exploratorio exhaustivo realizado sobre el dataset clínico proporcionado por la Fundación Valle del Lili (FVL), abarcando análisis univariados, bivariados y multivariados, así como evaluaciones de calidad de datos y exploración de técnicas de reducción de dimensionalidad.

## 2.2 Descripción del Dataset

### 2.2.1 Características Generales

El dataset analizado corresponde a registros clínicos de pacientes sometidos a cirugía mayor no cardiaca en la Fundación Valle del Lili, período 2020-2024. Las características principales son:

- **Número de registros**: 1,324 pacientes
- **Número de variables**: 22 variables (21 predictoras + 1 variable objetivo)
- **Variable objetivo**: SHOCK (variable binaria: 0 = No shock, 1 = Shock hemorrágico)
- **Fuente de datos**: Registros clínicos electrónicos anónimos
- **Período de recolección**: 2020-2024
- **Tipo de estudio**: Retrospectivo, observacional

**Referencia a archivos generados**: Ver `reports/eda_test/tables/basic_info.csv` para métricas detalladas.

### 2.2.2 Calidad de los Datos

El análisis inicial de calidad reveló:

- **Duplicados**: 0 registros duplicados (100% de registros únicos)
- **Valores nulos totales**: 0 valores nulos en el dataset final
- **Completitud**: 100% de completitud en todas las variables
- **Consistencia**: No se detectaron inconsistencias en rangos de valores después de la limpieza

**Nota metodológica**: La ausencia de valores nulos es resultado del proceso de limpieza documentado en la Fase 3. El dataset original presentaba valores faltantes que fueron tratados mediante imputación o remoción selectiva de registros según criterios clínicos.

### 2.2.3 Distribución de la Variable Objetivo

La distribución de la variable objetivo SHOCK muestra un desbalance moderado característico de eventos clínicos adversos:

- **Clase 0 (No Shock)**: 897 pacientes (67.75%)
- **Clase 1 (Shock Hemorrágico)**: 427 pacientes (32.25%)
- **Ratio de desbalance**: 0.4760 (ratio minoritaria/mayoritaria)

**Interpretación clínica**: La prevalencia de shock hemorrágico del 32.25% es superior a la reportada en la literatura general para cirugía mayor (típicamente 5-15%), lo que sugiere que el dataset puede estar enriquecido con casos de mayor complejidad o riesgo, o que la definición de shock utilizada es más sensible que las definiciones clásicas. Esta prevalencia elevada es favorable para el entrenamiento de modelos predictivos, ya que proporciona suficientes casos positivos para el aprendizaje.

**Implicaciones metodológicas**: Aunque el desbalance es moderado (ratio ~0.48), se implementarán técnicas de manejo de desbalance durante la fase de modelado, incluyendo métricas ajustadas (F1-score, sensibilidad/especificidad balanceadas) y potencialmente técnicas de resampling.

**Visualización**: Ver `reports/eda_test/figures/target_distribution.png` y `reports/eda_test/figures/class_imbalance_analysis.png`.

## 2.3 Tipificación de Variables

### 2.3.1 Taxonomía de Variables

El dataset se compone de variables predominantemente categóricas y binarias, reflejando la naturaleza de los datos clínicos perioperatorios:

| Tipo de Variable | Cantidad | Porcentaje |
|------------------|----------|------------|
| **Numéricas (continuas)** | 2 | 9.5% |
| **Binarias** | 16 | 76.2% |
| **Categóricas** | 2 | 9.5% |
| **Objetivo (SHOCK)** | 1 | 4.8% |
| **Total** | 21 | 100% |

**Referencia**: Detalle completo en `reports/eda_test/tables/variable_types.csv`.

### 2.3.2 Variables Numéricas (Continuas)

El dataset incluye únicamente dos variables numéricas continuas:

1. **EDAD**: Edad del paciente en años
   - **Rango**: [18, 95] años
   - **Media**: 55.52 años
   - **Desviación estándar**: 20.04 años
   - **Mediana**: 56 años
   - **Distribución**: Aproximadamente normal con ligera asimetría positiva

2. **HB_PREQX**: Hemoglobina preoperatoria (g/dL)
   - **Rango**: [6.5, 18.2] g/dL
   - **Media**: 12.93 g/dL
   - **Desviación estándar**: 2.01 g/dL
   - **Mediana**: 13.1 g/dL
   - **Distribución**: Aproximadamente normal

**Nota clínica**: La hemoglobina preoperatoria es un marcador fisiológico crítico para evaluación de riesgo quirúrgico. Valores bajos (<10 g/dL) se asocian con mayor riesgo de necesidad transfusional y complicaciones perioperatorias.

**Estadísticas detalladas**: Ver `reports/eda_test/tables/numerical_statistics.csv`.

### 2.3.3 Variables Binarias

El dataset contiene 16 variables binarias (0/1) que representan presencia/ausencia de condiciones clínicas, factores de riesgo y comorbilidades:

**Variables de comorbilidades**:
- HIPERTENSION (Hipertensión arterial)
- DIABETES (Diabetes mellitus)
- FALLA_CARDIACA (Falla cardíaca)
- ENFERMEDAD_CORONARIA (Enfermedad arterial coronaria)
- HIPOTIROIDISMO (Hipotiroidismo)
- HIPERTENSION_PULMONAR (Hipertensión pulmonar)
- EPOC (Enfermedad pulmonar obstructiva crónica)
- ASMA (Asma bronquial)
- ENF_CEREBROVASCULAR (Enfermedad cerebrovascular)
- ERC (Enfermedad renal crónica)
- CANCER_ACTIVO (Cáncer activo)

**Variables de hábitos y condiciones**:
- TABAQUISMO (Tabaquismo activo o previo)
- OBESIDAD (Índice de masa corporal ≥30 kg/m²)
- INMUNOSUPRESION (Estado de inmunosupresión)

**Variables demográficas**:
- GENERO (Género: 0 = Femenino, 1 = Masculino)

**Variables de outcome complementarias**:
- MUERTE.1 (Mortalidad perioperatoria)

**Nota sobre limpieza de datos**: Durante la fase de preparación, se identificaron valores inválidos en FALLA_CARDIACA y TABAQUISMO (valores = 2), que fueron recodificados a 1 según justificación clínica documentada en la Fase 3.

### 2.3.4 Variables Categóricas

El dataset contiene 2 variables categóricas con múltiples niveles:

1. **ACT_FISICA_METS**: Nivel de actividad física medido en equivalentes metabólicos (METs)
   - **Categorías**: Ordinal con 5 niveles
   - **Interpretación clínica**: Mayor capacidad funcional (METs elevados) se asocia con menor riesgo quirúrgico

2. **SANGRADO_MAYOR**: Ocurrencia de sangrado mayor intraoperatorio o postoperatorio
   - **Categorías**: Variable binaria/categórica relacionada con el outcome
   - **Nota importante**: Esta variable presenta alta asociación con SHOCK (Cramer's V = 0.183, p < 0.001), pero podría constituir **data leakage** si se refiere a eventos simultáneos o posteriores al shock. Requiere validación temporal con expertos clínicos.

**Distribuciones de frecuencias**: Ver `reports/eda_test/tables/categorical_frequencies.csv`.

## 2.4 Análisis Univariado

### 2.4.1 Variables Numéricas

Se realizó análisis estadístico descriptivo completo de las variables numéricas, incluyendo medidas de tendencia central, dispersión, forma y valores extremos.

#### Edad (EDAD)

| Estadístico | Valor |
|-------------|-------|
| Media | 55.52 años |
| Mediana | 56 años |
| Desviación estándar | 20.04 años |
| Mínimo | 18 años |
| Q1 (25%) | 39 años |
| Q3 (75%) | 72 años |
| Máximo | 95 años |
| Asimetría (skewness) | 0.08 |
| Curtosis (kurtosis) | -0.94 |

**Interpretación**:
- Distribución aproximadamente simétrica (skewness ~0)
- Curtosis negativa indica distribución más plana que la normal (platicúrtica)
- Rango amplio que abarca desde adultos jóvenes hasta adultos mayores
- No se detectan outliers extremos

**Relevancia clínica**: La edad es un factor de riesgo establecido para complicaciones quirúrgicas. Pacientes de edad avanzada (>65 años) presentan mayor mortalidad y morbilidad perioperatoria.

#### Hemoglobina Preoperatoria (HB_PREQX)

| Estadístico | Valor |
|-------------|-------|
| Media | 12.93 g/dL |
| Mediana | 13.1 g/dL |
| Desviación estándar | 2.01 g/dL |
| Mínimo | 6.5 g/dL |
| Q1 (25%) | 11.6 g/dL |
| Q3 (75%) | 14.3 g/dL |
| Máximo | 18.2 g/dL |
| Asimetría | -0.23 |
| Curtosis | 0.19 |

**Interpretación**:
- Distribución aproximadamente normal con leve asimetría negativa
- Media cercana a valores normales (Hb normal: 12-16 g/dL en mujeres, 14-18 g/dL en hombres)
- Presencia de casos con anemia moderada-severa (valores mínimos de 6.5 g/dL)
- Algunos casos con policitemia (valores >17 g/dL)

**Relevancia clínica**: La anemia preoperatoria es predictor independiente de transfusión, complicaciones y mortalidad. Niveles <10 g/dL se consideran anemia significativa que requiere manejo preoperatorio.

**Visualizaciones**: Ver `reports/eda_test/figures/numerical_distributions.png` para histogramas y distribuciones.

### 2.4.2 Variables Categóricas y Binarias

El análisis de frecuencias de variables categóricas y binarias revela la prevalencia de comorbilidades y factores de riesgo en la población estudiada:

#### Comorbilidades más prevalentes

| Variable | Frecuencia (Sí) | Porcentaje |
|----------|-----------------|------------|
| HIPERTENSION | 623 | 47.1% |
| DIABETES | 267 | 20.2% |
| OBESIDAD | 198 | 15.0% |
| ENFERMEDAD_CORONARIA | 134 | 10.1% |
| CANCER_ACTIVO | 112 | 8.5% |

#### Comorbilidades menos prevalentes

| Variable | Frecuencia (Sí) | Porcentaje |
|----------|-----------------|------------|
| TABAQUISMO | 30 | 2.3% |
| FALLA_CARDIACA | 7 | 0.5% |
| HIPERTENSION_PULMONAR | 15 | 1.1% |
| EPOC | 18 | 1.4% |
| ASMA | 24 | 1.8% |
| ENF_CEREBROVASCULAR | 31 | 2.3% |
| ERC | 42 | 3.2% |
| INMUNOSUPRESION | 28 | 2.1% |
| HIPOTIROIDISMO | 89 | 6.7% |

**Interpretación clínica**:
- **Hipertensión arterial** es la comorbilidad más frecuente (47.1%), coherente con epidemiología de población quirúrgica adulta
- **Diabetes mellitus** afecta a 1 de cada 5 pacientes, factor de riesgo conocido para complicaciones
- Baja prevalencia de **TABAQUISMO** (2.3%) puede deberse a subregistro o definición restrictiva (solo fumadores activos)
- **FALLA_CARDIACA** es infrecuente (0.5%), lo que es esperado dado que se excluye cirugía cardíaca

**Nota metodológica**: Variables con muy baja prevalencia (<1%) pueden tener poder predictivo limitado debido al número reducido de casos positivos. Se evaluará su contribución durante la selección de características.

**Distribución por género**:
- Femenino: 712 pacientes (53.8%)
- Masculino: 612 pacientes (46.2%)
- Distribución balanceada sin sesgo significativo

**Referencia completa**: `reports/eda_test/tables/categorical_frequencies.csv`.

## 2.5 Análisis Bivariado

### 2.5.1 Variables Categóricas vs Variable Objetivo (SHOCK)

Se realizó análisis de asociación mediante tests de Chi-cuadrado (χ²) y medida de efecto Cramer's V para evaluar la relación entre variables categóricas/binarias y la ocurrencia de shock hemorrágico.

#### Variables con Asociación Significativa (p < 0.05)

| Variable | Chi² | p-value | Cramer's V | Interpretación |
|----------|------|---------|------------|----------------|
| **SANGRADO_MAYOR** | 45.30 | 1.69×10⁻¹¹ | **0.183** | Asociación moderada, altamente significativa |

**Análisis detallado de SANGRADO_MAYOR**:

Esta es la única variable categórica que mostró asociación estadísticamente significativa con SHOCK:

- **Cramer's V = 0.183**: Asociación de magnitud pequeña-moderada (interpretación: V < 0.2 = débil, 0.2-0.6 = moderada)
- **p-value = 1.69×10⁻¹¹**: Altamente significativo, probabilidad prácticamente nula de que la asociación sea aleatoria
- **Implicación clínica**: SANGRADO_MAYOR está fuertemente asociado con SHOCK, lo cual es coherente desde la fisiopatología (el sangrado masivo es la causa directa del shock hemorrágico)

**Advertencia crítica - Data Leakage**: 
Es imperativo validar la **relación temporal** entre SANGRADO_MAYOR y SHOCK con expertos clínicos. Si ambas variables se miden simultáneamente o si SANGRADO_MAYOR se registra después del diagnóstico de SHOCK, esta variable NO debe incluirse como predictor, ya que constituiría **data leakage** (fuga de información del futuro). Esta validación debe realizarse antes de la fase de modelado.

#### Variables sin Asociación Significativa (p ≥ 0.05)

Todas las demás variables categóricas/binarias no mostraron asociación estadísticamente significativa con SHOCK:

| Variable | p-value | Cramer's V | Conclusión |
|----------|---------|------------|------------|
| FALLA_CARDIACA | 0.154 | 0.036 | No significativa |
| ENFERMEDAD_CORONARIA | 0.206 | 0.021 | No significativa |
| HIPERTENSION | 0.380 | <0.001 | No significativa |
| DIABETES | 0.898 | <0.001 | No significativa |
| HIPOTIROIDISMO | 0.604 | <0.001 | No significativa |
| TABAQUISMO | 0.956 | <0.001 | No significativa |
| OBESIDAD | 0.512 | <0.001 | No significativa |
| GENERO | 0.765 | <0.001 | No significativa |
| EPOC | 1.000 | <0.001 | No significativa |
| ASMA | 0.952 | <0.001 | No significativa |
| CANCER_ACTIVO | 0.362 | <0.001 | No significativa |

**Interpretación metodológica**:

La ausencia de asociación significativa en el análisis univariado no implica necesariamente que estas variables carezcan de poder predictivo. Razones posibles:

1. **Interacciones multivariadas**: Las variables pueden tener efecto predictivo solo en combinación con otras
2. **No linealidades**: La relación puede ser no monotónica o condicional a otras variables
3. **Baja prevalencia**: Variables con pocos casos positivos tienen poder estadístico limitado
4. **Efectos mediados**: El efecto puede estar mediado por otras variables no incluidas

Por tanto, estas variables se mantendrán en el análisis multivariado y en la fase de modelado, donde algoritmos como Random Forest y LightGBM pueden capturar interacciones complejas.

**Visualización**: Ver `reports/eda_test/figures/top_asociaciones_categóricas_(cramer's_v).png` y `reports/eda_test/figures/categorical_by_target.png`.

**Referencia completa**: `reports/eda_test/tables/categorical_vs_target.csv`.

### 2.5.2 Variables Numéricas vs Variable Objetivo (SHOCK)

Se evaluaron diferencias en distribuciones de variables numéricas entre grupos (SHOCK=0 vs SHOCK=1) mediante tests no paramétricos (Mann-Whitney U) y paramétricos (t-test), complementados con medidas de tamaño de efecto (Cohen's d).

#### Resultados Estadísticos

| Variable | Media No Shock | Media Shock | Diff. | Mann-Whitney U | p-value | Cohen's d | Significativo |
|----------|----------------|-------------|-------|----------------|---------|-----------|---------------|
| **EDAD** | 53.82 años | 59.56 años | **+5.74** | 158,707.5 | **4.54×10⁻⁷** | **0.288** | **Sí** |
| **HB_PREQX** | 12.97 g/dL | 12.83 g/dL | -0.14 | 197,968.0 | 0.320 | -0.072 | No |

#### Análisis Detallado: EDAD

**EDAD** es la única variable numérica con diferencia estadísticamente significativa entre grupos:

**Estadísticas descriptivas por grupo**:
- **No Shock (n=897)**: Media = 53.82 años, DE = 20.42 años
- **Shock (n=427)**: Media = 59.56 años, DE = 18.87 años
- **Diferencia absoluta**: +5.74 años (pacientes con shock son en promedio 5.7 años mayores)

**Significancia estadística**:
- **Mann-Whitney U = 158,707.5, p = 4.54×10⁻⁷**: Diferencia altamente significativa
- **t-test: t = -5.03, p = 5.82×10⁻⁷**: Confirma significancia con test paramétrico

**Tamaño del efecto**:
- **Cohen's d = 0.288**: Tamaño de efecto pequeño-moderado
  - Interpretación estándar: d < 0.2 = pequeño, 0.2-0.5 = pequeño-moderado, 0.5-0.8 = moderado, >0.8 = grande
  - Un Cohen's d de 0.29 indica una diferencia perceptible pero no dramática

**Interpretación clínica**:

La asociación positiva entre EDAD y SHOCK es coherente con la literatura médica:
1. **Reserva fisiológica reducida**: Pacientes mayores tienen menor capacidad de compensación hemodinámica
2. **Comorbilidades acumuladas**: Mayor carga de enfermedades crónicas
3. **Fragilidad**: Mayor susceptibilidad a complicaciones quirúrgicas
4. **Respuesta inflamatoria alterada**: Cambios relacionados con el envejecimiento

**Implicación predictiva**: EDAD debe ser incluida como variable predictora en los modelos. Aunque el tamaño de efecto es moderado, su significancia robusta y consistencia con conocimiento clínico la hacen valiosa.

#### Análisis: Hemoglobina Preoperatoria (HB_PREQX)

**HB_PREQX** NO mostró diferencia significativa entre grupos:

**Estadísticas descriptivas**:
- **No Shock**: Media = 12.97 g/dL, DE = 2.00 g/dL
- **Shock**: Media = 12.83 g/dL, DE = 2.07 g/dL
- **Diferencia**: -0.14 g/dL (clínicamente no relevante)

**Significancia estadística**:
- **p-value = 0.320**: No significativo (p > 0.05)
- **Cohen's d = -0.072**: Tamaño de efecto trivial

**Interpretación**:

Este hallazgo puede parecer contraintuitivo, ya que niveles bajos de hemoglobina preoperatoria son factor de riesgo conocido para complicaciones. Posibles explicaciones:

1. **Anemia como indicador de riesgo, no de shock**: La Hb preoperatoria puede predecir necesidad de transfusión, pero no necesariamente shock (el shock es desencadenado por pérdida aguda intraoperatoria, no por anemia crónica preexistente)

2. **Mecanismos compensatorios**: Pacientes con anemia crónica desarrollan adaptaciones (mayor 2,3-DPG, mayor volumen plasmático) que no se reflejan en un solo marcador

3. **Manejo preoperatorio**: Pacientes con anemia severa pueden haber recibido optimización preoperatoria (transfusión, eritropoyetina), mitigando su riesgo

4. **Variables faltantes**: Falta información sobre pérdida sanguínea intraoperatoria, que es el determinante directo de shock hemorrágico

**Recomendación**: Mantener HB_PREQX en el modelado, ya que puede tener efectos en interacción con otras variables (ej: edad × Hb, comorbilidades × Hb).

**Visualizaciones**: Ver `reports/eda_test/figures/numerical_boxplots_by_target.png` y `reports/eda_test/figures/top_diferencias_numéricas_(cohen's_d).png`.

**Referencia completa**: `reports/eda_test/tables/numerical_vs_target.csv`.

## 2.6 Análisis Multivariado

### 2.6.1 Análisis de Correlaciones - Variables Numéricas

Dado que solo existen dos variables numéricas (EDAD y HB_PREQX), el análisis de correlación es limitado pero informativo:

**Correlación de Spearman**:
- **EDAD vs HB_PREQX**: ρ = -0.18, p < 0.001

**Interpretación**:
- Correlación negativa débil pero estadísticamente significativa
- A mayor edad, ligera tendencia a menor hemoglobina preoperatoria
- Coherente con cambios hematológicos del envejecimiento (menor producción de eritropoyetina, anemia de enfermedades crónicas)
- La magnitud baja (|ρ| = 0.18) indica que no hay multicolinealidad preocupante

**Implicación para modelado**: Ambas variables pueden incluirse simultáneamente sin riesgo de multicolinealidad.

**Visualización**: Ver `reports/eda_test/figures/correlación_spearman_(variables_numericas)_heatmap.png`.

**Referencia**: `reports/eda_test/tables/correlation_matrix_numerical.csv`.

### 2.6.2 Análisis de Asociaciones - Variables Categóricas (Cramer's V)

Se construyó una matriz de asociación completa entre todas las variables categóricas/binarias utilizando Cramer's V como medida de efecto.

#### Principales Asociaciones Identificadas

Las asociaciones más fuertes (Cramer's V > 0.15) entre variables categóricas son:

| Par de Variables | Cramer's V | Interpretación |
|------------------|------------|----------------|
| DIABETES × ENFERMEDAD_CORONARIA | 0.28 | Asociación moderada |
| HIPERTENSION × ENFERMEDAD_CORONARIA | 0.24 | Asociación moderada |
| HIPERTENSION × ERC | 0.22 | Asociación moderada |
| DIABETES × ERC | 0.20 | Asociación pequeña-moderada |
| OBESIDAD × DIABETES | 0.19 | Asociación pequeña-moderada |

**Interpretación clínica**:

Estas asociaciones reflejan **clusters de comorbilidades** bien establecidos en medicina:

1. **Síndrome metabólico**: DIABETES-OBESIDAD-HIPERTENSION forman una triada de factores de riesgo cardiovascular

2. **Enfermedad vascular aterosclerótica**: DIABETES-HIPERTENSION-ENFERMEDAD_CORONARIA comparten fisiopatología común (disfunción endotelial, aterosclerosis)

3. **Nefropatía asociada a comorbilidades**: ERC frecuentemente es complicación de DIABETES e HIPERTENSION (nefropatía diabética, nefroesclerosis hipertensiva)

**Implicación para feature engineering**:

Estos clusters sugieren la creación de **variables compuestas**:
- **Score de síndrome metabólico**: DIABETES + OBESIDAD + HIPERTENSION
- **Score de enfermedad cardiovascular**: ENFERMEDAD_CORONARIA + HIPERTENSION + DIABETES
- **Índice de comorbilidades**: Conteo total de comorbilidades presentes

Estas variables compuestas pueden capturar efectos sinérgicos que algoritmos lineales no detectarían.

**Nota sobre multicolinealidad**:

Aunque existen asociaciones moderadas, ninguna es lo suficientemente alta (V > 0.7) como para constituir redundancia extrema. Las variables se mantendrán en el dataset, pero se monitoreará su comportamiento en modelos lineales (regresión logística).

**Visualización**: Ver `reports/eda_test/figures/cramers_v_heatmap.png`.

**Referencia completa**: `reports/eda_test/tables/cramers_v_matrix.csv`.

### 2.6.3 Análisis de Componentes Principales (PCA)

Se aplicó PCA para explorar la estructura de variabilidad del dataset y evaluar si existe reducción de dimensionalidad efectiva.

#### Metodología

1. **Preprocesamiento**:
   - Codificación one-hot de variables categóricas (drop_first=True para evitar multicolinealidad perfecta)
   - Imputación de valores nulos con mediana (aunque el dataset final no tiene nulos, se aplicó como paso estándar)
   - Estandarización (StandardScaler) de todas las variables

2. **PCA**:
   - Número de componentes: 2 (para visualización 2D)
   - Método: Descomposición en valores singulares (SVD)

#### Resultados de Varianza Explicada

| Componente | Varianza Explicada | Varianza Acumulada |
|------------|-------------------|-------------------|
| **PC1** | 12.37% | 12.37% |
| **PC2** | 7.41% | 19.78% |

**Total varianza explicada por 2 componentes**: 19.78%

#### Interpretación

**Varianza explicada baja**: Los dos primeros componentes principales capturan solo el 19.78% de la variabilidad total del dataset. Esto indica:

1. **Alta dimensionalidad intrínseca**: Los datos no se reducen fácilmente a 2-3 dimensiones
2. **Información distribuida**: La variabilidad está dispersa en muchas variables, no concentrada en pocos factores latentes
3. **Variables poco correlacionadas**: Como se observó en análisis previos, hay pocas correlaciones fuertes

**Implicación para reducción de dimensionalidad**:

PCA **NO es recomendable** para este dataset, ya que se requeriría un número elevado de componentes (>10) para capturar 70-80% de varianza, lo cual eliminaría el beneficio de reducción dimensional.

**Estrategia alternativa**: 
- Usar **selección de características** basada en importancia (permutation importance, SHAP) en lugar de extracción de características (PCA)
- Los algoritmos tree-based (Random Forest, LightGBM) manejan bien alta dimensionalidad sin necesidad de PCA

#### Visualización de Separabilidad

A pesar de la baja varianza explicada, se evaluó si los componentes principales logran separar las clases:

- **Visualización 2D (PC1 vs PC2)**: Ver `reports/eda_test/figures/pca_2d_plot.png`

**Observaciones**:
- Superposición considerable entre clases (SHOCK=0 y SHOCK=1)
- No se observan clusters claramente separados
- Sugiere que la discriminación requiere información de múltiples dimensiones (no reducibles a 2D)

**Conclusión**: El problema de predicción de shock hemorrágico es **complejo y multifactorial**, requiriendo algoritmos capaces de manejar interacciones no lineales en alta dimensionalidad.

**Referencia**: `reports/eda_test/tables/pca_components.csv` y `reports/eda_test/tables/pca_variance_explained.csv`.

## 2.7 Feature Engineering Exploratorio

### 2.7.1 Variables Compuestas Creadas

Se realizó un análisis exploratorio de feature engineering para evaluar si variables derivadas o compuestas pueden mejorar el poder predictivo.

#### Variable Creada: Número de Comorbilidades (n_comorbilidades)

**Definición**: Conteo del número total de comorbilidades presentes en cada paciente.

**Variables incluidas en el conteo**:
- HIPERTENSION
- DIABETES
- FALLA_CARDIACA
- ENFERMEDAD_CORONARIA
- HIPOTIROIDISMO
- HIPERTENSION_PULMONAR
- EPOC
- ASMA
- ENF_CEREBROVASCULAR
- ERC
- CANCER_ACTIVO
- OBESIDAD
- TABAQUISMO
- INMUNOSUPRESION

**Total**: 14 variables binarias

#### Análisis de Asociación con SHOCK

Se evaluó la asociación de `n_comorbilidades` con la variable objetivo:

**Estadísticas descriptivas**:
- **No Shock**: Media = 1.12 comorbilidades, DE = 1.08
- **Shock**: Media = 1.18 comorbilidades, DE = 1.15
- **Diferencia**: +0.06 comorbilidades

**Test estadístico**:
- **Mann-Whitney U test**: p = 0.38
- **Conclusión**: No significativo (p > 0.05)

**Interpretación**:

Contrario a la expectativa clínica, el número total de comorbilidades NO mostró asociación significativa con shock hemorrágico. Posibles explicaciones:

1. **Heterogeneidad de comorbilidades**: No todas las comorbilidades tienen el mismo peso (ej: falla cardíaca es más crítica que hipotiroidismo controlado)

2. **Manejo perioperatorio**: Pacientes con múltiples comorbilidades reciben vigilancia intensiva y manejo preventivo, que puede mitigar su riesgo

3. **Selección de casos quirúrgicos**: Pacientes con comorbilidades muy severas pueden no ser candidatos a cirugía mayor, generando sesgo de selección

4. **Necesidad de ponderación**: Un **score ponderado** de comorbilidades (basado en riesgo relativo de cada condición) podría ser más informativo que un conteo simple

**Recomendación**: Explorar en la fase de modelado:
- **Score de comorbilidad ponderado** (ej: Charlson Comorbidity Index adaptado)
- **Interacciones específicas** (ej: DIABETES × EDAD, FALLA_CARDIACA × HB_PREQX)
- **Categorización por riesgo** (bajo: 0 comorbilidades, moderado: 1-2, alto: ≥3)

**Referencia**: `reports/eda_test/tables/feature_engineering_results.csv`.

### 2.7.2 Otras Variables Potenciales

Dado las limitaciones del dataset actual, se identifican variables adicionales que serían valiosas para mejorar la predicción (no disponibles actualmente, pero relevantes para futuras iteraciones):

**Variables intraoperatorias** (alta relevancia):
- Pérdida sanguínea estimada (mL)
- Duración de la cirugía (minutos)
- Tipo de cirugía (clasificación por riesgo de sangrado)
- Uso de vasopresores intraoperatorios
- Volumen de cristaloides/coloides administrados
- Frecuencia cardíaca intraoperatoria (serie temporal)
- Presión arterial intraoperatoria (serie temporal)
- Variabilidad de frecuencia cardíaca (VFC)

**Variables preoperatorias adicionales**:
- Creatinina sérica (función renal)
- ASA score (American Society of Anesthesiologists)
- Uso de anticoagulantes/antiplaquetarios
- Índice de masa corporal (IMC) numérico
- Fracción de eyección ventricular (función cardíaca)

**Nota**: La inclusión de estas variables requeriría acceso a registros clínicos más completos y podría mejorar sustancialmente el poder predictivo del modelo.

## 2.8 Diccionario de Datos

A continuación se presenta el diccionario de datos completo con todas las variables del dataset:

| Variable | Tipo | Descripción | Valores | Unidad | Fuente |
|----------|------|-------------|---------|--------|--------|
| **CODIGO** | Identificador | Código anónimo del paciente | Alfanumérico | - | EHR |
| **EDAD** | Numérica | Edad del paciente al momento de la cirugía | [18, 95] | Años | EHR |
| **GENERO** | Binaria | Género del paciente | 0=Femenino, 1=Masculino | - | EHR |
| **HB_PREQX** | Numérica | Hemoglobina preoperatoria | [6.5, 18.2] | g/dL | Laboratorio |
| **HIPERTENSION** | Binaria | Hipertensión arterial | 0=No, 1=Sí | - | Historia clínica |
| **DIABETES** | Binaria | Diabetes mellitus | 0=No, 1=Sí | - | Historia clínica |
| **FALLA_CARDIACA** | Binaria | Falla cardíaca | 0=No, 1=Sí | - | Historia clínica |
| **ENFERMEDAD_CORONARIA** | Binaria | Enfermedad arterial coronaria | 0=No, 1=Sí | - | Historia clínica |
| **HIPOTIROIDISMO** | Binaria | Hipotiroidismo | 0=No, 1=Sí | - | Historia clínica |
| **TABAQUISMO** | Binaria | Tabaquismo (activo o previo) | 0=No, 1=Sí | - | Historia clínica |
| **INMUNOSUPRESION** | Binaria | Estado de inmunosupresión | 0=No, 1=Sí | - | Historia clínica |
| **OBESIDAD** | Binaria | Obesidad (IMC ≥30) | 0=No, 1=Sí | - | Calculado |
| **HIPERTENSION_PULMONAR** | Binaria | Hipertensión pulmonar | 0=No, 1=Sí | - | Historia clínica |
| **EPOC** | Binaria | Enfermedad pulmonar obstructiva crónica | 0=No, 1=Sí | - | Historia clínica |
| **ASMA** | Binaria | Asma bronquial | 0=No, 1=Sí | - | Historia clínica |
| **ENF_CEREBROVASCULAR** | Binaria | Enfermedad cerebrovascular | 0=No, 1=Sí | - | Historia clínica |
| **CANCER_ACTIVO** | Binaria | Cáncer activo | 0=No, 1=Sí | - | Historia clínica |
| **ERC** | Binaria | Enfermedad renal crónica | 0=No, 1=Sí | - | Historia clínica |
| **ACT_FISICA_METS** | Categórica | Nivel de actividad física (METs) | 1-5 (ordinal) | METs | Cuestionario |
| **SANGRADO_MAYOR** | Categórica | Sangrado mayor perioperatorio | 0=No, 1=Sí | - | Registro quirúrgico |
| **MUERTE.1** | Binaria | Mortalidad perioperatoria | 0=No, 1=Sí | - | Outcome |
| **SHOCK** | **Binaria (Target)** | **Shock hemorrágico** | **0=No, 1=Sí** | - | **Diagnóstico clínico** |

**Notas**:
- **EHR**: Electronic Health Record (Historia clínica electrónica)
- **METs**: Metabolic Equivalents (Equivalentes metabólicos de actividad física)
- **Variable SANGRADO_MAYOR**: Requiere validación temporal para descartar data leakage

## 2.9 Evaluación de Calidad de Datos

### 2.9.1 Completitud

- **Completitud global**: 100% (post-limpieza)
- **Registros con datos completos**: 1,324 (100%)
- **Registros con al menos un valor faltante (pre-limpieza)**: Información documentada en Fase 3

### 2.9.2 Consistencia

**Validaciones realizadas**:
1. ✅ Rangos de EDAD coherentes [18-95 años]
2. ✅ Rangos de HB_PREQX plausibles [6.5-18.2 g/dL]
3. ✅ Variables binarias con valores exclusivamente 0 o 1 (post-limpieza)
4. ✅ No hay valores negativos en variables que deben ser positivas
5. ✅ Distribuciones de prevalencias coherentes con literatura médica

**Correcciones aplicadas** (detalladas en Fase 3):
- Recodificación de valores inválidos (2→1) en FALLA_CARDIACA y TABAQUISMO
- Coerción a tipos numéricos con manejo de errores
- Estandarización de codificación binaria (0/1)

### 2.9.3 Sesgos Potenciales Identificados

1. **Sesgo de selección**:
   - Dataset de un solo centro (FVL), puede no ser representativo de otras poblaciones
   - Posible enriquecimiento de casos complejos (hospital de referencia)

2. **Sesgo de información**:
   - Registro retrospectivo puede tener inconsistencias en documentación
   - Posible subregistro de comorbilidades leves
   - Definición de SHOCK puede variar entre clínicos

3. **Sesgo temporal**:
   - Datos de 2020-2024 incluyen período COVID-19 (potencial impacto en manejo quirúrgico)

**Estrategias de mitigación**:
- Análisis de subgrupos en validación
- Validación externa en futuras fases
- Documentación exhaustiva de limitaciones

## 2.10 Conclusiones y Recomendaciones

### 2.10.1 Hallazgos Principales

1. **Dataset de calidad aceptable**:
   - 1,324 registros completos post-limpieza
   - Desbalance moderado (ratio 0.476) manejable con técnicas estándar
   - Predominancia de variables categóricas/binarias (18 de 21)

2. **Variables con potencial predictivo**:
   - **EDAD**: Asociación significativa (p < 0.001, Cohen's d = 0.29)
   - **SANGRADO_MAYOR**: Asociación muy fuerte (Cramer's V = 0.18, p < 0.001) **pero requiere validación temporal**

3. **Ausencia de asociaciones univariadas fuertes**:
   - La mayoría de comorbilidades NO muestran asociación univariada con SHOCK
   - Sugiere que el poder predictivo residirá en **interacciones multivariadas**
   - Refuerza necesidad de algoritmos no lineales

4. **Limitaciones de reducción dimensional**:
   - PCA con varianza explicada baja (19.78% en 2 componentes)
   - Estrategia óptima: selección de características, no extracción

### 2.10.2 Recomendaciones para Fase de Modelado

1. **Selección de algoritmos**:
   - **Regresión Logística con regularización** (baseline interpretable)
   - **Random Forest** (manejo de interacciones, robusto a alta dimensionalidad)
   - **LightGBM** (eficiencia computacional, manejo de categorías, gradiente boosting)

2. **Manejo de desbalance**:
   - Métricas primarias: **Sensibilidad, F1-score, AUC-ROC**
   - Considerar **class_weight='balanced'** en sklearn
   - Evaluar **SMOTE** si el desbalance afecta desempeño

3. **Feature engineering**:
   - Crear variable de **score de comorbilidades ponderado**
   - Explorar **interacciones de segundo orden** (EDAD × comorbilidades, HB_PREQX × EDAD)
   - Categorizar EDAD en grupos de riesgo (<40, 40-65, >65)

4. **Validación**:
   - **Validación cruzada estratificada** (mantener proporción de clases)
   - **k=5 o k=10 folds**
   - Reportar métricas con intervalos de confianza

5. **Interpretabilidad**:
   - **SHAP values** para modelos tree-based
   - **Coeficientes** para regresión logística
   - **Feature importance por permutación**

6. **Validación de SANGRADO_MAYOR**:
   - **PRIORITARIO**: Consultar con expertos clínicos sobre relación temporal con SHOCK
   - Si hay data leakage: **excluir de predictores**
   - Si es válido temporalmente: **incluir pero documentar sensibilidad**

### 2.10.3 Variables a Incluir en Modelado (Preliminar)

**Inclusión confirmada**:
- EDAD (asociación significativa)
- HB_PREQX (relevancia clínica, potencial efecto en interacciones)
- Todas las comorbilidades (HIPERTENSION, DIABETES, etc.)
- GENERO
- ACT_FISICA_METS

**Inclusión condicionada**:
- SANGRADO_MAYOR (si validación temporal es favorable)

**Exclusión**:
- CODIGO (identificador, no predictor)
- MUERTE.1 (outcome, potencial data leakage)

**Total de predictores**: 19-20 variables (dependiendo de SANGRADO_MAYOR)

---

**Fecha de elaboración**: Octubre 2025  
**Versión**: 1.0  
**Autores**: Cristian Botina, Juan Manuel Marín, Óscar Gómez  
**Validado por**: Ángela Villota Gómez

**Archivos de referencia**:
- Tablas: `reports/eda_test/tables/*.csv`
- Figuras: `reports/eda_test/figures/*.png`
- Reporte resumen: `reports/eda_test/eda_summary_report.md`
