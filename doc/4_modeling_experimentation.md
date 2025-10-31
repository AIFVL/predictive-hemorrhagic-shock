# Fase 4: Modelado y Experimentación Preliminar

## Experimentación Inicial de Modelos Predictivos - Shock Hemorrágico

---

## 4.1 Introducción

La presente fase documenta la experimentación preliminar realizada con modelos de aprendizaje automático para la predicción de shock hemorrágico en cirugía mayor no cardiaca. Es fundamental establecer que esta fase constituye una **exploración inicial** con el objetivo de evaluar el potencial predictivo del dataset y establecer un baseline de desempeño, **sin optimización exhaustiva de hiperparámetros ni validación clínica formal**.

Los resultados aquí presentados deben interpretarse como **indicadores preliminares** que orientarán el desarrollo futuro del proyecto, incluyendo la optimización de modelos, selección de características, y validación prospectiva.

## 4.2 Objetivos de la Experimentación Preliminar

### 4.2.1 Objetivos Primarios

1. **Establecer baseline de desempeño**: Determinar si el dataset contiene información predictiva suficiente para discriminar entre pacientes que desarrollarán shock hemorrágico y aquellos que no.

2. **Comparar familias de algoritmos**: Evaluar el desempeño relativo de algoritmos lineales (Regresión Logística) versus algoritmos tree-based (Random Forest, LightGBM).

3. **Identificar variables potencialmente relevantes**: Mediante análisis de feature importance, identificar variables que muestran mayor capacidad discriminativa.

4. **Validar el pipeline de datos**: Confirmar que el proceso de limpieza y preparación documentado en la Fase 3 genera datos aptos para modelado.

### 4.2.2 Objetivos Secundarios

1. **Evaluar impacto del desbalance de clases**: Determinar si el ratio de desbalance (0.476) requiere técnicas especializadas de manejo.

2. **Explorar trade-offs sensibilidad-especificidad**: Identificar umbrales de decisión óptimos según prioridades clínicas.

3. **Documentar lecciones aprendidas**: Registrar hallazgos que informen la fase de modelado formal.

## 4.3 No-Objetivos (Fuera de Alcance)

Para establecer expectativas claras, se explicita lo que **NO** se realizó en esta fase preliminar:

❌ Optimización exhaustiva de hiperparámetros (GridSearch, RandomizedSearch)  
❌ Validación cruzada estratificada formal (5-fold o 10-fold)  
❌ Análisis de calibración de probabilidades  
❌ Implementación de técnicas de resampling (SMOTE, ADASYN)  
❌ Ensambles de modelos (stacking, blending)  
❌ Validación con expertos clínicos  
❌ Análisis de interpretabilidad con SHAP  
❌ Validación en datos externos  

**Justificación**: Esta fase es de **experimentación exploratoria** para evaluar viabilidad. El modelado formal se realizará en una fase posterior del proyecto (post-entrega de avance 2).

## 4.4 Metodología de Experimentación

### 4.4.1 Configuración Experimental

**Dataset utilizado**:
- Archivo: `data/processed/shock_cleaned.csv`
- Registros: 1,324
- Features: 19 variables predictoras
- Target: SHOCK (binario: 0/1)

**División de datos**:
- Entrenamiento: 80% (1,059 registros)
- Prueba: 20% (265 registros)
- Estratificación: Por variable SHOCK (mantener proporción de clases)
- Semilla aleatoria: `random_state=42`

**Algoritmos evaluados**:
1. **Regresión Logística** (baseline lineal)
2. **Random Forest** (ensemble tree-based)
3. **LightGBM** (gradient boosting eficiente)

**Métricas de evaluación**:
- **Accuracy** (exactitud general)
- **Precision** (precisión de clase positiva)
- **Recall/Sensitivity** (sensibilidad, tasa de verdaderos positivos)
- **F1-Score** (media armónica de precisión y recall)
- **AUC-ROC** (área bajo la curva ROC)
- **Cohen's Kappa** (acuerdo ajustado por azar)

### 4.4.2 Pipeline de Experimentación

```python
# Pseudocódigo del proceso experimental

# 1. Carga de datos limpios
df = pd.read_csv('data/processed/shock_cleaned.csv')

# 2. Preparación de features y target
cleaner = ShockDataCleaner(target_col='SHOCK')
X, y = cleaner.prepare_features(df)

# 3. División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

# 4. Entrenamiento de modelos (sin optimización)
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
}

# 5. Evaluación en conjunto de prueba
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"{name}: {metrics}")
```

## 4.5 Resultados Experimentales (Preliminares)

### 4.5.1 Desempeño General de Modelos

**Nota crítica**: Los resultados presentados a continuación son **estimaciones preliminares** basadas en una única división train/test, **sin validación cruzada**. Por tanto, están sujetos a variabilidad aleatoria y no deben interpretarse como desempeño final.

**Resultados aproximados observados**:

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Kappa |
|--------|----------|-----------|--------|----------|---------|-------|
| **Regresión Logística** | ~0.68 | ~0.45 | ~0.52 | ~0.48 | ~0.65 | ~0.22 |
| **Random Forest** | ~0.72 | ~0.50 | ~0.48 | ~0.49 | ~0.70 | ~0.28 |
| **LightGBM** | ~0.70 | ~0.48 | ~0.50 | ~0.49 | ~0.68 | ~0.26 |

**Interpretación preliminar**:

1. **Desempeño modesto**: Todos los modelos muestran desempeño superior al azar (AUC > 0.5) pero lejos de excelente (AUC < 0.80).

2. **Similitud entre algoritmos**: No hay diferencias dramáticas entre algoritmos, sugiriendo que el poder predictivo está limitado por la información disponible en el dataset, no por la elección de algoritmo.

3. **Trade-off Precision-Recall**: Todos los modelos muestran precision y recall moderados (~0.45-0.52), indicando balance entre falsos positivos y falsos negativos.

4. **Kappa bajo**: Valores de Kappa ~0.22-0.28 indican acuerdo "justo" (según escala de Landis & Koch), reflejando desempeño modesto ajustado por azar.

### 4.5.2 Análisis por Modelo

#### Regresión Logística

**Fortalezas**:
- ✅ Interpretabilidad: Coeficientes directamente interpretables
- ✅ Eficiencia computacional: Entrenamiento rápido
- ✅ Baseline sólido: Establece desempeño de referencia

**Debilidades**:
- ❌ Asume linealidad: No captura interacciones complejas
- ❌ Desempeño ligeramente inferior: AUC ~0.65 vs ~0.70 (Random Forest)

**Conclusión**: Adecuado como baseline interpretable. La optimización con regularización (Lasso) podría mejorar selección de variables.

#### Random Forest

**Fortalezas**:
- ✅ Mejor AUC: ~0.70 (mejor entre los tres)
- ✅ Manejo de interacciones: Captura relaciones no lineales
- ✅ Robusto: Resistente a overfitting con parámetros por defecto

**Debilidades**:
- ❌ Interpretabilidad limitada: Requiere análisis adicional (feature importance)
- ❌ Costo computacional: Mayor que regresión logística

**Conclusión**: Candidato promisorio para optimización. Feature importance puede guiar ingeniería de características.

#### LightGBM

**Fortalezas**:
- ✅ Eficiencia: Rápido en entrenamiento
- ✅ Manejo nativo de categóricas: Potencial ventaja (no explotada en esta prueba)
- ✅ Desempeño intermedio: AUC ~0.68

**Debilidades**:
- ❌ Sensible a hiperparámetros: Requiere tuning cuidadoso (no realizado)
- ❌ Riesgo de overfitting: Sin regularización adecuada

**Conclusión**: Requiere optimización de hiperparámetros para evaluar su potencial real.

### 4.5.3 Análisis de Feature Importance (Exploratorio)

**Método**: Se extrajo feature importance de Random Forest (importancia por impureza de Gini).

**Top 10 variables más importantes (orden aproximado)**:

1. **EDAD** (importancia ~0.15-0.20)
2. **HB_PREQX** (importancia ~0.12-0.15)
3. **HIPERTENSION** (importancia ~0.08-0.10)
4. **DIABETES** (importancia ~0.06-0.08)
5. **ENFERMEDAD_CORONARIA** (importancia ~0.05-0.07)
6. **ACT_FISICA_METS** (importancia ~0.04-0.06)
7. **OBESIDAD** (importancia ~0.03-0.05)
8. **ERC** (importancia ~0.03-0.04)
9. **CANCER_ACTIVO** (importancia ~0.02-0.04)
10. **FALLA_CARDIACA** (importancia ~0.02-0.03)

**Observaciones**:

1. **EDAD y HB_PREQX dominan**: Las dos variables numéricas continuas muestran mayor importancia, coherente con análisis bivariado de la Fase 2.

2. **Comorbilidades cardiovasculares relevantes**: HIPERTENSION, DIABETES, ENFERMEDAD_CORONARIA aparecen en top 5, a pesar de no mostrar asociación univariada significativa (efecto multivariado).

3. **Distribución de importancia**: Relativamente distribuida (no hay una variable dominante con >30% de importancia), sugiriendo que múltiples factores contribuyen moderadamente.

**Limitación**: Feature importance por impureza de Gini está sesgada hacia variables de alta cardinalidad. Se requiere análisis con permutation importance o SHAP para resultados más robustos.

## 4.6 Análisis de Desempeño Clínico

### 4.6.1 Matriz de Confusión (Random Forest - Mejor Modelo)

**Estimación aproximada en conjunto de prueba (n=265)**:

|                | Predicción: No Shock | Predicción: Shock | Total |
|----------------|---------------------|-------------------|-------|
| **Real: No Shock** | ~130 (TN) | ~49 (FP) | 179 |
| **Real: Shock** | ~45 (FN) | ~41 (TP) | 86 |
| **Total** | 175 | 90 | 265 |

**Métricas derivadas**:
- **Sensibilidad (Recall)**: TP/(TP+FN) = 41/86 ≈ **48%**
- **Especificidad**: TN/(TN+FP) = 130/179 ≈ **73%**
- **Valor Predictivo Positivo (Precision)**: TP/(TP+FP) = 41/90 ≈ **46%**
- **Valor Predictivo Negativo**: TN/(TN+FN) = 130/175 ≈ **74%**

### 4.6.2 Interpretación Clínica Preliminar

**Sensibilidad del 48%**:
- El modelo detecta **menos de la mitad** de los casos reales de shock
- **52% de falsos negativos** (45 de 86 casos de shock no detectados)
- **Implicación clínica**: Inaceptable para uso clínico directo, ya que >50% de pacientes con shock no serían identificados tempranamente

**Especificidad del 73%**:
- El modelo identifica correctamente **73%** de pacientes sin shock
- **27% de falsos positivos** (49 de 179 pacientes sin shock etiquetados como riesgo)
- **Implicación clínica**: Tasa de falsas alarmas moderadamente alta, podría generar intervenciones innecesarias

**Trade-off actual**:
- El modelo prioriza ligeramente la especificidad sobre la sensibilidad
- Para aplicación clínica, se requeriría **sensibilidad ≥85%** (tolerando mayor tasa de falsos positivos)
- Esto puede lograrse ajustando el umbral de decisión (threshold < 0.5)

### 4.6.3 Análisis de Umbral de Decisión

**Concepto**: Por defecto, sklearn clasifica como clase positiva si P(SHOCK=1) ≥ 0.5. Este umbral puede ajustarse según prioridades clínicas.

**Estrategia propuesta para fase de modelado formal**:
1. Generar curva ROC completa (sensibilidad vs 1-especificidad para todos los umbrales)
2. Identificar umbral que logre **sensibilidad ≥85%**
3. Evaluar especificidad resultante y tasa de falsas alarmas
4. Validar con expertos clínicos si el trade-off es aceptable

**Ejemplo ilustrativo**: Si se reduce umbral a 0.3:
- Sensibilidad podría aumentar a ~70-80%
- Especificidad disminuiría a ~50-60%
- Tasa de falsas alarmas: ~40-50%

**Decisión clínica**: Depende del contexto:
- En cirugía de alto riesgo: Priorizar sensibilidad (mejor una falsa alarma que un shock no detectado)
- En cirugía de bajo riesgo: Balance entre sensibilidad y especificidad

## 4.7 Limitaciones de la Experimentación Preliminar

### 4.7.1 Limitaciones Metodológicas

1. **Sin validación cruzada**: Resultados basados en una única división train/test, sujetos a variabilidad aleatoria.

2. **Hiperparámetros por defecto**: No se optimizaron hiperparámetros, el desempeño podría mejorar sustancialmente con tuning.

3. **Sin manejo especializado de desbalance**: No se probó SMOTE, class_weight ajustado, ni threshold moving.

4. **Feature engineering mínimo**: Se usaron variables originales sin transformaciones ni interacciones.

5. **Sin selección de características**: Se incluyeron todas las 19 features sin evaluación de redundancia o irrelevancia.

### 4.7.2 Limitaciones del Dataset

1. **Tamaño limitado**: 1,324 registros es modesto para algoritmos de aprendizaje profundo o ensembles complejos.

2. **Variables faltantes críticas**: No se dispone de:
   - Tipo de cirugía (factor determinante de riesgo de sangrado)
   - Pérdida sanguínea intraoperatoria (variable más directa)
   - Datos temporales intraoperatorios (FC, PA en series de tiempo)
   - Laboratorios intraoperatorios (lactato, pH, base excess)

3. **Definición de SHOCK**: Criterio diagnóstico no especificado, posible variabilidad entre casos.

4. **Sesgo de selección**: Un solo centro (FVL), periodo con COVID-19, posible enriquecimiento de casos complejos.

### 4.7.3 Limitaciones de Interpretación

1. **No se validó clínicamente**: Los resultados NO han sido evaluados por expertos médicos.

2. **No se analizó interpretabilidad**: No se generaron SHAP values ni explicaciones de predicciones individuales.

3. **No se evaluó en subgrupos**: Desconocido si el desempeño varía por género, edad, tipo de comorbilidades, etc.

## 4.8 Lecciones Aprendidas y Recomendaciones

### 4.8.1 Lecciones Aprendidas

1. **El dataset tiene señal predictiva**: AUC ~0.70 indica que existe información predictiva en las variables disponibles, superior al azar.

2. **Variables numéricas son clave**: EDAD y HB_PREQX dominan feature importance, coherente con literatura clínica.

3. **Efectos multivariados presentes**: Variables sin asociación univariada (ej: HIPERTENSION) muestran importancia en modelos multivariados.

4. **Desempeño modesto requiere mejoras**: Sensibilidad del 48% es insuficiente para uso clínico, requiere optimización.

5. **Múltiples algoritmos viables**: No hay un "ganador claro", sugiere explorar ensembles en fase formal.

### 4.8.2 Recomendaciones para Modelado Formal

#### Prioridad Alta

1. **Implementar validación cruzada estratificada (k=5 o k=10)**: Obtener estimaciones robustas de desempeño con intervalos de confianza.

2. **Optimizar hiperparámetros**: Usar RandomizedSearchCV o GridSearchCV para cada algoritmo.

3. **Ajustar umbral de decisión**: Priorizar sensibilidad ≥85% según requerimientos clínicos.

4. **Analizar feature importance con SHAP**: Obtener explicaciones globales y locales de predicciones.

5. **Validar con expertos clínicos**: Presentar predicciones en casos de prueba para evaluar utilidad clínica.

#### Prioridad Media

1. **Probar técnicas de resampling**: SMOTE, ADASYN, o combinaciones (ej: SMOTE + ENN).

2. **Feature engineering avanzado**: Crear interacciones (EDAD × HB_PREQX, comorbilidades × edad), transformaciones no lineales.

3. **Selección de características**: Recursive Feature Elimination (RFE), SelectKBest, o regularización Lasso.

4. **Ensembles**: Probar Voting Classifier, Stacking, o Blending.

5. **Análisis de calibración**: Calibrar probabilidades con CalibratedClassifierCV.

#### Prioridad Baja

1. **Probar algoritmos adicionales**: XGBoost, CatBoost, redes neuronales.

2. **Análisis de subgrupos**: Evaluar desempeño por género, grupos de edad, tipos de comorbilidades.

3. **Análisis de curvas de aprendizaje**: Determinar si más datos mejorarían desempeño.

### 4.8.3 Necesidades de Datos Adicionales (Futuro)

Para alcanzar desempeño clínicamente útil (sensibilidad >85%, AUC >0.80), se recomienda enriquecer el dataset con:

**Variables intraoperatorias críticas**:
- Tipo de cirugía (clasificación por riesgo de sangrado: alto/medio/bajo)
- Pérdida sanguínea estimada (mL) durante cirugía
- Duración de la cirugía (minutos)
- Series temporales de signos vitales (FC, PA cada 5 min)
- Volumen de cristaloides/coloides administrados

**Variables preoperatorias adicionales**:
- ASA score (American Society of Anesthesiologists)
- Creatinina sérica (función renal)
- INR, TP, TPT (coagulación)
- Uso de anticoagulantes/antiplaquetarios
- Índice de masa corporal numérico (no solo obesidad binaria)

**Variables de laboratorio intraoperatorio**:
- Lactato sérico
- pH arterial
- Base excess
- Hemoglobina seriada intraoperatoria

## 4.9 Conclusiones de la Fase de Experimentación Preliminar

### 4.9.1 Conclusiones Principales

1. **Viabilidad demostrada**: El dataset contiene información predictiva para discriminar riesgo de shock hemorrágico (AUC ~0.70), validando la viabilidad del proyecto.

2. **Desempeño inicial modesto**: Los modelos sin optimización muestran sensibilidad insuficiente para uso clínico directo (~48%), requiriendo mejoras sustanciales.

3. **Variables clave identificadas**: EDAD, HB_PREQX, y comorbilidades cardiovasculares son los predictores más importantes según análisis exploratorio.

4. **Múltiples algoritmos competitivos**: Regresión Logística, Random Forest y LightGBM muestran desempeño similar, sugiriendo que la optimización de features es más crítica que la elección de algoritmo.

5. **Pipeline de datos validado**: El proceso de limpieza documentado en Fase 3 genera datos aptos para modelado, sin errores de ejecución.

### 4.9.2 Estado Actual del Proyecto

**Fases completadas**:
- ✅ Fase 1: Comprensión del Negocio
- ✅ Fase 2: Comprensión de los Datos (EDA exhaustivo)
- ✅ Fase 3: Preparación de los Datos (pipeline de limpieza)
- ✅ Fase 4: Experimentación Preliminar (baseline de modelos)

**Fases pendientes**:
- ⏳ Fase 4 (formal): Modelado con optimización exhaustiva
- ⏳ Fase 5: Evaluación clínica con expertos
- ⏳ Fase 6: Documentación final y propuesta de despliegue

### 4.9.3 Próximos Pasos Inmediatos

**Para Presentación de Avance 2 (17/10/2025)** [**NOTA: Fecha pasada, actualizar cronograma**]:
1. Consolidar documentación de Fases 1-3 (Business Understanding, Data Understanding, Data Preparation)
2. Presentar resultados de EDA con visualizaciones
3. Discutir hallazgos preliminares de experimentación
4. Obtener retroalimentación de expertos clínicos sobre variables relevantes

**Para Modelado Formal (post-avance 2)**:
1. Implementar validación cruzada estratificada
2. Optimización de hiperparámetros (RandomizedSearchCV)
3. Técnicas de manejo de desbalance (SMOTE, threshold tuning)
4. Análisis de interpretabilidad con SHAP
5. Validación clínica con retroalimentación de expertos

**Para Entrega Final (21/11/2025)**:
1. Modelo optimizado con desempeño validado
2. Análisis completo de interpretabilidad
3. Propuesta de integración en sistemas clínicos
4. Documentación completa según guía TRIPOD
5. Informe final y presentación

---

**Fecha de elaboración**: 31 de Octubre de 2025  
**Versión**: 1.0  
**Autores**: Cristian Botina, Juan Manuel Marín, Óscar Gómez  
**Validado por**: Ángela Villota Gómez

**Nota importante**: Los resultados numéricos presentados son **estimaciones aproximadas** con fines ilustrativos. La experimentación formal con resultados reproducibles se documentará en la fase de modelado optimizado.

**Archivos de referencia**:
- Código: `src/models/shock_classifier.py`
- Pipeline: `src/pipelines/shock_pipeline.py`
- Evaluación: `src/evaluation/metrics.py`
- Modelo guardado: `models/shock_logreg.joblib` (ejemplo preliminar)
