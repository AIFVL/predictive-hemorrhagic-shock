# Fase 3: Preparación de los Datos

## Pipeline de Limpieza y Transformación - Predicción de Shock Hemorrágico

---

## 3.1 Introducción

La fase de preparación de datos constituye uno de los pilares fundamentales en el desarrollo de modelos predictivos robustos y confiables. En el contexto clínico, esta etapa adquiere relevancia crítica debido a la naturaleza heterogénea de los datos médicos, la presencia frecuente de valores faltantes, inconsistencias en registros, y la necesidad de garantizar la trazabilidad y reproducibilidad de todas las transformaciones aplicadas.

El presente documento detalla exhaustivamente el proceso de limpieza, transformación y preparación del dataset de shock hemorrágico, siguiendo principios de transparencia metodológica, documentación rigurosa, y justificación clínica de cada decisión tomada.

## 3.2 Dataset Original: Evaluación Inicial

### 3.2.1 Características del Dataset Crudo

El dataset original proporcionado por la Fundación Valle del Lili presentaba las siguientes características:

- **Archivo fuente**: `data/raw/shock.csv` (posteriormente movido a `data/processed/shock.csv`)
- **Número de registros**: 1,324 pacientes
- **Número de variables**: 22 variables
- **Período de recolección**: 2020-2024
- **Formato**: CSV con codificación UTF-8

### 3.2.2 Problemas de Calidad Identificados

Durante la inspección inicial del dataset crudo, se identificaron los siguientes problemas de calidad que requirieron intervención:

#### 1. Inconsistencia en Mayúsculas/Minúsculas (Case Sensitivity)

**Problema detectado**:
- Nombre de la variable objetivo: `SHOCK` (mayúsculas)
- Referencias en código: `shock` (minúsculas)
- Error generado: `KeyError: 'shock'` al ejecutar pipeline

**Impacto**: Fallas en ejecución de scripts, inconsistencia en procesamiento.

**Solución implementada**: Búsqueda case-insensitive en módulo `data_loader.py`:

```python
# Búsqueda case-insensitive del target
target_lower = target_col.lower()
self.target_col = next(
    (c for c in self.df.columns if c.lower() == target_lower),
    target_col  # Fallback al valor original
)
```

**Módulo afectado**: `src/utils/data_loader.py`, `src/analysis/eda.py`

#### 2. Valores Inválidos en Variables Binarias

**Problema detectado**:
- Variables **FALLA_CARDIACA** y **TABAQUISMO** contenían valores fuera del rango esperado [0, 1]
- Valores encontrados: `{0.0, 1.0, 2.0}`
- Según diccionario de datos de FVL, ambas variables deben ser **estrictamente binarias** (0 = No, 1 = Sí)

**Análisis de magnitud**:

| Variable | Total Registros | Valor = 0 | Valor = 1 | **Valor = 2** (Inválido) | % Inválidos |
|----------|----------------|-----------|-----------|--------------------------|-------------|
| FALLA_CARDIACA | 1,324 | 1,317 | 1 | **6** | 0.45% |
| TABAQUISMO | 1,324 | 1,294 | 19 | **11** | 0.83% |
| **Total** | - | - | - | **17** | **1.28%** |

**Distribución de SHOCK en registros con valores inválidos**:

```
Total filas con valores inválidos: 17
  SHOCK = 0: 9 registros (52.94%)
  SHOCK = 1: 8 registros (47.06%)
```

**Análisis de opciones de manejo**:

Se evaluaron dos estrategias principales:

**Opción 1: Eliminar registros con valores inválidos**
- **Pros**: Garantiza pureza de datos, evita supuestos sobre significado de valor 2
- **Contras**: Pérdida de 17 registros (1.28% del dataset), pérdida de 8 casos de SHOCK (1.87% de casos positivos)
- **Resultado esperado**: Dataset de 1,307 registros
  - SHOCK = 0: 888 (67.94%)
  - SHOCK = 1: 419 (32.06%)
  - Ratio de desbalance: 0.4718

**Opción 2: Recodificar valores 2 → 1 (Colapsar categorías)**
- **Pros**: Conserva todos los datos, conserva 8 casos valiosos de SHOCK
- **Contras**: Requiere justificación clínica del significado de valor 2
- **Resultado esperado**: Dataset de 1,324 registros (conservación del 100%)
  - SHOCK = 0: 897 (67.75%)
  - SHOCK = 1: 427 (32.25%)
  - Ratio de desbalance: 0.4760

**Consulta con experto clínico**:

Se consultó con el Dr. Gustavo Cruz (Anestesiólogo, FVL) sobre la interpretación del valor 2:
- **FALLA_CARDIACA = 2**: Probablemente indica "Sí, severa" o "Sí, descompensada"
- **TABAQUISMO = 2**: Probablemente indica "Sí, fumador activo" vs "Sí, ex-fumador" (valor 1)

**Justificación clínica de colapso**:

1. **Relevancia fisiopatológica**: Para predicción de riesgo quirúrgico, la distinción "Sí leve" vs "Sí severo" es menos crítica que "No" vs "Sí". La presencia de la condición es el factor determinante.

2. **Interpretación razonable**: Colapsar valor 2→1 es clínicamente sensato:
   - Falla cardíaca severa (2) → Falla cardíaca presente (1) ✓
   - Fumador activo (2) → Fumador (1) ✓

3. **Conservación de información**: Mantener 8 casos de SHOCK es valioso dado el desbalance de clases (cada caso positivo es escaso).

4. **Impacto en desbalance**: Diferencia mínima en ratio (0.4760 vs 0.4718, Δ = 0.0042).

**Decisión final**: **Opción 2 - Recodificar 2→1**

**Trazabilidad**: Decisión documentada y aprobada en reunión de equipo del 30/10/2025.

#### 3. Valores Nulos (Manejo Previo a Limpieza Final)

**Nota importante**: El dataset final (`shock.csv`) **NO contiene valores nulos** (0% missingness). Sin embargo, durante el proceso de curación de datos interno de FVL (fase previa a la entrega del dataset al equipo), se realizó manejo de valores faltantes mediante:

- **Imputación por mediana**: Variables numéricas con <5% valores faltantes
- **Remoción de registros**: Variables con >30% valores faltantes
- **Imputación por moda**: Variables categóricas con <10% valores faltantes

Esta información fue proporcionada por el Centro de Investigaciones Clínicas de FVL y se documenta aquí para trazabilidad completa del proceso.

## 3.3 Pipeline de Limpieza Implementado

### 3.3.1 Arquitectura del Pipeline

El pipeline de limpieza se implementó en el módulo `src/preprocessing/data_cleaner.py` mediante la clase `ShockDataCleaner`, siguiendo principios de programación orientada a objetos, modularidad y reproducibilidad.

**Estructura del pipeline**:

```
┌─────────────────────────────────────────┐
│  1. Carga de datos (CSV)                │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  2. Detección de variable objetivo      │
│     (búsqueda case-insensitive)         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  3. Identificación de tipos de vars     │
│     - Continuas (EDAD, HB_PREQX)        │
│     - Binarias (16 vars)                │
│     - Categóricas (2 vars)              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  4. Limpieza de valores inválidos       │
│     - Recodificación 2→1                │
│     - Coerción a tipos numéricos        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  5. Identificación de data leakage      │
│     - Detectar vars postoperatorias     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  6. Preparación de features             │
│     - Exclusión de ID, target, leakage  │
│     - Remoción de varianza cero         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  7. Generación de X, y                  │
└─────────────────────────────────────────┘
```

### 3.3.2 Métodos de la Clase ShockDataCleaner

#### Método 1: `__init__()` - Inicialización

```python
def __init__(self, target_col: str = None, postop_keywords: List[str] = None):
    """
    Inicializa el limpiador de datos.
    
    Args:
        target_col: Nombre de la columna objetivo
        postop_keywords: Palabras clave para identificar variables postoperatorias
    """
    self.target_col = target_col
    self.postop_keywords = postop_keywords or ['sangrad', 'sangrado', 'muerte', 'mortalidad']
    self.continuous_cols = []
    self.categorical_cols = []
    self.features = []
    self.postop_vars = []
    self.n_rows_removed = 0  # Contador de registros removidos
```

**Justificación**: Centraliza configuración y mantiene estado del proceso de limpieza.

#### Método 2: `detect_target()` - Detección de Variable Objetivo

```python
def detect_target(self, df: pd.DataFrame) -> str:
    """Detecta la columna objetivo (case-insensitive)."""
    target_col = next((c for c in df.columns if 'shock' in c.lower()), None)
    if target_col is None:
        raise ValueError("No se encontro columna 'shock'.")
    self.target_col = target_col
    return target_col
```

**Función**: Búsqueda robusta de variable objetivo, tolerante a variaciones de capitalización.

#### Método 3: `identify_leakage_vars()` - Identificación de Data Leakage

```python
def identify_leakage_vars(self, df: pd.DataFrame) -> List[str]:
    """Identifica variables postoperatorias que pueden causar data leakage."""
    self.postop_vars = [
        c for c in df.columns
        if any(x in c.lower() for x in self.postop_keywords)
    ]
    return self.postop_vars
```

**Variables identificadas como potencial leakage**:
- `SANGRADO_MAYOR` (contiene 'sangrado')
- `MUERTE.1` (contiene 'muerte')

**Justificación**:
- **MUERTE.1**: Claramente es un outcome posterior (mortalidad), no puede usarse para predecir shock (que ocurre antes).
- **SANGRADO_MAYOR**: **Requiere validación temporal**. Si se registra simultáneamente o después del shock, constituye leakage. Pendiente de confirmación con expertos clínicos.

**Acción**: Estas variables se excluyen automáticamente de las features predictoras.

#### Método 4: `identify_feature_types()` - Tipificación de Variables

```python
def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifica tipos de features (continuas vs categoricas)."""
    cont_keywords = ['edad', 'age', 'hb', 'hemoglob', 'hb_pre', 'globul']
    cont_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in cont_keywords)
    ]
    self.continuous_cols = [c for c in cont_candidates if c != self.target_col]
    
    # Features: excluir ID, target y variables postoperatorias
    exclude = ['CODIGO', 'codigo', self.target_col] + self.postop_vars
    self.features = [c for c in df.columns if c not in exclude]
    
    return self.continuous_cols, self.features
```

**Resultado**:
- **Continuas**: `['EDAD', 'HB_PREQX']`
- **Features totales**: 19 variables (excluyendo CODIGO, SHOCK, SANGRADO_MAYOR, MUERTE.1)

#### Método 5: `remove_invalid_binary_values()` - **LIMPIEZA CRÍTICA**

Este es el método más importante del pipeline de limpieza. Se documenta exhaustivamente:

```python
def remove_invalid_binary_values(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Recodifica valores invalidos en variables binarias.
    
    Segun el diccionario de datos de la clinica, ciertas variables
    categoricas deben ser binarias (0 o 1), pero pueden tener valores
    erroneos (ej: 2). 
    
    DECISION: En lugar de eliminar filas, se COLAPSAN valores 2 -> 1
    
    JUSTIFICACION:
    - Conserva 17 filas adicionales (1.28% mas de datos)
    - Conserva 8 casos de SHOCK (casi 2% de casos positivos)
    - No afecta ratio de desbalance (0.4760 vs 0.4718)
    - Interpretacion clinica razonable:
      * FALLA_CARDIACA: valor 2 probablemente = "Si, severa"
      * TABAQUISMO: valor 2 probablemente = "Si, fumador actual"
      * Colapsar a 1 (Si) tiene sentido clinico
    
    Analisis de las 17 filas con valores invalidos:
    - SHOCK = 0: 9 filas (52.94%)
    - SHOCK = 1: 8 filas (47.06%)
    - Distribucion casi balanceada (contrario al dataset: 67.75% / 32.25%)
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        pd.DataFrame: DataFrame con valores recodificados
    """
    df_clean = df.copy()
    
    # Variables que segun diccionario de datos deben ser binarias (0 o 1)
    binary_vars = ['FALLA_CARDIACA', 'TABAQUISMO']
    
    total_recodificados = 0
    
    for var in binary_vars:
        if var in df_clean.columns:
            # Identificar valores invalidos (diferentes de 0 y 1)
            valores_invalidos = df_clean[~df_clean[var].isin([0, 1, 0.0, 1.0])][var]
            
            if len(valores_invalidos) > 0:
                print(f"  Variable '{var}': {len(valores_invalidos)} valores recodificados")
                print(f"    Valores encontrados: {valores_invalidos.value_counts().to_dict()}")
                
                # RECODIFICAR: 2 -> 1 (cualquier valor != 0 se convierte a 1)
                df_clean.loc[df_clean[var] > 1, var] = 1
                total_recodificados += len(valores_invalidos)
    
    if total_recodificados > 0:
        print(f"\nTotal valores recodificados a 1: {total_recodificados}")
        print(f"Filas conservadas: {len(df_clean)} (100% del dataset original)")
    
    return df_clean
```

**Output del método en ejecución**:

```
Recodificacion de valores invalidos en variables binarias:
  Variable 'FALLA_CARDIACA': 6 valores recodificados
    Valores encontrados: {2.0: 6}
  Variable 'TABAQUISMO': 11 valores recodificados
    Valores encontrados: {2.0: 11}

Total valores recodificados a 1: 17
Filas conservadas: 1324 (100% del dataset original)
```

**Impacto en distribuciones**:

| Variable | Pre-Limpieza | Post-Limpieza |
|----------|--------------|---------------|
| **FALLA_CARDIACA** | {0: 1317, 1: 1, 2: 6} | {0.0: 1317, 1.0: 7} |
| **TABAQUISMO** | {0: 1294, 1: 19, 2: 11} | {0.0: 1294, 1.0: 30} |

**Validación post-limpieza**:
- ✅ FALLA_CARDIACA: solo valores [0.0, 1.0]
- ✅ TABAQUISMO: solo valores [0.0, 1.0]
- ✅ Todas las demás variables binarias: solo valores [0.0, 1.0]

#### Método 6: `clean_data()` - Orquestación de Limpieza

```python
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara los datos."""
    print("\nRecodificacion de valores invalidos en variables binarias:")
    df_clean = self.remove_invalid_binary_values(df)
    
    # Coercer columnas continuas a numericas
    for c in self.continuous_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    
    # Coercer features a numericas
    for c in self.features:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    
    # Asegurar que target sea binario 0/1
    df_clean[self.target_col] = (
        pd.to_numeric(df_clean[self.target_col], errors='coerce') > 0
    ).astype(int)
    
    return df_clean
```

**Funciones**:
1. Ejecuta limpieza de valores inválidos
2. Convierte todas las variables a tipos numéricos (facilita procesamiento ML)
3. Asegura que target es estrictamente binario (0/1) tipo int

#### Método 7: `remove_zero_variance_features()` - Remoción de Variables Constantes

```python
def remove_zero_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
    """Elimina features con varianza cero."""
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        print(f"Eliminando {len(zero_var)} features con varianza cero: {zero_var}")
        X = X.drop(columns=zero_var)
        self.features = [f for f in self.features if f not in zero_var]
    return X
```

**Justificación**: Variables sin variabilidad (todas constantes) no aportan información predictiva y pueden causar problemas numéricos en algoritmos de ML.

**Resultado en dataset actual**: No se detectaron variables de varianza cero.

#### Método 8: `prepare_features()` - Método Principal

```python
def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara features y target para modelado."""
    # Detectar target si no esta definido
    if self.target_col is None:
        self.detect_target(df)
    
    # Identificar variables de leakage
    self.identify_leakage_vars(df)
    
    # Identificar tipos de features
    self.identify_feature_types(df)
    
    # Limpiar datos
    df_clean = self.clean_data(df)
    
    # Preparar X e y
    X = df_clean[self.features].copy()
    y = df_clean[self.target_col].copy()
    
    # Eliminar features de varianza cero
    X = self.remove_zero_variance_features(X)
    
    return X, y
```

**Output**: Tupla (X, y) lista para modelado
- **X**: DataFrame con 19 features (numéricas, valores en [0, 1])
- **y**: Series con variable objetivo SHOCK (0/1)

### 3.3.3 Script de Validación de Limpieza

Se desarrolló un script independiente (`test_data_cleaning.py`) para validar el proceso de limpieza:

**Código completo**:

```python
"""
Script de validación del proceso de limpieza de datos.
Valida la recodificación de valores inválidos en variables binarias.
"""

import pandas as pd
import numpy as np

# Cargar dataset original
df = pd.read_csv('data/processed/shock.csv')

print("="*70)
print("VALIDACIÓN DE LIMPIEZA DE DATOS - SHOCK HEMORRÁGICO")
print("="*70)

print(f"\nDataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

# Distribución de SHOCK original
shock_dist = df['SHOCK'].value_counts().sort_index()
print(f"\nSHOCK original:")
for val, count in shock_dist.items():
    pct = count / len(df) * 100
    print(f"  {val} = {count} ({pct:.2f}%)")

# Análisis de variables binarias problemáticas
print("\n" + "="*70)
print("ANÁLISIS DE VARIABLES BINARIAS")
print("="*70)

binary_vars = ['FALLA_CARDIACA', 'TABAQUISMO']

for var in binary_vars:
    unique_vals = df[var].unique()
    value_counts = df[var].value_counts().sort_index()
    
    print(f"\n{var}:")
    print(f"  Valores únicos: {sorted(unique_vals)}")
    print(f"  Distribución:")
    for val, count in value_counts.items():
        print(f"    {val}: {count}")

# Aplicar limpieza
print("\n" + "="*70)
print("APLICANDO LIMPIEZA")
print("="*70)

from src.preprocessing.data_cleaner import ShockDataCleaner

cleaner = ShockDataCleaner(target_col='SHOCK')
df_clean = cleaner.clean_data(df.copy())

# Validar limpieza
print("\n" + "="*70)
print("VALIDACIÓN POST-LIMPIEZA")
print("="*70)

print(f"\nDataset limpio: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")

# Distribución de SHOCK post-limpieza
shock_dist_clean = df_clean['SHOCK'].value_counts().sort_index()
print(f"\nSHOCK final:")
for val, count in shock_dist_clean.items():
    pct = count / len(df_clean) * 100
    print(f"  {val} = {count} ({pct:.2f}%)")

# Calcular ratio de desbalance
minority = shock_dist_clean.min()
majority = shock_dist_clean.max()
ratio = minority / majority
print(f"\nRatio desbalance: {ratio:.4f}")

# Verificar variables binarias limpias
print(f"\nVerificación de variables binarias:")
for var in binary_vars:
    unique_vals = sorted(df_clean[var].unique())
    value_counts = df_clean[var].value_counts().sort_index()
    
    print(f"\n{var}:")
    print(f"  Valores únicos: {unique_vals}")
    print(f"  Distribución:")
    for val, count in value_counts.items():
        print(f"    {val}: {count}")
    
    # Validar que solo hay 0.0 y 1.0
    if set(unique_vals) == {0.0, 1.0}:
        print(f"  ✓ VÁLIDA (solo 0.0 y 1.0)")
    else:
        print(f"  ✗ INVÁLIDA (contiene valores no binarios)")

# Guardar dataset limpio
output_path = 'data/processed/shock_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nDataset limpio guardado en: {output_path}")
```

**Ejecución y output**:

```bash
$ python test_data_cleaning.py

======================================================================
VALIDACIÓN DE LIMPIEZA DE DATOS - SHOCK HEMORRÁGICO
======================================================================

Dataset original: 1324 filas x 22 columnas

SHOCK original:
  0.0 = 897 (67.75%)
  1.0 = 427 (32.25%)

======================================================================
ANÁLISIS DE VARIABLES BINARIAS
======================================================================

FALLA_CARDIACA:
  Valores únicos: [0.0, 1.0, 2.0]
  Distribución:
    0.0: 1317
    1.0: 1
    2.0: 6

TABAQUISMO:
  Valores únicos: [0.0, 1.0, 2.0]
  Distribución:
    0.0: 1294
    1.0: 19
    2.0: 11

======================================================================
APLICANDO LIMPIEZA
======================================================================

Recodificacion de valores invalidos en variables binarias:
  Variable 'FALLA_CARDIACA': 6 valores recodificados
    Valores encontrados: {2.0: 6}
  Variable 'TABAQUISMO': 11 valores recodificados
    Valores encontrados: {2.0: 11}

Total valores recodificados a 1: 17
Filas conservadas: 1324 (100% del dataset original)

======================================================================
VALIDACIÓN POST-LIMPIEZA
======================================================================

Dataset limpio: 1324 filas x 22 columnas

SHOCK final:
  0 = 897 (67.75%)
  1 = 427 (32.25%)

Ratio desbalance: 0.4760

Verificación de variables binarias:

FALLA_CARDIACA:
  Valores únicos: [0.0, 1.0]
  Distribución:
    0.0: 1317
    1.0: 7
  ✓ VÁLIDA (solo 0.0 y 1.0)

TABAQUISMO:
  Valores únicos: [0.0, 1.0]
  Distribución:
    0.0: 1294
    1.0: 30
  ✓ VÁLIDA (solo 0.0 y 1.0)

Dataset limpio guardado en: data/processed/shock_cleaned.csv
```

## 3.4 Transformaciones Aplicadas

### 3.4.1 Resumen de Transformaciones

| Transformación | Variables Afectadas | Justificación | Resultado |
|----------------|---------------------|---------------|-----------|
| **Búsqueda case-insensitive** | SHOCK (target) | Robustez a variaciones de capitalización | ✓ Ejecución sin errores |
| **Recodificación 2→1** | FALLA_CARDIACA, TABAQUISMO | Valores inválidos, justificación clínica | 17 valores corregidos |
| **Coerción a numérico** | Todas las variables | Compatibilidad con sklearn | 21 variables numéricas |
| **Binarización de target** | SHOCK | Garantizar target binario 0/1 (int) | SHOCK ∈ {0, 1} |
| **Exclusión de leakage** | SANGRADO_MAYOR, MUERTE.1 | Prevenir contaminación de datos futuros | 19 features finales |
| **Exclusión de ID** | CODIGO | No es predictor | 19 features finales |

### 3.4.2 No Se Aplicaron (con Justificación)

| Transformación | Razón de No Aplicación |
|----------------|------------------------|
| **Imputación de valores nulos** | Dataset final no tiene nulos (0% missingness) |
| **Normalización/Estandarización** | Se aplicará en fase de modelado dentro de pipelines de sklearn |
| **One-Hot Encoding** | Variables categóricas (ACT_FISICA_METS) se codificarán en modelado según algoritmo |
| **Remoción de outliers** | No se detectaron outliers extremos; valores fisiológicamente plausibles |
| **Balanceo de clases** | Se manejará en fase de modelado (class_weight, métricas ajustadas) |

## 3.5 Dataset Limpio Final

### 3.5.1 Características del Dataset Limpio

- **Archivo de salida**: `data/processed/shock_cleaned.csv`
- **Número de registros**: 1,324 (conservación del 100%)
- **Número de variables**: 22 (sin cambios)
- **Número de features para modelado**: 19 (excluyendo CODIGO, SHOCK, SANGRADO_MAYOR, MUERTE.1)
- **Completitud**: 100% (sin valores nulos)
- **Variables binarias**: 100% válidas (solo valores 0.0 y 1.0)

### 3.5.2 Estadísticas del Dataset Limpio

**Variable objetivo (SHOCK)**:
- Clase 0: 897 registros (67.75%)
- Clase 1: 427 registros (32.25%)
- Ratio de desbalance: 0.4760

**Variables numéricas**:
- EDAD: [18, 95] años, media = 55.52, DE = 20.04
- HB_PREQX: [6.5, 18.2] g/dL, media = 12.93, DE = 2.01

**Variables binarias (16)**:
- Todas con valores exclusivamente en {0.0, 1.0}
- Sin valores faltantes
- Distribuciones consistentes con análisis EDA

**Variables categóricas (2)**:
- ACT_FISICA_METS: 5 niveles ordinales
- (SANGRADO_MAYOR excluida de features por potencial leakage)

### 3.5.3 Validación de Calidad Final

| Criterio | Status | Evidencia |
|----------|--------|-----------|
| **Completitud** | ✅ PASS | 0% valores nulos |
| **Consistencia** | ✅ PASS | Todas las binarias en [0, 1] |
| **Unicidad** | ✅ PASS | 0 duplicados |
| **Validez** | ✅ PASS | Rangos fisiológicamente plausibles |
| **Trazabilidad** | ✅ PASS | Pipeline documentado y reproducible |
| **Reproducibilidad** | ✅ PASS | Script de validación ejecutado exitosamente |

## 3.6 Feature Engineering Preliminar

### 3.6.1 Variables Compuestas Creadas (Exploración)

Durante la fase de EDA (Fase 2), se exploraron las siguientes variables compuestas:

**1. Número de Comorbilidades (n_comorbilidades)**:
- **Definición**: Suma de 14 variables binarias de comorbilidades
- **Resultado EDA**: No mostró asociación significativa con SHOCK (p = 0.38)
- **Decisión**: No incluir en modelado inicial, evaluar en feature selection

**2. Score de Síndrome Metabólico (pendiente)**:
- **Definición propuesta**: DIABETES + OBESIDAD + HIPERTENSION
- **Justificación**: Cluster de comorbilidades con fisiopatología compartida
- **Status**: Pendiente de implementación en fase de modelado

**3. Categorización de Edad (pendiente)**:
- **Definición propuesta**: <40 años (bajo riesgo), 40-65 (moderado), >65 (alto)
- **Justificación**: Umbrales clínicos para riesgo quirúrgico
- **Status**: Pendiente de evaluación comparativa vs EDAD continua

### 3.6.2 Estrategia de Feature Engineering para Modelado

El feature engineering final se realizará de manera **iterativa y basada en datos** durante la fase de modelado, siguiendo esta estrategia:

1. **Baseline sin feature engineering**: Usar las 19 features originales limpias
2. **Análisis de feature importance**: Identificar variables más predictivas
3. **Creación de interacciones**: Basadas en conocimiento clínico y feature importance
4. **Validación de contribución**: Evaluar si nuevas features mejoran métricas de desempeño
5. **Selección final**: Mantener solo features que aporten valor predictivo

## 3.7 Preparación para Modelado

### 3.7.1 División de Datos (Train/Test)

La división de datos se realizará en la fase de modelado con las siguientes especificaciones:

- **Estratificación**: Por variable objetivo SHOCK (mantener proporción 67.75% / 32.25%)
- **Proporción**: 80% entrenamiento, 20% prueba
- **Semilla aleatoria**: `random_state=42` (reproducibilidad)
- **Método**: `train_test_split` de sklearn

**Tamaños esperados**:
- Entrenamiento: 1,059 registros (SHOCK=0: 718, SHOCK=1: 341)
- Prueba: 265 registros (SHOCK=0: 179, SHOCK=1: 86)

### 3.7.2 Validación Cruzada

Se implementará validación cruzada estratificada:
- **Método**: StratifiedKFold
- **Número de folds**: k=5
- **Estratificación**: Por SHOCK
- **Métricas a reportar**: Media ± desviación estándar

### 3.7.3 Pipelines de Preprocesamiento por Algoritmo

Se definirán pipelines específicos según el algoritmo de ML:

**Para Regresión Logística**:
```python
Pipeline([
    ('scaler', StandardScaler()),  # Normalización necesaria
    ('model', LogisticRegression(
        class_weight='balanced',
        penalty='l1',  # Lasso para selección de variables
        solver='liblinear',
        random_state=42
    ))
])
```

**Para Random Forest**:
```python
Pipeline([
    ('model', RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    ))
])
# No requiere escalado (tree-based)
```

**Para LightGBM**:
```python
Pipeline([
    ('model', LGBMClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    ))
])
# Maneja variables categóricas nativamente
```

## 3.8 Control de Calidad y Trazabilidad

### 3.8.1 Versionado de Datos

| Versión | Archivo | Fecha | Descripción |
|---------|---------|-------|-------------|
| v0.1 | `shock.csv` (raw) | Oct 2025 | Dataset original de FVL (recibido) |
| v1.0 | `shock.csv` (processed) | Oct 2025 | Dataset curado por FVL (sin nulos) |
| v1.1 | `shock_cleaned.csv` | 30/10/2025 | Dataset post-limpieza (valores binarios corregidos) |

**Política de versionado**: 
- Todos los datasets se conservan en `data/` con nombres descriptivos
- No se sobrescriben versiones anteriores
- Git rastrea cambios en código de limpieza (`data_cleaner.py`)

### 3.8.2 Documentación de Decisiones

Todas las decisiones críticas de limpieza están documentadas en:

1. **Código fuente**: Docstrings exhaustivos en `data_cleaner.py`
2. **Este documento**: Justificaciones clínicas y metodológicas
3. **Commits de Git**: Mensajes descriptivos de cambios
4. **Reuniones de equipo**: Actas con aprobaciones de stakeholders

### 3.8.3 Reproducibilidad Garantizada

**Reproducibilidad completa del pipeline**:

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd shock

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline de limpieza
python -c "
from src.preprocessing.data_cleaner import ShockDataCleaner
import pandas as pd

df = pd.read_csv('data/processed/shock.csv')
cleaner = ShockDataCleaner(target_col='SHOCK')
X, y = cleaner.prepare_features(df)

print(f'Features: {X.shape}')
print(f'Target: {y.shape}')
print(f'Ratio desbalance: {y.value_counts()[1]/y.value_counts()[0]:.4f}')
"

# 4. Validar limpieza
python test_data_cleaning.py
```

**Output esperado**:
```
Features: (1324, 19)
Target: (1324,)
Ratio desbalance: 0.4760
```

## 3.9 Limitaciones y Supuestos

### 3.9.1 Limitaciones Identificadas

1. **Dataset retrospectivo**:
   - Limitado a registros de un solo centro (FVL)
   - Período 2020-2024 incluye pandemia COVID-19 (potencial sesgo)
   - No se capturan variables dinámicas intraoperatorias (series temporales)

2. **Variables faltantes**:
   - No se dispone de información sobre tipo específico de cirugía
   - Ausencia de pérdida sanguínea estimada (mL)
   - Falta de datos de laboratorio intraoperatorios (lactato, pH)

3. **Definición de SHOCK**:
   - No se especifica criterio exacto utilizado para etiquetar casos
   - Posible variabilidad entre clínicos en diagnóstico
   - No se distingue severidad del shock (leve, moderado, severo)

4. **Sesgo de supervivencia**:
   - Solo se incluyen pacientes que llegaron a sala de recuperación
   - Muertes intraoperatorias pueden estar subrepresentadas

### 3.9.2 Supuestos Críticos

1. **Calidad de etiquetado**:
   - Se asume que la variable SHOCK fue diagnosticada por personal médico competente
   - Se asume criterio consistente de diagnóstico (aunque no especificado)

2. **Independencia temporal**:
   - Se asume que casos son independientes entre sí
   - No hay información de pacientes con múltiples cirugías (posible duplicación)

3. **Representatividad**:
   - Se asume que la muestra es representativa de la población quirúrgica de FVL
   - Se asume que hallazgos son generalizables a contextos similares (hospitales de tercer nivel)

4. **Validez de recodificación**:
   - Se asume que recodificación 2→1 en FALLA_CARDIACA y TABAQUISMO es clínicamente razonable
   - Se requiere validación prospectiva con expertos

## 3.10 Conclusiones de la Fase de Preparación

### 3.10.1 Logros Principales

1. ✅ **Dataset limpio y consistente**: 1,324 registros, 100% completos, variables binarias validadas
2. ✅ **Pipeline reproducible**: Código modular, documentado y versionado en Git
3. ✅ **Trazabilidad completa**: Todas las decisiones justificadas y documentadas
4. ✅ **Prevención de data leakage**: Variables postoperatorias excluidas
5. ✅ **Validación exitosa**: Script de prueba confirma integridad del proceso

### 3.10.2 Dataset Listo para Modelado

El dataset `shock_cleaned.csv` está listo para la fase de modelado con las siguientes características:

- **Dimensiones**: 1,324 registros × 19 features + 1 target
- **Calidad**: 100% completitud, 100% consistencia en variables binarias
- **Balance**: Ratio 0.476 (manejable con técnicas estándar)
- **Formato**: CSV numérico compatible con sklearn

### 3.10.3 Próximos Pasos

**Fase 4: Modelado y Experimentación**
1. División estratificada train/test
2. Entrenamiento de modelos baseline (Regresión Logística, Random Forest, LightGBM)
3. Optimización de hiperparámetros
4. Análisis de feature importance
5. Evaluación con métricas clínicas (sensibilidad, F1, AUC-ROC)

**Validaciones Pendientes**:
- Confirmar relación temporal de SANGRADO_MAYOR con expertos clínicos
- Evaluar si variables excluidas (MUERTE.1) pueden usarse para análisis post-hoc de utilidad clínica

---

**Fecha de elaboración**: 30 de Octubre de 2025  
**Versión**: 1.0  
**Autores**: Cristian Botina, Juan Manuel Marín, Óscar Gómez  
**Validado por**: Ángela Villota Gómez

**Archivos de referencia**:
- Código: `src/preprocessing/data_cleaner.py`
- Script de validación: `test_data_cleaning.py`
- Dataset limpio: `data/processed/shock_cleaned.csv`
- Dataset original: `data/processed/shock.csv`
