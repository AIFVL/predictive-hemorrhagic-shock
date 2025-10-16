"""
Funciones auxiliares para el proyecto de prediccion de shock hemorragico.
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    """
    Calcula Cramer's V para asociacion categorica (robusto a varianza cero).

    Args:
        x: Serie categorica 1
        y: Serie categorica 2

    Returns:
        float: Valor de Cramer's V entre 0 y 1
    """
    confusion = pd.crosstab(x, y)
    if confusion.size == 0 or confusion.shape[0] == 1 or confusion.shape[1] == 1:
        return 0.0
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    # Correccion
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min(kcorr-1, rcorr-1)
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)


def bootstrap_ci(metric_func, y_true, y_pred, n_boot=1000, alpha=0.05, seed=42):
    """
    Calcula intervalo de confianza bootstrap para una metrica dada.

    Args:
        metric_func: Funcion que acepta (y_true, y_pred) y retorna una metrica
        y_true: Valores verdaderos
        y_pred: Valores predichos
        n_boot: Numero de iteraciones bootstrap
        alpha: Nivel de significancia
        seed: Semilla aleatoria

    Returns:
        tuple: (limite_inferior, limite_superior)
    """
    stats = []
    n = len(y_true)
    rng = np.random.RandomState(seed)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        stats.append(metric_func(y_true[idx], y_pred[idx]))
    lower = np.percentile(stats, 100*alpha/2)
    upper = np.percentile(stats, 100*(1-alpha/2))
    return lower, upper


def save_json(obj, path):
    """
    Guarda un objeto como JSON.

    Args:
        obj: Objeto a guardar
        path: Ruta del archivo
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    """
    Carga un objeto desde JSON.

    Args:
        path: Ruta del archivo

    Returns:
        dict: Objeto cargado
    """
    with open(path, "r") as f:
        return json.load(f)
