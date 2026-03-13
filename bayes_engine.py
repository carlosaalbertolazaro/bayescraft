"""
bayes_engine.py
───────────────
Toda la lógica matemática y de Machine Learning de BayesCraft.
Contiene: detección de columnas, cálculos bayesianos, Naive Bayes,
          preparación de datos para gráficas.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1. DETECCIÓN DE TIPOS DE COLUMNA
# ══════════════════════════════════════════════════════════════

def detectar_columnas(df: pd.DataFrame) -> dict:
    """
    Analiza el DataFrame y clasifica cada columna en:
    - numericas   : valores continuos/discretos numéricos
    - categoricas : texto con múltiples categorías
    - fechas      : columnas de tipo datetime
    - binarias    : columnas con exactamente 2 valores (0/1, Sí/No)

    Retorna un dict con listas de nombres de columna por tipo.
    """
    info = {"numericas": [], "categoricas": [], "fechas": [], "binarias": []}

    for col in df.columns:
        serie = df[col].dropna()

        # ── Intentar parsear como fecha ──────────────────────
        if df[col].dtype == "object":
            try:
                pd.to_datetime(serie, infer_datetime_format=True)
                info["fechas"].append(col)
                continue
            except Exception:
                pass

        # ── Ya es datetime ───────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            info["fechas"].append(col)
            continue

        # ── Numéricas ────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(df[col]):
            valores_unicos = set(serie.unique())
            if valores_unicos.issubset({0, 1, 0.0, 1.0, True, False}):
                info["binarias"].append(col)
            else:
                info["numericas"].append(col)

        # ── Texto / Categórico ────────────────────────────────
        else:
            vals_lower = serie.astype(str).str.lower().unique()
            binarias_texto = {"si", "no", "yes", "sí", "true", "false",
                              "verdadero", "falso", "1", "0", "si ", "no "}
            if set(vals_lower).issubset(binarias_texto):
                info["binarias"].append(col)
            else:
                info["categoricas"].append(col)

    return info


def resumen_dataset(df: pd.DataFrame) -> dict:
    """Estadísticas rápidas del dataset cargado."""
    return {
        "filas":    len(df),
        "columnas": len(df.columns),
        "nulos":    int(df.isnull().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }


# ══════════════════════════════════════════════════════════════
# 2. ANÁLISIS BAYESIANO
# ══════════════════════════════════════════════════════════════

def _binarizar_serie(serie: pd.Series) -> pd.Series:
    """Convierte una serie a valores 0/1 de forma inteligente."""
    s = serie.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    # Texto → 0/1
    mapa = {}
    for v in s.dropna().unique():
        vl = str(v).lower().strip()
        if vl in {"1", "si", "sí", "yes", "true", "verdadero"}:
            mapa[v] = 1
        else:
            mapa[v] = 0
    return s.map(mapa).fillna(0).astype(int)


def calcular_bayes(df: pd.DataFrame, target_col: str,
                   feature_col: str, umbral=None) -> dict:
    """
    Aplica el Teorema de Bayes para dos variables.

    Calcula:
      P(A)      = P(Fallo)
      P(B)      = P(Evidencia > umbral)
      P(B|A)    = P(Evidencia | Fallo)
      P(B|¬A)   = P(Evidencia | No Fallo)
      P(A|B)    = [P(B|A) × P(A)] / P(B)   ← Teorema de Bayes

    Retorna dict con todas las probabilidades y metadatos.
    """
    df_clean = df[[target_col, feature_col]].dropna()
    n = len(df_clean)
    if n == 0:
        raise ValueError("No hay datos válidos para el análisis.")

    # Target binario
    target = _binarizar_serie(df_clean[target_col])
    n_fallo = int(target.sum())

    if n_fallo == 0:
        raise ValueError(f"La variable '{target_col}' no contiene eventos positivos (fallos).")

    P_A = n_fallo / n

    # Evidencia binaria
    col_data = df_clean[feature_col]
    if pd.api.types.is_numeric_dtype(col_data):
        if umbral is None:
            umbral = float(col_data.median())
        evidencia = (col_data > umbral).astype(int)
        label_B = f"{feature_col} > {umbral:.2f}"
    else:
        le = LabelEncoder()
        enc = le.fit_transform(col_data.astype(str))
        evidencia = pd.Series((enc > 0).astype(int), index=df_clean.index)
        label_B = f"{feature_col} activo"

    n_B = int(evidencia.sum())
    P_B = n_B / n if n > 0 else 0

    mask_fallo   = (target == 1)
    mask_no_fallo = (target == 0)

    n_fallo_total    = mask_fallo.sum()
    n_no_fallo_total = mask_no_fallo.sum()

    P_B_dado_A     = float(evidencia[mask_fallo].sum())   / n_fallo_total    if n_fallo_total > 0    else 0.0
    P_B_dado_no_A  = float(evidencia[mask_no_fallo].sum()) / n_no_fallo_total if n_no_fallo_total > 0 else 0.0

    # Teorema de Bayes
    P_A_dado_B = (P_B_dado_A * P_A) / P_B if P_B > 0 else 0.0
    P_A_dado_B = min(P_A_dado_B, 1.0)

    # Factor de actualización (cuánto cambia la creencia)
    delta        = P_A_dado_B - P_A
    factor_bayes = P_A_dado_B / P_A if P_A > 0 else 1.0

    return {
        "P_A":           round(P_A, 6),
        "P_B":           round(P_B, 6),
        "P_B_dado_A":    round(P_B_dado_A, 6),
        "P_B_dado_no_A": round(P_B_dado_no_A, 6),
        "P_A_dado_B":    round(P_A_dado_B, 6),
        "delta":         round(delta, 6),
        "factor_bayes":  round(factor_bayes, 4),
        "label_B":       label_B,
        "umbral":        umbral,
        "n_total":       n,
        "n_fallo":       n_fallo,
        "target_series": target,
        "evidencia_series": evidencia,
    }


def calcular_probabilidades_multiples(df: pd.DataFrame, target_col: str,
                                      feature_cols: list) -> pd.DataFrame:
    """
    Calcula P(Fallo | cada feature) para todas las features numéricas.
    Útil para comparar qué variable tiene más impacto.
    """
    rows = []
    for fc in feature_cols:
        try:
            res = calcular_bayes(df, target_col, fc)
            rows.append({
                "Feature":        fc,
                "P(Fallo)":       res["P_A"],
                "P(Feature>med)": res["P_B"],
                "P(F|Fallo)":     res["P_B_dado_A"],
                "P(Fallo|F)":     res["P_A_dado_B"],
                "Factor Bayes":   res["factor_bayes"],
                "Δ Riesgo":       res["delta"],
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 3. CURVA DE PROBABILIDAD POSTERIOR
# ══════════════════════════════════════════════════════════════

def curva_posterior(df: pd.DataFrame, target_col: str,
                    feature_col: str, n_puntos: int = 60) -> dict:
    """
    Calcula P(Fallo | feature > umbral) para una grilla de umbrales.
    Devuelve arrays para graficar la curva posterior.
    """
    df_clean = df[[target_col, feature_col]].dropna()
    target   = _binarizar_serie(df_clean[target_col])
    col_data = df_clean[feature_col]

    if not pd.api.types.is_numeric_dtype(col_data):
        return {}

    P_A  = target.mean()
    xs   = np.linspace(col_data.min(), col_data.max(), n_puntos)
    post = []

    for x_val in xs:
        ev   = (col_data > x_val).astype(int)
        p_b  = ev.mean()
        p_bga = ev[target == 1].mean() if target.sum() > 0 else 0
        posterior = (p_bga * P_A / p_b) if p_b > 0 else P_A
        post.append(min(posterior, 1.0))

    return {"xs": xs, "posterior": np.array(post), "P_A": P_A}


# ══════════════════════════════════════════════════════════════
# 4. NAIVE BAYES CLASIFICADOR
# ══════════════════════════════════════════════════════════════

def entrenar_naive_bayes(df: pd.DataFrame, target_col: str,
                         feature_cols: list) -> dict:
    """
    Entrena un Clasificador Gaussiano Naive Bayes.

    P(Fallo | x₁, x₂, ..., xₙ) ∝ P(Fallo) × ∏ P(xᵢ | Fallo)

    Retorna:
      - confusion_matrix
      - accuracy, sensibilidad, especificidad
      - y_test, y_pred para gráficas
    """
    if len(feature_cols) < 1:
        raise ValueError("Necesitas al menos 1 feature.")

    df_clean = df[feature_cols + [target_col]].dropna()
    if len(df_clean) < 20:
        raise ValueError(f"Solo hay {len(df_clean)} filas válidas. Necesitas al menos 20.")

    # Preparar Y
    y = _binarizar_serie(df_clean[target_col])

    if y.nunique() < 2:
        raise ValueError("La variable objetivo solo tiene una clase. Necesita tener 0 y 1.")

    # Preparar X
    X = pd.DataFrame()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            X[col] = df_clean[col].values
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(df_clean[col].astype(str))

    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    # Entrenar
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_prob = gnb.predict_proba(X_test)[:, 1]

    # Métricas
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensibilidad  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1            = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        sensibilidad = especificidad = precision = f1 = 0.0

    # Importancia de features (log-verosimilitud aproximada)
    feature_importance = {}
    for i, col in enumerate(feature_cols):
        try:
            importancia = abs(gnb.theta_[1, i] - gnb.theta_[0, i])
            feature_importance[col] = round(float(importancia), 4)
        except Exception:
            feature_importance[col] = 0.0

    return {
        "cm":                cm,
        "accuracy":          round(acc, 4),
        "sensibilidad":      round(sensibilidad, 4),
        "especificidad":     round(especificidad, 4),
        "precision":         round(precision, 4),
        "f1":                round(f1, 4),
        "y_test":            y_test.values,
        "y_pred":            y_pred,
        "y_prob":            y_prob,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "feature_importance": feature_importance,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
    }


# ══════════════════════════════════════════════════════════════
# 5. UTILIDADES DE DATOS
# ══════════════════════════════════════════════════════════════

def preparar_serie_temporal(df: pd.DataFrame, fecha_col: str,
                             valor_col: str, freq: str = "D") -> pd.DataFrame:
    """Agrupa y prepara datos para gráfica temporal."""
    df_ts = df[[fecha_col, valor_col]].dropna().copy()
    df_ts[fecha_col] = pd.to_datetime(df_ts[fecha_col], infer_datetime_format=True)
    df_ts = df_ts.sort_values(fecha_col)
    df_ts = df_ts.set_index(fecha_col)
    return df_ts


def calcular_correlaciones(df: pd.DataFrame, cols_num: list,
                            target_col: str = None) -> pd.DataFrame:
    """
    Calcula matriz de correlación. Si se pasa target_col,
    también incluye correlación de cada variable con el target.
    """
    cols = [c for c in cols_num if c in df.columns]
    if target_col and target_col in df.columns:
        df_t = df.copy()
        df_t[target_col] = _binarizar_serie(df_t[target_col])
        if target_col not in cols:
            cols = cols + [target_col]
    return df[cols].corr()
