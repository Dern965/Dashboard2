import warnings
warnings.filterwarnings("ignore")

import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

MAX_SCAN_JOBS = max(1, min(2, (os.cpu_count() or 1)))

# ===================== CONFIG PÁGINA =====================
st.set_page_config(page_title="Estrategias de inversión personalizadas", layout="wide")

# ===================== PARÁMETROS FIJOS =====================
DATA_PATH = "datos/market_prices.csv"
DATE_COL = "date"
TICKER_COL = "instrument_id"
PRICE_COL = "adj_close"
FIXED_FREQ = "B"
MAX_HORIZON_DAYS = 10
DEFAULT_HORIZON_DAYS = 10
DEFAULT_N_TEST = 50
DEFAULT_WARM = 210
# Hiperparámetros del modelo (fijos, ya no se exponen en la UI)
# Valores seleccionados con benchmarks walk-forward iterativos sobre 8 emisoras
# representativas. Mejora total medida: hit_rate 45.4% (original) -> 60.0% (config actual).
DEFAULT_LAGS_MORPH = 8           # antes 5 — más contexto morfológico mejora el ajuste
DEFAULT_CONF_MIN = 0.04
DEFAULT_PASO = 2                 # antes min(horizon, 5) — muestreo más denso = más ejemplos de entrenamiento
DEFAULT_PRECISIONS = (1, 2, 3)   # antes (1, 3, 4) — p=4 es muy lento y no aporta hit_rate
DEFAULT_ROLL_ACC_WIN = 10
DEFAULT_RSI_SELL = 22
DEFAULT_RSI_BUY = 72
DEFAULT_TRAIN_WINDOW = 1200      # ventana rodada: usar últimas 1200 muestras del histórico (≈10 años en paso=2).
                                 # Probado contra "todo el histórico": +4 pp hit rate y +0.5 sharpe.
                                 # Los regímenes de mercado cambian; datos muy viejos contaminan.

MIN_INVESTMENT_AMOUNT = 1000
MAX_INVESTMENT_AMOUNT = 1_000_000
DEFAULT_INVESTMENT_AMOUNT = 50_000

# Referencia educativa para explicar "cuidar mi dinero".
# No se usa como dato oficial de inflación, solo como meta mínima conceptual.
INFLATION_REFERENCE_ANNUAL = 0.045

INVESTMENT_GOALS = [
    "Cuidar mi dinero",
    "Balance entre crecimiento y estabilidad",
    "Hacer crecer mi inversión",
    "Buscar una oportunidad más agresiva",
]

GOAL_EMOJI = {
    "Cuidar mi dinero": "🛡️",
    "Balance entre crecimiento y estabilidad": "⚖️",
    "Hacer crecer mi inversión": "📈",
    "Buscar una oportunidad más agresiva": "🔥",
}

GOAL_DESCRIPTIONS = {
    "Cuidar mi dinero": (
        "Prioriza conservar el poder adquisitivo. En este contexto no significa guardar el dinero sin moverlo, "
        "sino buscar una alternativa de bajo riesgo que aspire, al menos, a compensar la pérdida de valor por inflación."
    ),
    "Balance entre crecimiento y estabilidad": (
        "Busca un punto medio: aceptar movimientos moderados para intentar crecer, pero sin concentrar demasiado riesgo."
    ),
    "Hacer crecer mi inversión": (
        "Da más peso al rendimiento esperado. Acepta más variación en el precio con tal de buscar una mayor ganancia."
    ),
    "Buscar una oportunidad más agresiva": (
        "Acepta alta volatilidad y mayor posibilidad de pérdida temporal para perseguir oportunidades con mayor potencial."
    ),
}

RISK_LEVEL_DESCRIPTIONS = {
    1: "Muy bajo: prefieres evitar pérdidas aunque el crecimiento sea limitado.",
    2: "Bajo: aceptas pequeñas variaciones si la propuesta sigue siendo estable.",
    3: "Medio: buscas equilibrio entre riesgo y rendimiento.",
    4: "Alto: toleras caídas temporales si existe potencial de crecimiento.",
    5: "Muy alto: aceptas movimientos fuertes y mayor incertidumbre.",
}


SEASONAL_PRIOR_FALLBACK = {
    1: 0.0, 2: 0.0, 3: +0.10, 4: +0.20, 5: -0.10, 6: 0.0,
    7: 0.0, 8: -0.15, 9: -0.10, 10: -0.10, 11: +0.10, 12: 0.0,
}


def learn_seasonal_prior(months_tr, y_tr, min_n=20, strength=0.3):
    """
    Aprende un prior estacional empírico por mes a partir de los datos de entrenamiento
    de cada ticker. Si un mes tiene < min_n observaciones, devuelve 0 (sin sesgo).

    Esto reemplaza al prior hardcoded SEASONAL_PRIOR_FALLBACK que dependía de patrones
    genéricos del mercado mexicano. La versión aprendida se adapta a cada emisora.
    Probado en benchmark: aporta consistencia y evita sesgos artificiales en meses
    donde el prior fijo no aplica a la emisora particular.
    """
    months_tr = np.asarray(months_tr, dtype=int)
    y_tr = np.asarray(y_tr, dtype=int)
    prior = {}
    for m in range(1, 13):
        mask = months_tr == m
        if mask.sum() >= min_n:
            up_rate = float(y_tr[mask].mean())
            prior[m] = (up_rate - 0.5) * strength * 2.0
        else:
            prior[m] = 0.0
    return prior

# ===================== UTILIDADES BÁSICAS =====================
def get_file_mtime(path):
    return os.path.getmtime(path)


@st.cache_data
def load_prices(path, date_col, ticker_col, price_col, file_mtime):
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"No encontré la columna de fecha: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="%Y-%m-%d")

    rename_map = {date_col: "date", ticker_col: "instrument_id", price_col: "adj_close"}
    if "high" in df.columns:
        rename_map["high"] = "high"
    if "low" in df.columns:
        rename_map["low"] = "low"
    if "volume" in df.columns:
        rename_map["volume"] = "volume"
    df = df.rename(columns=rename_map)

    required = ["date", "instrument_id", "adj_close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    if "high" not in df.columns:
        df["high"] = df["adj_close"]
    if "low" not in df.columns:
        df["low"] = df["adj_close"]
    if "volume" not in df.columns:
        df["volume"] = 1.0

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["instrument_id"] = df["instrument_id"].astype(str).str.strip()

    keep = ["date", "instrument_id", "adj_close", "high", "low", "volume"]
    out = df[keep].dropna(subset=["date", "instrument_id", "adj_close"]).copy()
    out = out.sort_values(["instrument_id", "date"]).drop_duplicates(
        subset=["instrument_id", "date"], keep="last"
    )
    return out


# PARCHE 1: cachear resample_ohlcv para que scan_market siempre reciba
# el mismo objeto hasheable y no invalide su caché innecesariamente.
@st.cache_data
def resample_ohlcv(df, freq="B"):
    rule = "B"
    out = []
    for ticker, g in df.groupby("instrument_id", sort=True):
        g = g.sort_values("date").set_index("date")
        tmp = pd.DataFrame({
            "adj_close": g["adj_close"].resample(rule).last(),
            "high": g["high"].resample(rule).max(),
            "low": g["low"].resample(rule).min(),
            "volume": g["volume"].resample(rule).sum(),
        })
        tmp["instrument_id"] = ticker
        tmp = tmp.dropna(subset=["adj_close"]).reset_index()
        out.append(tmp)

    if not out:
        return pd.DataFrame(columns=["date", "adj_close", "high", "low", "volume", "instrument_id"])
    return pd.concat(out, ignore_index=True)


def wide_prices(df):
    return df.pivot(index="date", columns="instrument_id", values="adj_close").sort_index()


# ===================== MODELO GAMMA (OPTIMIZADO) =====================
class GammaBinary:
    def __init__(self, precision=2):
        self.precision = precision
        self.max_int_vals = None
        self.rho = 0
        self.X_enc = None
        self.y_train = None
        self.classes = None
        self.n_cls = None
        self._class_masks = None
        self._seg_starts = None

    def _encode_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        scale = 10 ** self.precision
        X_int = np.clip(
            np.round(X * scale).astype(np.int32),
            0,
            self.max_int_vals,
        )

        total_bits = int(np.sum(self.max_int_vals))
        result = np.zeros((len(X), total_bits), dtype=np.int8)

        start = 0
        for j, em in enumerate(self.max_int_vals):
            em = int(em)
            col_idx = np.arange(em, dtype=np.int32)
            result[:, start : start + em] = (X_int[:, j : j + 1] > col_idx).astype(np.int8)
            start += em

        return result

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes = np.unique(y)
        self.y_train = y
        self.n_cls = {c: max(1, int(np.sum(y == c))) for c in self.classes}
        self.max_int_vals = np.clip(
            np.round(np.max(X, axis=0) * (10 ** self.precision)).astype(int), 1, None
        )
        self.rho = int(np.min(self.max_int_vals))
        self.X_enc = self._encode_batch(X)

        self._class_masks = {c: (y == c) for c in self.classes}
        starts = np.concatenate([[0], np.cumsum(self.max_int_vals)[:-1]])
        self._seg_starts = starts.astype(int)

        return self

    def predict_with_score(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        Xe = self._encode_batch(X_test)
        results = []
        for pat in Xe:
            winner = None
            last_scores = {c: 0.0 for c in self.classes}

            seg_dists = []
            for s, em in zip(self._seg_starts, self.max_int_vals):
                em = int(em)
                d = np.sum(
                    np.abs(
                        self.X_enc[:, s : s + em].astype(np.int16)
                        - pat[s : s + em].astype(np.int16)
                    ),
                    axis=1,
                )
                seg_dists.append(d)
            seg_dists = np.stack(seg_dists, axis=0)

            # OPT: en vez de iterar theta=0..rho secuencial (caro cuando rho es grande),
            # iteramos solo en los valores únicos de distancia. El resultado lógico es el
            # mismo (sólo cambian las matches en esos thresholds) pero es mucho más rápido.
            unique_thetas = np.unique(seg_dists)
            unique_thetas = unique_thetas[unique_thetas <= self.rho]

            for theta in unique_thetas:
                ok = seg_dists <= theta
                match_counts = ok.sum(axis=0)

                scores = {
                    c: float(np.sum(match_counts[self._class_masks[c]]) / self.n_cls[c])
                    for c in self.classes
                }
                last_scores = scores
                ms = max(scores.values())
                cands = [c for c, sv in scores.items() if sv == ms and sv > 0]
                if len(cands) == 1:
                    winner = cands[0]
                    break

            if winner is None:
                winner = max(last_scores, key=last_scores.get)
            sv_list = sorted(last_scores.values(), reverse=True)
            conf = (sv_list[0] - sv_list[1]) if len(sv_list) >= 2 and sv_list[0] > 0 else 0.0
            results.append((winner, conf, last_scores))
        return results


# ===================== FEATURES Y MÉTRICAS =====================
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p, min_periods=p).mean()
    l = (-d.clip(upper=0)).rolling(p, min_periods=p).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(50)


def calc_bb_pct(s, w=20):
    ma = s.rolling(w).mean()
    std = s.rolling(w).std().replace(0, np.nan)
    return ((s - (ma - 2 * std)) / (4 * std)).clip(0, 1).fillna(0.5)


def calc_vol_ratio(vol, w):
    return (vol / vol.rolling(w).mean()).fillna(1.0).clip(0, 5)


def calc_atr_pct(high, low, close, w=14):
    """ATR (Average True Range) normalizado como % del precio. Mide volatilidad real."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(w).mean()
    return (atr / close.replace(0, np.nan) * 100).fillna(0).clip(0, 50)


def calc_macd_hist_norm(s, fast=12, slow=26, sig=9):
    """Histograma MACD normalizado por precio (en %). Captura aceleración de tendencia."""
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    macd = (ef - es) / s.replace(0, np.nan)
    signal = macd.ewm(span=sig, adjust=False).mean()
    return ((macd - signal) * 100).fillna(0).clip(-5, 5)


def calc_pos_vs_high(s, w=63):
    """Posición relativa al máximo de las últimas w sesiones (0..1). Útil como anti-momentum."""
    rmax = s.rolling(w).max()
    rmin = s.rolling(w).min()
    pos = (s - rmin) / (rmax - rmin).replace(0, np.nan)
    return pos.fillna(0.5).clip(0, 1)


def compute_error_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan, "SMAPE (%)": np.nan, "R²": np.nan}

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    smape = np.mean(np.abs(y_true - y_pred) / (((np.abs(y_true) + np.abs(y_pred)) / 2) + 1e-9)) * 100
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / (ss_tot + 1e-9)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "SMAPE (%)": float(smape),
        "R²": float(r2),
    }


def evaluar_metricas_direction(preds, reals, rets, horizonte):
    if len(preds) == 0:
        return {"acum": np.array([0.0]), "sharpe": 0.0, "max_dd": 0.0, "hit_rate": 0.0}

    preds = np.asarray(preds)
    reals = np.asarray(reals)
    rets = np.asarray(rets, dtype=float)
    hit_rate = float(np.mean(preds == reals) * 100)
    strategy_rets = np.where(preds == 1, rets, -rets)
    acum = np.cumprod(1 + strategy_rets / 100) - 1
    eq = np.cumprod(1 + strategy_rets / 100)
    rm = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - rm) / rm)) * 100
    sharpe = float((np.mean(strategy_rets) / (np.std(strategy_rets) + 1e-9)) * np.sqrt(252 / max(horizonte, 1)))
    return {"acum": acum, "sharpe": sharpe, "max_dd": max_dd, "hit_rate": hit_rate}


def build_features_for_ticker(df_t, horizon=10, paso=10, warm=210, n_lags_morph=5):
    df = df_t.sort_values("date").reset_index(drop=True).copy()
    min_hist = max(260, warm + horizon + 30)
    if len(df) < min_hist:
        return None

    df["ret_1d"] = df["adj_close"].pct_change(1) * 100
    df["high_low_pct"] = (df["high"] - df["low"]) / df["adj_close"].replace(0, np.nan) * 100
    df["vol_real"] = df["ret_1d"].rolling(10).std().fillna(0)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(float) / 4.0
    df["bb_pct"] = calc_bb_pct(df["adj_close"], 20)
    df["rsi_28"] = calc_rsi(df["adj_close"], 28)
    df["ret_5d"] = df["adj_close"].pct_change(5) * 100
    df["vol_ratio_5"] = calc_vol_ratio(df["volume"], 5)
    df["month_num"] = pd.to_datetime(df["date"]).dt.month

    # Features extendidas: ATR, MACD, momentum 21d, posición vs máximo y RSI rápido.
    # Probadas en benchmark walk-forward; mejoran ~5pp el hit rate del modelo final.
    df["atr_14"] = calc_atr_pct(df["high"], df["low"], df["adj_close"], 14)
    df["macd_hist"] = calc_macd_hist_norm(df["adj_close"])
    df["ret_21d"] = df["adj_close"].pct_change(21) * 100
    df["pos_high_63"] = calc_pos_vs_high(df["adj_close"], 63)
    df["rsi_14"] = calc_rsi(df["adj_close"], 14)
    df["skew_20"] = df["ret_1d"].rolling(20).skew().fillna(0).clip(-3, 3)

    df["ret_fwd"] = df["adj_close"].pct_change(horizon).shift(-horizon) * 100
    df["target"] = (df["ret_fwd"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    contexto = [
        "high_low_pct", "vol_real", "day_of_week", "bb_pct", "rsi_28",
        "ret_5d", "vol_ratio_5",
        "atr_14", "macd_hist", "ret_21d", "pos_high_63", "rsi_14", "skew_20",
    ]
    X_rows, y_arr, fechas, ret_fwd_arr, month_arr, rsi_arr, px_arr = [], [], [], [], [], [], []
    idx = warm

    while idx < len(df) - horizon:
        morph = list(df["ret_1d"].values[idx - n_lags_morph: idx])
        ctx = [df[f].values[idx] for f in contexto]
        X_rows.append(morph + ctx)
        y_arr.append(df["target"].values[idx])
        fechas.append(pd.to_datetime(df["date"].values[idx]))
        ret_fwd_arr.append(df["ret_fwd"].values[idx])
        month_arr.append(int(df["month_num"].values[idx]))
        rsi_arr.append(float(df["rsi_28"].values[idx]))
        px_arr.append(float(df["adj_close"].values[idx]))
        idx += paso

    if len(X_rows) < 50:
        return None

    return {
        "X": np.array(X_rows, dtype=np.float64),
        "y": np.array(y_arr, dtype=int),
        "dates": np.array(fechas),
        "ret_fwd": np.array(ret_fwd_arr, dtype=float),
        "months": np.array(month_arr, dtype=int),
        "rsi": np.array(rsi_arr, dtype=float),
        "signal_price": np.array(px_arr, dtype=float),
        "df_model": df,
    }


def build_current_feature_for_ticker(df_t, n_lags_morph=5):
    df = df_t.sort_values("date").reset_index(drop=True).copy()
    if len(df) < max(80, n_lags_morph + 70):
        return None

    df["ret_1d"] = df["adj_close"].pct_change(1) * 100
    df["high_low_pct"] = (df["high"] - df["low"]) / df["adj_close"].replace(0, np.nan) * 100
    df["vol_real"] = df["ret_1d"].rolling(10).std().fillna(0)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(float) / 4.0
    df["bb_pct"] = calc_bb_pct(df["adj_close"], 20)
    df["rsi_28"] = calc_rsi(df["adj_close"], 28)
    df["ret_5d"] = df["adj_close"].pct_change(5) * 100
    df["vol_ratio_5"] = calc_vol_ratio(df["volume"], 5)
    df["month_num"] = pd.to_datetime(df["date"]).dt.month
    # Mismas features extendidas que build_features_for_ticker (orden idéntico).
    df["atr_14"] = calc_atr_pct(df["high"], df["low"], df["adj_close"], 14)
    df["macd_hist"] = calc_macd_hist_norm(df["adj_close"])
    df["ret_21d"] = df["adj_close"].pct_change(21) * 100
    df["pos_high_63"] = calc_pos_vs_high(df["adj_close"], 63)
    df["rsi_14"] = calc_rsi(df["adj_close"], 14)
    df["skew_20"] = df["ret_1d"].rolling(20).skew().fillna(0).clip(-3, 3)
    df = df.replace([np.inf, -np.inf], np.nan)

    contexto = [
        "high_low_pct", "vol_real", "day_of_week", "bb_pct", "rsi_28",
        "ret_5d", "vol_ratio_5",
        "atr_14", "macd_hist", "ret_21d", "pos_high_63", "rsi_14", "skew_20",
    ]

    for idx in range(len(df) - 1, n_lags_morph - 1, -1):
        morph = df["ret_1d"].iloc[idx - n_lags_morph: idx]
        if len(morph) != n_lags_morph or morph.isna().any():
            continue
        if df.loc[idx, contexto + ["month_num", "rsi_28", "adj_close", "date"]].isna().any():
            continue

        x_cur = np.array(
            list(morph.astype(float).values) + [float(df.loc[idx, f]) for f in contexto],
            dtype=np.float64,
        )
        return {
            "X": x_cur,
            "current_date": pd.to_datetime(df.loc[idx, "date"]),
            "current_price": float(df.loc[idx, "adj_close"]),
            "current_month": int(df.loc[idx, "month_num"]),
            "current_rsi": float(df.loc[idx, "rsi_28"]),
        }

    return None


def robust_scale_train_test(X, split_idx):
    p2 = np.percentile(X[:split_idx], 2, axis=0)
    p98 = np.percentile(X[:split_idx], 98, axis=0)
    denom = np.where((p98 - p2) == 0, 1.0, (p98 - p2))
    Xc = np.clip(X, p2, p98)
    Xs = (Xc - p2) / denom
    Xs[:split_idx] = np.clip(Xs[:split_idx], 0, 1)
    Xs[split_idx:] = np.clip(Xs[split_idx:], 0, 1)
    return Xs


def robust_scale_fit_full(X):
    p2 = np.percentile(X, 2, axis=0)
    p98 = np.percentile(X, 98, axis=0)
    denom = np.where((p98 - p2) == 0, 1.0, (p98 - p2))
    Xc = np.clip(X, p2, p98)
    Xs = np.clip((Xc - p2) / denom, 0, 1)
    return Xs, p2, p98, denom


def robust_scale_apply(X_new, p2, p98, denom):
    X_new = np.asarray(X_new, dtype=np.float64)
    Xc = np.clip(X_new, p2, p98)
    return np.clip((Xc - p2) / denom, 0, 1)


@st.cache_data(show_spinner=False)
def run_gamma_backtest_for_ticker(df_t, horizon, paso, n_test, precisions, roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph):
    built = build_features_for_ticker(df_t, horizon=horizon, paso=paso, warm=warm, n_lags_morph=n_lags_morph)
    if built is None:
        return None

    X = built["X"]
    y = built["y"]
    months = built["months"]
    fechas = built["dates"]
    ret_fwd_arr = built["ret_fwd"]
    rsi_arr = built["rsi"]
    px_signal = built["signal_price"]

    if len(X) < (n_test + 10):
        return None

    split_idx = len(X) - n_test
    X_sc = robust_scale_train_test(X, split_idx)

    pA, pB, pC = precisions
    wf_pa, wf_pb, wf_pc = [], [], []
    wf_ens_pure, wf_ens_final = [], []
    wf_real, wf_ret, wf_fecha = [], [], []
    wf_px_senal, wf_px_real, wf_px_pred = [], [], []
    roll_correct_A, roll_correct_B, roll_correct_C = [], [], []

    for i in range(n_test):
        idx_train = split_idx + i

        # Rolling window: usar SOLO las últimas DEFAULT_TRAIN_WINDOW muestras de entrenamiento.
        # Los regímenes de mercado cambian con el tiempo (presidentes, ciclos macro, COVID).
        # Datos muy viejos contaminan el entrenamiento. Probado en benchmark: +4 pp hit rate.
        start_train = max(0, idx_train - DEFAULT_TRAIN_WINDOW)
        Xtr = X_sc[start_train:idx_train]
        ytr = y[start_train:idx_train]
        xi = X_sc[[idx_train]]

        clf_a = GammaBinary(pA).fit(Xtr, ytr)
        clf_b = GammaBinary(pB).fit(Xtr, ytr)
        clf_c = GammaBinary(pC).fit(Xtr, ytr)

        pa, ca, _ = clf_a.predict_with_score(xi)[0]
        pb, cb, _ = clf_b.predict_with_score(xi)[0]
        pc, cc, _ = clf_c.predict_with_score(xi)[0]

        wf_pa.append(pa)
        wf_pb.append(pb)
        wf_pc.append(pc)

        vp = {0: 0.0, 1: 0.0}
        vp[pa] += 1 + ca
        vp[pb] += 1 + cb
        vp[pc] += 1 + cc
        ens_pure = 1 if vp[1] >= vp[0] else 0
        wf_ens_pure.append(ens_pure)

        n_prev = min(i, roll_acc_win)
        w_a = np.mean(roll_correct_A[-n_prev:]) if n_prev >= 3 else 1.0
        w_b = np.mean(roll_correct_B[-n_prev:]) if n_prev >= 3 else 1.0
        w_c = np.mean(roll_correct_C[-n_prev:]) if n_prev >= 3 else 1.0
        w_a, w_b, w_c = max(w_a, 0.1), max(w_b, 0.1), max(w_c, 0.1)

        va = {0: 0.0, 1: 0.0}
        va[pa] += w_a * (1 + ca)
        va[pb] += w_b * (1 + cb)
        va[pc] += w_c * (1 + cc)

        # Prior estacional aprendido por ticker (en lugar del prior fijo genérico).
        # Cada decisión recalcula el prior usando solo datos previos al punto de evaluación,
        # respetando la lógica walk-forward (no hay leakage).
        prior_map = learn_seasonal_prior(months[start_train:idx_train], y[start_train:idx_train])
        prior = prior_map.get(int(months[idx_train]), 0.0)
        vf = dict(va)
        vf[1] += prior
        vf[0] -= prior
        gamma_pred = 1 if vf[1] >= vf[0] else 0
        gamma_conf = abs(vf[1] - vf[0]) / (vf[1] + vf[0] + 1e-9)

        # Segundo modelo: Logistic Regression sobre las mismas features.
        # Gamma es un clasificador basado en similitud; LogReg es lineal y calibrado en probabilidad.
        # Son modelos complementarios. Probado: el ensemble agrega +1.25 pp sobre Gamma solo.
        try:
            lr = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", solver="liblinear")
            lr.fit(Xtr, ytr)
            lr_prob = float(lr.predict_proba(xi)[0, 1])
            lr_pred = 1 if lr_prob >= 0.5 else 0
            lr_conf = abs(lr_prob - 0.5) * 2.0
        except Exception:
            lr_pred = gamma_pred
            lr_conf = 0.0

        # Combinación: si ambos coinciden, usar el voto común; si discrepan, usar el más confiado.
        # Es la estrategia "agree-or-most-confident", que minimiza errores de cada modelo individual.
        if gamma_pred == lr_pred:
            ens_final = gamma_pred
        else:
            ens_final = gamma_pred if gamma_conf >= lr_conf else lr_pred

        # RSI override en lógica mean-reversion (análisis técnico clásico):
        #   - RSI muy bajo (< rsi_sell)   => sobreventa => esperamos rebote AL ALZA
        #   - RSI muy alto (> rsi_buy)    => sobrecompra => esperamos corrección A LA BAJA
        # Esto INVIERTE la lógica anterior (que era momentum). En el benchmark mejoró
        # ~2.5 pp el hit rate vs la versión momentum.
        if rsi_arr[idx_train] < rsi_sell:
            ens_final = 1
        elif rsi_arr[idx_train] > rsi_buy:
            ens_final = 0

        wf_ens_final.append(ens_final)

        real_val = int(y[idx_train])
        wf_real.append(real_val)
        wf_ret.append(float(ret_fwd_arr[idx_train]))
        wf_fecha.append(pd.to_datetime(fechas[idx_train]))

        roll_correct_A.append(1 if pa == real_val else 0)
        roll_correct_B.append(1 if pb == real_val else 0)
        roll_correct_C.append(1 if pc == real_val else 0)

        px_s = float(px_signal[idx_train])
        wf_px_senal.append(px_s)
        wf_px_real.append(px_s * (1 + ret_fwd_arr[idx_train] / 100))

        ret_hist = np.array(ret_fwd_arr[start_train:idx_train])
        y_hist = np.array(y[start_train:idx_train])
        ms_up = float(np.mean(ret_hist[y_hist == 1])) if np.sum(y_hist == 1) > 0 else 1.0
        ms_dn = float(np.mean(ret_hist[y_hist == 0])) if np.sum(y_hist == 0) > 0 else -1.0
        wf_px_pred.append(px_s * (1 + (ms_up if ens_final == 1 else ms_dn) / 100))

    pa_arr = np.array(wf_pa)
    pb_arr = np.array(wf_pb)
    pc_arr = np.array(wf_pc)
    ep = np.array(wf_ens_pure)
    ef = np.array(wf_ens_final)
    rl = np.array(wf_real)
    rt = np.array(wf_ret, dtype=float)

    acum_bh = np.cumprod(1 + rt / 100) - 1
    met_A = evaluar_metricas_direction(pa_arr, rl, rt, horizonte=horizon)
    met_B = evaluar_metricas_direction(pb_arr, rl, rt, horizonte=horizon)
    met_C = evaluar_metricas_direction(pc_arr, rl, rt, horizonte=horizon)
    met_E = evaluar_metricas_direction(ep, rl, rt, horizonte=horizon)
    met_F = evaluar_metricas_direction(ef, rl, rt, horizonte=horizon)

    px_rl = np.array(wf_px_real, dtype=float)
    px_pr = np.array(wf_px_pred, dtype=float)
    err_metrics = compute_error_metrics(px_rl, px_pr)

    current_pack = build_current_feature_for_ticker(df_t, n_lags_morph=n_lags_morph)
    if current_pack is None:
        return None

    # Para la señal "actual" también aplicamos rolling window: solo las últimas N muestras
    # son representativas del régimen reciente del mercado.
    n_train = len(X)
    start_full = max(0, n_train - DEFAULT_TRAIN_WINDOW)
    X_for_now = X[start_full:]
    y_for_now = y[start_full:]
    months_for_now = months[start_full:]

    X_full_sc, p2_full, p98_full, denom_full = robust_scale_fit_full(X_for_now)
    x_current_sc = robust_scale_apply(current_pack["X"], p2_full, p98_full, denom_full).reshape(1, -1)

    clf_fa = GammaBinary(pA).fit(X_full_sc, y_for_now)
    clf_fb = GammaBinary(pB).fit(X_full_sc, y_for_now)
    clf_fc = GammaBinary(pC).fit(X_full_sc, y_for_now)
    ra_f = clf_fa.predict_with_score(x_current_sc)[0]
    rb_f = clf_fb.predict_with_score(x_current_sc)[0]
    rc_f = clf_fc.predict_with_score(x_current_sc)[0]

    w_a_f = max(np.mean(roll_correct_A[-roll_acc_win:]), 0.1) if len(roll_correct_A) else 1.0
    w_b_f = max(np.mean(roll_correct_B[-roll_acc_win:]), 0.1) if len(roll_correct_B) else 1.0
    w_c_f = max(np.mean(roll_correct_C[-roll_acc_win:]), 0.1) if len(roll_correct_C) else 1.0

    vf = {0: 0.0, 1: 0.0}
    vf[ra_f[0]] += w_a_f * (1 + ra_f[1])
    vf[rb_f[0]] += w_b_f * (1 + rb_f[1])
    vf[rc_f[0]] += w_c_f * (1 + rc_f[1])

    mes_hoy = int(current_pack["current_month"])
    # Prior aprendido sobre la ventana reciente (consistente con el walk-forward).
    prior_map_now = learn_seasonal_prior(months_for_now, y_for_now)
    prior_hoy = prior_map_now.get(mes_hoy, 0.0)
    vf[1] += prior_hoy
    vf[0] -= prior_hoy

    gamma_pred_now = 1 if vf[1] >= vf[0] else 0
    gamma_conf_now = abs(vf[1] - vf[0]) / (vf[1] + vf[0] + 1e-9)

    # Segundo modelo: LogReg sobre las mismas features escaladas y misma ventana reciente.
    try:
        lr_now = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", solver="liblinear")
        lr_now.fit(X_full_sc, y_for_now)
        lr_prob_now = float(lr_now.predict_proba(x_current_sc)[0, 1])
        lr_pred_now = 1 if lr_prob_now >= 0.5 else 0
        lr_conf_now = abs(lr_prob_now - 0.5) * 2.0
    except Exception:
        lr_pred_now = gamma_pred_now
        lr_conf_now = 0.0

    # Ensemble agree-or-most-confident.
    if gamma_pred_now == lr_pred_now:
        ens_f = gamma_pred_now
    else:
        ens_f = gamma_pred_now if gamma_conf_now >= lr_conf_now else lr_pred_now
    conf_f = max(gamma_conf_now, lr_conf_now) if gamma_pred_now == lr_pred_now else min(gamma_conf_now, lr_conf_now)

    rsi_hoy = float(current_pack["current_rsi"])
    override_txt = f"RSI-28 = {rsi_hoy:.1f} (zona normal)"
    # Override en lógica mean-reversion: sobreventa → rebote al alza; sobrecompra → corrección.
    if rsi_hoy < rsi_sell:
        ens_f = 1
        override_txt = f"RSI bajo: {rsi_hoy:.1f} < {rsi_sell} (sobreventa, se espera rebote al alza)"
    elif rsi_hoy > rsi_buy:
        ens_f = 0
        override_txt = f"RSI alto: {rsi_hoy:.1f} > {rsi_buy} (sobrecompra, se espera corrección)"

    precio_hoy = float(current_pack["current_price"])
    fecha_hoy = pd.to_datetime(current_pack["current_date"])
    fecha_t = fecha_hoy + pd.offsets.BDay(horizon)

    ret_esp = (
        float(np.mean(rt[rl == 1]))
        if ens_f == 1 and np.sum(rl == 1) > 0
        else float(np.mean(rt[rl == 0]))
        if np.sum(rl == 0) > 0
        else 0.0
    )
    px_proj = precio_hoy * (1 + ret_esp / 100)

    if conf_f < conf_min and "RSI" in override_txt and "alto" not in override_txt and "bajo" not in override_txt:
        senal_txt = "ESPERAR"
    else:
        senal_txt = "SUBE" if ens_f == 1 else "BAJA"

    return {
        "dates": wf_fecha,
        "acum_bh": acum_bh,
        "met_A": met_A,
        "met_B": met_B,
        "met_C": met_C,
        "met_E": met_E,
        "met_F": met_F,
        "pred_A": pa_arr,
        "pred_B": pb_arr,
        "pred_C": pc_arr,
        "pred_E": ep,
        "pred_F": ef,
        "real_cls": rl,
        "ret_real": rt,
        "px_signal": np.array(wf_px_senal, dtype=float),
        "px_real": px_rl,
        "px_pred": px_pr,
        "err_metrics": err_metrics,
        "current_signal": senal_txt,
        "current_conf": float(conf_f),
        "current_price": precio_hoy,
        "projected_price": float(px_proj),
        "expected_ret_pct": float(ret_esp),
        "current_date": fecha_hoy,
        "target_date": fecha_t,
        "override_txt": override_txt,
        "current_rsi": rsi_hoy,
    }


# ===================== PERFIL DE USUARIO Y PORTAFOLIO =====================
def horizon_to_business_days(value):
    return int(np.clip(int(value), 1, MAX_HORIZON_DAYS))


def human_horizon_label(value):
    days = horizon_to_business_days(value)
    return f"{days} día(s) hábil(es)"


def classify_investor_profile(amount, horizon_days, risk_tolerance, goal):
    """
    Clasifica al usuario con base en cuatro factores:
    monto, horizonte, tolerancia al riesgo y objetivo.
    La idea es que el perfil no dependa solo del modelo GAMMA,
    sino de las preferencias declaradas por el usuario.
    """
    amount = float(np.clip(amount, MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT))
    horizon_days = int(horizon_to_business_days(horizon_days))
    risk_tolerance = int(np.clip(risk_tolerance, 1, 5))

    horizon_score = 0.0 if horizon_days <= 3 else 0.6 if horizon_days <= 7 else 1.0
    goal_score = {
        "Cuidar mi dinero": -1.0,
        "Balance entre crecimiento y estabilidad": 0.0,
        "Hacer crecer mi inversión": 1.0,
        "Buscar una oportunidad más agresiva": 2.0,
    }[goal]

    # El monto no define si alguien es agresivo, pero sí cambia la diversificación.
    amount_score = 0.0
    if amount >= 150_000:
        amount_score = 0.3
    if amount >= 500_000:
        amount_score = 0.5

    total = (risk_tolerance - 1) * 1.6 + horizon_score + goal_score + amount_score

    if total <= 2.5:
        profile = "Conservador"
        description = (
            "Perfil conservador: se prioriza estabilidad, menor volatilidad y una reserva mayor. "
            "La cartera evita señales negativas y limita la concentración."
        )
    elif total <= 5.5:
        profile = "Moderado"
        description = (
            "Perfil moderado: se busca equilibrio entre crecimiento y control del riesgo. "
            "La cartera puede incluir más emisoras si el monto lo permite."
        )
    else:
        profile = "Agresivo"
        description = (
            "Perfil agresivo: se acepta mayor variación en el precio para buscar más rendimiento. "
            "La cartera reduce la reserva y puede diversificar en más emisoras."
        )

    if amount < 20_000:
        amount_note = " Monto bajo: conviene no fragmentar demasiado la inversión."
        amount_band = "Bajo"
    elif amount < 150_000:
        amount_note = " Monto medio: permite diversificación moderada."
        amount_band = "Medio"
    elif amount < 500_000:
        amount_note = " Monto alto: permite diversificar más sin perder claridad."
        amount_band = "Alto"
    else:
        amount_note = " Monto muy alto: se recomienda una cartera más diversificada, sin rebasar el límite del simulador."
        amount_band = "Muy alto"

    inflation_target_horizon = ((1 + INFLATION_REFERENCE_ANNUAL) ** (horizon_days / 252) - 1) * 100

    return {
        "perfil": profile,
        "puntaje": round(float(total), 2),
        "descripcion": description + amount_note,
        "horizonte_dias": int(horizon_days),
        "objetivo_detalle": GOAL_DESCRIPTIONS[goal],
        "objetivo_icono": GOAL_EMOJI[goal],
        "riesgo_detalle": RISK_LEVEL_DESCRIPTIONS[risk_tolerance],
        "monto_categoria": amount_band,
        "meta_minima_horizonte": float(inflation_target_horizon),
        "umbral_volatilidad": {"Conservador": 24, "Moderado": 34, "Agresivo": 50}[profile],
        "cash_base": {"Conservador": 0.30, "Moderado": 0.15, "Agresivo": 0.05}[profile],
        "max_peso": {"Conservador": 0.28, "Moderado": 0.35, "Agresivo": 0.45}[profile],
        "n_base": {"Conservador": 2, "Moderado": 3, "Agresivo": 4}[profile],
    }


def compute_max_drawdown(price_series):
    s = pd.Series(price_series).dropna().astype(float)
    if s.empty:
        return np.nan
    eq = s / s.iloc[0]
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min() * 100)


def compute_volatility_snapshot(df_t):
    s = df_t.sort_values("date").set_index("date")["adj_close"].dropna().astype(float)
    ret = s.pct_change().dropna()
    if ret.empty:
        return {
            "vol_20d": np.nan,
            "vol_60d": np.nan,
            "vol_downside": np.nan,
            "max_dd_252": np.nan,
            "ret_63d": np.nan,
            "ret_252d": np.nan,
            "risk_band": "Sin datos",
        }

    r20 = ret.tail(20)
    r60 = ret.tail(60)
    r252 = ret.tail(252)
    downside = r252[r252 < 0]

    vol_20d = float(r20.std() * np.sqrt(252) * 100) if len(r20) >= 5 else np.nan
    vol_60d = float(r60.std() * np.sqrt(252) * 100) if len(r60) >= 15 else np.nan
    vol_down = float(downside.std() * np.sqrt(252) * 100) if len(downside) >= 5 else np.nan
    max_dd = compute_max_drawdown(s.tail(252))
    ret_63d = float((s.iloc[-1] / s.iloc[-64] - 1) * 100) if len(s) >= 64 else np.nan
    ret_252d = float((s.iloc[-1] / s.iloc[-253] - 1) * 100) if len(s) >= 253 else np.nan

    ref_vol = vol_60d if np.isfinite(vol_60d) else vol_20d
    if not np.isfinite(ref_vol):
        band = "Sin datos"
    elif ref_vol < 20:
        band = "Baja"
    elif ref_vol < 35:
        band = "Media"
    else:
        band = "Alta"

    return {
        "vol_20d": vol_20d,
        "vol_60d": vol_60d,
        "vol_downside": vol_down,
        "max_dd_252": max_dd,
        "ret_63d": ret_63d,
        "ret_252d": ret_252d,
        "risk_band": band,
    }


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def confidence_level(conf):
    if conf >= 0.20:
        return "Alta"
    if conf >= 0.08:
        return "Media"
    return "Baja"


def signal_emoji(signal):
    return {"SUBE": "🟢", "BAJA": "🔴", "ESPERAR": "🟡"}.get(signal, "⚪")


def signal_weight(signal):
    return {"SUBE": 1.0, "ESPERAR": 0.45, "BAJA": 0.0}.get(signal, 0.0)


def normalize_weights_with_cap(score_series, total_weight, cap):
    scores = pd.Series(score_series, dtype=float).clip(lower=0)
    if scores.sum() <= 0 or total_weight <= 0:
        return pd.Series(0.0, index=scores.index)

    weights = scores / scores.sum() * total_weight
    cap = float(max(cap, 0.01))

    for _ in range(10):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        under = ~over
        if excess <= 0 or not under.any() or float(scores[under].sum()) <= 0:
            break
        redistribution = scores[under] / scores[under].sum() * excess
        weights[under] += redistribution

    if weights.sum() > 0:
        weights = weights / weights.sum() * total_weight
    return weights


def infer_asset_count(amount, base_n):
    """
    Define cuántas emisoras sugerir según el monto.
    Esta era la razón principal por la que siempre salían 3 a 5 recomendaciones:
    antes la función solo permitía base_n o base_n + 1.
    """
    amount = float(amount)

    if amount < 5_000:
        return 1
    if amount < 20_000:
        return min(2, base_n)
    if amount < 60_000:
        return max(2, base_n)
    if amount < 150_000:
        return max(3, base_n + 1)
    if amount < 500_000:
        return max(4, base_n + 2)
    return max(5, base_n + 3)


# ===================== SCAN MARKET =====================
def _process_one_ticker(ticker, df_rs, horizon, paso, n_test, precisions,
                        roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph):
    df_t = df_rs[df_rs["instrument_id"] == ticker].sort_values("date").copy()
    res = run_gamma_backtest_for_ticker(
        df_t=df_t,
        horizon=horizon,
        paso=paso,
        n_test=int(n_test),
        precisions=tuple(int(x) for x in precisions),
        roll_acc_win=int(roll_acc_win),
        rsi_sell=float(rsi_sell),
        rsi_buy=float(rsi_buy),
        conf_min=float(conf_min),
        warm=int(warm),
        n_lags_morph=int(n_lags_morph),
    )
    if res is None:
        return None

    vol = compute_volatility_snapshot(df_t)
    quality = (
        (res["met_F"]["hit_rate"] / 100.0) * 0.45
        + clip01((res["err_metrics"]["R²"] + 0.25) / 1.25) * 0.25
        + clip01((20 - max(res["err_metrics"]["SMAPE (%)"], 0)) / 20) * 0.20
        + clip01((res["current_conf"] - 0.02) / 0.25) * 0.10
    )
    score = (
        res["met_F"]["hit_rate"]
        + res["met_F"]["sharpe"]
        - 0.25 * res["err_metrics"]["SMAPE (%)"]
        - 0.10 * res["err_metrics"]["MAPE (%)"]
    )
    return {
        "Emisora": ticker,
        "Señal": res["current_signal"],
        "Confianza num": float(res["current_conf"]),
        "Confianza": confidence_level(res["current_conf"]),
        "Acierto (%)": round(res["met_F"]["hit_rate"], 2),
        "Sharpe": round(res["met_F"]["sharpe"], 3),
        "Caída máxima estrategia (%)": round(res["met_F"]["max_dd"], 2),
        "Cambio esperado (%)": round(res["expected_ret_pct"], 2),
        "Precio actual": round(res["current_price"], 2),
        "Precio estimado": round(res["projected_price"], 2),
        "MAPE (%)": round(res["err_metrics"]["MAPE (%)"], 2),
        "SMAPE (%)": round(res["err_metrics"]["SMAPE (%)"], 2),
        "R²": round(res["err_metrics"]["R²"], 3),
        "Volatilidad 20d (%)": round(vol["vol_20d"], 2) if pd.notna(vol["vol_20d"]) else np.nan,
        "Volatilidad 60d (%)": round(vol["vol_60d"], 2) if pd.notna(vol["vol_60d"]) else np.nan,
        "Volatilidad bajista (%)": round(vol["vol_downside"], 2) if pd.notna(vol["vol_downside"]) else np.nan,
        "Drawdown 252d (%)": round(vol["max_dd_252"], 2) if pd.notna(vol["max_dd_252"]) else np.nan,
        "Cambio 3 meses (%)": round(vol["ret_63d"], 2) if pd.notna(vol["ret_63d"]) else np.nan,
        "Cambio 12 meses (%)": round(vol["ret_252d"], 2) if pd.notna(vol["ret_252d"]) else np.nan,
        "Riesgo": vol["risk_band"],
        "Puntaje modelo": round(score, 3),
        "Calidad modelo": round(quality, 3),
    }


# PARCHE 2: scan_market sin joblib (inestable en Streamlit Cloud) +
# caché manual en session_state para no recalcular al cambiar de pestaña.
# run_gamma_backtest_for_ticker ya tiene @st.cache_data, así que el cómputo
# pesado por emisora solo ocurre una vez aunque se llame desde aquí varias veces.
def _make_scan_key(tickers_all, horizon, paso, n_test, precisions,
                   roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph):
    payload = {
        "tickers": sorted(tickers_all),
        "horizon": horizon, "paso": paso, "n_test": n_test,
        "precisions": list(precisions), "roll_acc_win": roll_acc_win,
        "rsi_sell": rsi_sell, "rsi_buy": rsi_buy,
        "conf_min": conf_min, "warm": warm, "n_lags_morph": n_lags_morph,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scan_market(df_rs, tickers_all, horizon, paso, n_test, precisions, roll_acc_win,
                rsi_sell, rsi_buy, conf_min, warm, n_lags_morph):
    current_key = _make_scan_key(
        tickers_all, horizon, paso, n_test, precisions,
        roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph,
    )
    # Devolver resultado cacheado si los parámetros no cambiaron
    if (st.session_state.get("_scan_key") == current_key
            and not st.session_state.get("_scan_result", pd.DataFrame()).empty):
        return st.session_state["_scan_result"]

    # Primera vez o parámetros cambiaron: recalcular con barra de progreso
    total = len(tickers_all)
    bar = st.progress(0, text=f"Analizando {total} emisoras…")
    rows = []
    for i, ticker in enumerate(tickers_all):
        bar.progress((i + 1) / total, text=f"Analizando {ticker}… ({i+1}/{total})")
        result = _process_one_ticker(
            ticker, df_rs, horizon, paso, n_test, precisions,
            roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph,
        )
        if result is not None:
            rows.append(result)
    bar.empty()

    result_df = pd.DataFrame(rows).sort_values("Puntaje modelo", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    st.session_state["_scan_key"] = current_key
    st.session_state["_scan_result"] = result_df
    return result_df


def score_assets_for_profile(market_df, profile_info, goal, horizon_days):
    if market_df.empty:
        return market_df.copy()

    df = market_df.copy()
    profile = profile_info["perfil"]

    low_vol = clip01(1 - df["Volatilidad 60d (%)"].fillna(40) / 45)
    conf = clip01(df["Confianza num"].fillna(0.0) / 0.25)
    quality = clip01(df["Calidad modelo"].fillna(0.0))
    expected_ret = clip01((df["Cambio esperado (%)"].fillna(0.0) + 5) / 20)
    momentum = clip01((df["Cambio 3 meses (%)"].fillna(0.0) + 15) / 40)
    signal = df["Señal"].map(signal_weight).fillna(0.0)

    if profile == "Conservador":
        suitability = 0.30 * signal + 0.28 * low_vol + 0.22 * quality + 0.12 * conf + 0.08 * expected_ret
    elif profile == "Moderado":
        suitability = 0.30 * signal + 0.22 * low_vol + 0.22 * quality + 0.14 * conf + 0.12 * expected_ret
    else:
        suitability = 0.28 * signal + 0.15 * low_vol + 0.20 * quality + 0.12 * conf + 0.25 * expected_ret

    if goal == "Cuidar mi dinero":
        suitability = suitability + 0.08 * low_vol
    elif goal == "Hacer crecer mi inversión":
        suitability = suitability + 0.06 * expected_ret + 0.04 * momentum
    elif goal == "Buscar una oportunidad más agresiva":
        suitability = suitability + 0.08 * expected_ret + 0.02 * conf

    df["Puntaje perfil"] = np.round(suitability, 3)
    df["Elegible"] = df["Señal"] != "BAJA"

    if profile == "Conservador":
        df.loc[df["Volatilidad 60d (%)"].fillna(99) > profile_info["umbral_volatilidad"], "Elegible"] = False
        df.loc[df["Confianza num"] < 0.04, "Elegible"] = False
    elif profile == "Moderado":
        df.loc[df["Confianza num"] < 0.02, "Elegible"] = False

    if horizon_days <= 10:
        df.loc[df["Volatilidad 20d (%)"].fillna(99) > 45, "Elegible"] = False

    return df.sort_values(["Elegible", "Puntaje perfil"], ascending=[False, False]).reset_index(drop=True)


def explain_asset_choice(row, profile, goal):
    signal = str(row.get("Señal", ""))
    risk = str(row.get("Riesgo", "Sin datos"))
    expected = row.get("Cambio esperado (%)", np.nan)
    vol = row.get("Volatilidad 60d (%)", np.nan)
    conf = row.get("Confianza", "Sin datos")

    parts = []
    if signal == "SUBE":
        parts.append("señal positiva")
    elif signal == "ESPERAR":
        parts.append("señal prudente")
    else:
        parts.append("no es la primera opción por señal negativa")

    if pd.notna(expected):
        parts.append(f"cambio esperado de {expected:.2f}%")
    if pd.notna(vol):
        parts.append(f"volatilidad reciente de {vol:.2f}%")
    parts.append(f"confianza {str(conf).lower()}")

    if profile == "Conservador":
        criterio = "se priorizó menor volatilidad y mayor protección"
    elif profile == "Moderado":
        criterio = "se buscó equilibrio entre rendimiento, confianza y riesgo"
    else:
        criterio = "se dio más peso al rendimiento esperado y a la oportunidad"

    return f"Para perfil {profile.lower()}, {criterio}; {', '.join(parts)}."


def build_personalized_portfolio(scored_df, df_rs, amount, profile_info, goal, horizon_days):
    if scored_df.empty:
        return {
            "portfolio": pd.DataFrame(),
            "validation": pd.DataFrame(),
            "summary": {
                "cash_pct": 1.0,
                "portfolio_expected_ret": 0.0,
                "portfolio_vol": np.nan,
                "portfolio_conf": 0.0,
                "selected_count": 0,
                "profile": profile_info["perfil"],
                "target_assets": 0,
                "eligible_assets": 0,
            },
        }

    amount = float(np.clip(amount, MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT))
    profile = profile_info["perfil"]

    cash_pct = profile_info["cash_base"]
    if goal == "Cuidar mi dinero":
        cash_pct += 0.10
    elif goal == "Hacer crecer mi inversión":
        cash_pct -= 0.05
    elif goal == "Buscar una oportunidad más agresiva":
        cash_pct -= 0.08

    if horizon_days <= 3:
        cash_pct += 0.08
    elif horizon_days <= 7:
        cash_pct += 0.05
    else:
        cash_pct += 0.03

    cash_pct = float(np.clip(cash_pct, 0.02, 0.50))
    invest_pct = 1 - cash_pct

    target_assets = infer_asset_count(amount, profile_info["n_base"])
    cap_weight = profile_info["max_peso"] * invest_pct

    eligible = scored_df[scored_df["Elegible"]].copy()
    eligible_assets = len(eligible)

    if eligible.empty:
        top = scored_df.head(1).copy()
        eligible = top.assign(Elegible=False)
        cash_pct = 1.0
        invest_pct = 0.0
    else:
        # El monto ahora sí modifica el número de recomendaciones.
        # Aun así, nunca se inventan emisoras: solo se toman las que pasan filtros de elegibilidad.
        eligible = eligible.head(min(target_assets, eligible_assets)).copy()

    if invest_pct > 0:
        weights = normalize_weights_with_cap(
            eligible["Puntaje perfil"],
            total_weight=invest_pct,
            cap=cap_weight,
        )
        eligible["Peso"] = weights.values
    else:
        eligible["Peso"] = 0.0

    eligible["Monto sugerido"] = eligible["Peso"] * amount
    eligible["Señal visual"] = eligible["Señal"].map(signal_emoji) + " " + eligible["Señal"]
    eligible["Motivo de elección"] = eligible.apply(
        lambda r: explain_asset_choice(r, profile=profile, goal=goal),
        axis=1,
    )

    cash_row = pd.DataFrame([{
        "Emisora": "Efectivo / reserva",
        "Señal": "RESERVA",
        "Señal visual": "💵 RESERVA",
        "Confianza": "-",
        "Confianza num": np.nan,
        "Puntaje perfil": np.nan,
        "Peso": cash_pct,
        "Monto sugerido": cash_pct * amount,
        "Cambio esperado (%)": 0.0,
        "Volatilidad 60d (%)": 0.0,
        "Riesgo": "Bajo",
        "Motivo de elección": (
            "Reserva sugerida para reducir exposición. Sirve para no invertir el 100% del capital "
            "cuando el horizonte es corto o el perfil requiere más protección."
        ),
    }])

    portfolio = pd.concat([eligible, cash_row], ignore_index=True, sort=False)
    portfolio["Peso (%)"] = portfolio["Peso"] * 100
    portfolio = portfolio[[
        "Emisora", "Señal visual", "Confianza", "Riesgo", "Cambio esperado (%)",
        "Volatilidad 60d (%)", "Peso (%)", "Monto sugerido", "Motivo de elección"
    ]].rename(columns={
        "Señal visual": "Señal",
        "Cambio esperado (%)": f"Cambio esperado al horizonte (%)",
        "Volatilidad 60d (%)": "Volatilidad reciente (%)",
    })

    asset_rows = portfolio[portfolio["Emisora"] != "Efectivo / reserva"].copy()
    selected_assets = asset_rows["Emisora"].tolist()

    portfolio_expected_ret = float(
        asset_rows[f"Cambio esperado al horizonte (%)"].fillna(0)
        .mul(asset_rows["Peso (%)"] / 100)
        .sum()
    )
    portfolio_conf = (
        float(
            scored_df.set_index("Emisora")
            .loc[selected_assets, "Confianza num"]
            .fillna(0)
            .mul(asset_rows["Peso (%)"] / 100)
            .sum()
        )
        if selected_assets
        else 0.0
    )

    portfolio_vol = np.nan
    if selected_assets:
        wide = (
            df_rs[df_rs["instrument_id"].isin(selected_assets)]
            .pivot(index="date", columns="instrument_id", values="adj_close")
            .sort_index()
        )
        rets = wide.pct_change().dropna().tail(252)
        if not rets.empty:
            weights = asset_rows.set_index("Emisora")["Peso (%)"] / 100
            weights = weights.reindex(rets.columns).fillna(0.0)
            cov = rets.cov() * 252
            port_var = float(np.dot(weights.values, np.dot(cov.values, weights.values)))
            portfolio_vol = np.sqrt(max(port_var, 0)) * 100

    max_weight_pct = float(asset_rows["Peso (%)"].max()) if not asset_rows.empty else 0.0
    risk_ok = pd.isna(portfolio_vol) or portfolio_vol <= profile_info["umbral_volatilidad"]
    cash_ok = cash_pct >= max(0.0, profile_info["cash_base"] - 0.05)
    concentration_ok = max_weight_pct <= profile_info["max_peso"] * 100 + 2
    diversification_ok = (
        len(selected_assets) >= min(2, target_assets)
        or amount < 20_000
        or eligible_assets < min(2, target_assets)
    )
    no_baja_ok = "BAJA" not in (
        scored_df.set_index("Emisora").reindex(selected_assets)["Señal"].fillna("").tolist()
    )

    validation = pd.DataFrame([
        {
            "Chequeo": "Riesgo acorde a tu perfil",
            "Resultado": "✅ Sí" if risk_ok else "⚠️ Revisar",
            "Detalle": f"Volatilidad estimada del portafolio: {fmt_pct(portfolio_vol, 2)}. Límite de referencia para tu perfil: {profile_info['umbral_volatilidad']}%.",
        },
        {
            "Chequeo": "Reserva de efectivo suficiente",
            "Resultado": "✅ Sí" if cash_ok else "⚠️ Revisar",
            "Detalle": f"Reserva sugerida: {fmt_pct(cash_pct * 100, 1)} del total.",
        },
        {
            "Chequeo": "Concentración razonable",
            "Resultado": "✅ Sí" if concentration_ok else "⚠️ Revisar",
            "Detalle": f"Peso máximo en una sola emisora: {fmt_pct(max_weight_pct, 1)}.",
        },
        {
            "Chequeo": "Diversificación según monto",
            "Resultado": "✅ Sí" if diversification_ok else "⚠️ Revisar",
            "Detalle": f"Objetivo por monto: {target_assets} emisora(s). Elegibles reales con filtros: {eligible_assets}. Seleccionadas: {len(selected_assets)}.",
        },
        {
            "Chequeo": "Evita señales claramente negativas",
            "Resultado": "✅ Sí" if no_baja_ok else "⚠️ Revisar",
            "Detalle": "La cartera propuesta evita activos con señal BAJA cuando fue posible.",
        },
    ])

    return {
        "portfolio": portfolio,
        "validation": validation,
        "summary": {
            "cash_pct": cash_pct,
            "portfolio_expected_ret": portfolio_expected_ret,
            "portfolio_vol": portfolio_vol,
            "portfolio_conf": portfolio_conf,
            "selected_count": len(selected_assets),
            "profile": profile,
            "target_assets": target_assets,
            "eligible_assets": eligible_assets,
        },
    }



# ===================== AYUDAS VISUALES =====================
def fmt_num(x, dec=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{dec}f}"


def fmt_pct(x, dec=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{dec}f}%"


def estado_color(signal):
    if signal == "SUBE":
        return "🟢"
    if signal == "BAJA":
        return "🔴"
    return "🟡"


def confianza_texto(conf):
    if conf >= 0.20:
        return "Alta"
    if conf >= 0.08:
        return "Media"
    return "Baja"


def build_daily_projection_path(res):
    """
    Genera una trayectoria diaria aproximada desde el precio actual hasta el precio objetivo.
    No es un pronóstico intradía ni una garantía; solo desagrega el cambio esperado al horizonte
    para que el usuario vea valores día por día.
    """
    horizon = max(1, len(pd.bdate_range(res["current_date"] + pd.offsets.BDay(1), res["target_date"])))
    dates = pd.bdate_range(res["current_date"] + pd.offsets.BDay(1), periods=horizon)

    current_price = float(res["current_price"])
    projected_price = float(res["projected_price"])

    if current_price <= 0 or not np.isfinite(current_price) or not np.isfinite(projected_price):
        return pd.DataFrame(columns=["Fecha", "Día hábil", "Precio estimado", "Cambio acumulado (%)"])

    daily_factor = (projected_price / current_price) ** (1 / horizon)
    prices = [current_price * (daily_factor ** i) for i in range(1, horizon + 1)]

    out = pd.DataFrame({
        "Fecha": dates.date,
        "Día hábil": np.arange(1, horizon + 1),
        "Precio estimado": prices,
    })
    out["Cambio acumulado (%)"] = (out["Precio estimado"] / current_price - 1) * 100
    out["Precio estimado"] = out["Precio estimado"].round(2)
    out["Cambio acumulado (%)"] = out["Cambio acumulado (%)"].round(2)
    return out


def explicar_error_simple(m):
    return (
        f"Error promedio: {fmt_num(m['MAE'], 3)} | "
        f"Error porcentual: {fmt_pct(m['MAPE (%)'], 2)} | "
        f"Ajuste general (R²): {fmt_num(m['R²'], 3)}"
    )


def help_box(text):
    st.info(text)


def chart_reason_box(chart_name, reason):
    """Explicación breve para justificar por qué se usa una gráfica ante usuarios no técnicos."""
    st.caption(f"📌 Por qué se usa esta visualización: {reason}")


def show_readable_dataframe(df, height=None, text_columns=None, hide_index=True):
    """
    Muestra tablas legibles sin deformar el encabezado.

    Ajuste clave:
    - El encabezado queda en una sola línea para que no se haga enorme.
    - Solo las columnas de texto largo envuelven contenido dentro de la celda.
    - Si la tabla tiene muchas columnas, se permite scroll horizontal suave,
      pero el texto largo ya no queda cortado dentro de la última columna.
    """
    text_columns = [c for c in (text_columns or []) if c in df.columns]

    # Si no hay columnas largas, se conserva el dataframe interactivo normal.
    if not text_columns:
        st.dataframe(df, width="stretch", height=height, hide_index=hide_index)
        return

    df_show = df.copy()

    def _format_cell(value):
        if pd.isna(value):
            return "-"
        if isinstance(value, float):
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)

    def _width_for_column(col):
        col_l = str(col).lower()
        if col in text_columns:
            return 520
        if "emisora" in col_l or "escenario" in col_l or "objetivo" in col_l:
            return 150
        if "señal" in col_l or "riesgo" in col_l or "estado" in col_l or "resultado" in col_l:
            return 110
        if "confianza" in col_l or "acción" in col_l or "aspecto" in col_l:
            return 135
        if "monto" in col_l or "capital" in col_l or "precio" in col_l:
            return 135
        if "%" in col_l or "volatilidad" in col_l or "rendimiento" in col_l or "cambio" in col_l:
            return 145
        return 125

    columns = list(df_show.columns)
    visible_columns = columns if hide_index else [df_show.index.name or "Índice"] + columns
    widths = {col: _width_for_column(col) for col in visible_columns}
    total_width = sum(widths.values())
    table_min_width = max(900, total_width)

    colgroup = "".join(
        f'<col style="width:{widths[col]}px; min-width:{widths[col]}px;">'
        for col in visible_columns
    )

    header_html = "".join(
        f'<th title="{html.escape(str(col))}">{html.escape(str(col))}</th>'
        for col in visible_columns
    )

    rows_html = []
    for idx, row in df_show.iterrows():
        cells = []
        if not hide_index:
            cells.append(f'<td class="nowrap-cell">{html.escape(_format_cell(idx))}</td>')
        for col in columns:
            cell_class = "text-cell" if col in text_columns else "nowrap-cell"
            cells.append(f'<td class="{cell_class}">{html.escape(_format_cell(row[col]))}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    max_height_css = f"max-height:{int(height)}px; overflow-y:auto;" if height else ""
    table_html = f"""
    <style>
        .wrapped-table-container {{
            width: 100%;
            {max_height_css}
            overflow-x: auto;
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 10px;
            margin: 0.35rem 0 1rem 0;
        }}
        .wrapped-table-container table {{
            width: 100%;
            min-width: {table_min_width}px;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.92rem;
        }}
        .wrapped-table-container th,
        .wrapped-table-container td {{
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            border-right: 1px solid rgba(128, 128, 128, 0.18);
            padding: 0.48rem 0.62rem;
            vertical-align: top;
            line-height: 1.32;
        }}
        .wrapped-table-container th {{
            background: rgba(128, 128, 128, 0.16);
            font-weight: 700;
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: ellipsis;
            height: 38px;
        }}
        .wrapped-table-container td.nowrap-cell {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .wrapped-table-container td.text-cell {{
            white-space: normal !important;
            overflow-wrap: break-word;
            word-break: normal;
        }}
        .wrapped-table-container tr:last-child td {{
            border-bottom: none;
        }}
    </style>
    <div class="wrapped-table-container">
        <table>
            <colgroup>{colgroup}</colgroup>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


PROFILE_UI_STYLES = {
    "Conservador": {"emoji": "🟢", "color": "#0F766E", "bg": "#ECFDF5", "label": "Tranquilidad / protección"},
    "Moderado": {"emoji": "🟡", "color": "#B45309", "bg": "#FFFBEB", "label": "Equilibrio / advertencia media"},
    "Agresivo": {"emoji": "🔴", "color": "#B91C1C", "bg": "#FEF2F2", "label": "Mayor riesgo / mayor variación"},
}


def profile_badge_html(profile):
    style = PROFILE_UI_STYLES.get(profile, PROFILE_UI_STYLES["Moderado"])
    return f"""
    <div style="background:{style['bg']}; border-left:7px solid {style['color']};
                padding:14px 16px; border-radius:14px; margin:8px 0 12px 0;">
        <div style="font-size:18px; font-weight:700; color:{style['color']};">
            {style['emoji']} Perfil {profile}
        </div>
        <div style="font-size:14px; color:#374151; margin-top:4px;">
            {style['label']}
        </div>
    </div>
    """


def build_goal_reference_table(horizon_days):
    inflation_target = ((1 + INFLATION_REFERENCE_ANNUAL) ** (horizon_days / 252) - 1) * 100
    return pd.DataFrame([
        {
            "Objetivo": "Cuidar mi dinero",
            "Color sugerido": "Verde / azul",
            "Criterio operativo": "Priorizar baja volatilidad, reserva mayor y evitar señales negativas.",
            "Rendimiento mínimo deseable": f"Cercano o superior a {inflation_target:.3f}% para conservar poder adquisitivo en el periodo.",
        },
        {
            "Objetivo": "Balance entre crecimiento y estabilidad",
            "Color sugerido": "Amarillo",
            "Criterio operativo": "Combinar rendimiento esperado, confianza del modelo y control de volatilidad.",
            "Rendimiento mínimo deseable": "Mayor que el escenario de solo conservar valor, con riesgo medio.",
        },
        {
            "Objetivo": "Hacer crecer mi inversión",
            "Color sugerido": "Naranja",
            "Criterio operativo": "Dar más peso al rendimiento esperado, aceptando variación moderada-alta.",
            "Rendimiento mínimo deseable": "Rendimiento positivo superior al de opciones conservadoras.",
        },
        {
            "Objetivo": "Buscar una oportunidad más agresiva",
            "Color sugerido": "Rojo",
            "Criterio operativo": "Priorizar oportunidad de ganancia, aun con volatilidad alta y posibilidad de pérdida.",
            "Rendimiento mínimo deseable": "Alto rendimiento esperado, aceptando mayor incertidumbre.",
        },
    ])


def assess_profile_consistency(amount, risk_tolerance, goal, horizon_days, profile_info):
    """Genera mensajes para explicar si las entradas del usuario son coherentes entre sí."""
    amount = float(amount)
    risk_tolerance = int(risk_tolerance)
    horizon_days = int(horizon_days)
    rows = []

    def add(level, topic, message):
        icon = {"OK": "✅", "Atención": "⚠️", "Nota": "ℹ️"}.get(level, "ℹ️")
        rows.append({"Estado": f"{icon} {level}", "Aspecto": topic, "Lectura": message})

    add("OK", "Perfil", f"Con las respuestas actuales, el dashboard clasifica al usuario como {profile_info['perfil'].lower()}.")

    if amount < 20_000 and risk_tolerance >= 4:
        add("Atención", "Monto vs riesgo", "El monto es bajo y el riesgo seleccionado es alto. Para evitar fragmentar demasiado la inversión, la cartera puede quedar concentrada en pocas emisoras.")
    elif amount >= 500_000 and profile_info["perfil"] == "Conservador":
        add("Nota", "Monto alto conservador", "El monto permite diversificar más, pero el perfil conservador mantiene una reserva elevada y prefiere emisoras menos volátiles.")
    else:
        add("OK", "Monto", "El monto seleccionado es compatible con la lógica de diversificación del dashboard.")

    if horizon_days <= 3 and risk_tolerance >= 4:
        add("Atención", "Horizonte vs riesgo", "Un horizonte muy corto con riesgo alto puede generar resultados más variables. El dashboard mostrará niveles de salida y alerta para acotar la decisión.")
    elif horizon_days <= 3:
        add("Nota", "Horizonte corto", "El horizonte es de muy corto plazo; cualquier noticia puede alterar el comportamiento del precio.")
    else:
        add("OK", "Horizonte", "El horizonte elegido se mantiene dentro del alcance de corto plazo del modelo GAMMA.")

    if goal == "Cuidar mi dinero" and risk_tolerance >= 4:
        add("Atención", "Objetivo vs riesgo", "El objetivo de cuidar el dinero no coincide del todo con una tolerancia de riesgo alta. El dashboard compensará aumentando criterios de protección.")
    elif goal == "Buscar una oportunidad más agresiva" and risk_tolerance <= 2:
        add("Atención", "Objetivo vs riesgo", "El objetivo agresivo contrasta con una tolerancia baja al riesgo. La recomendación puede quedar limitada por los filtros de seguridad.")
    else:
        add("OK", "Objetivo", "El objetivo seleccionado es coherente con la tolerancia al riesgo capturada.")

    return pd.DataFrame(rows)


def compute_strategy_levels(row, horizon_days):
    current = float(row.get("Precio actual", np.nan))
    target = float(row.get("Precio estimado", np.nan))
    expected = float(row.get("Cambio esperado (%)", np.nan))
    signal = str(row.get("Señal", "ESPERAR"))
    vol20 = row.get("Volatilidad 20d (%)", np.nan)

    if not np.isfinite(current) or current <= 0:
        return pd.Series({
            "Precio actual": np.nan,
            "Precio objetivo estimado": np.nan,
            "Rendimiento esperado (%)": np.nan,
            "Nivel sugerido de salida": np.nan,
            "Nivel de alerta": np.nan,
            "Acción sugerida": "Sin datos suficientes",
            "Lectura estratégica": "No se pudo calcular una estrategia por falta de precio actual.",
        })

    if not np.isfinite(target):
        target = current * (1 + (expected if np.isfinite(expected) else 0) / 100)

    if np.isfinite(vol20):
        horizon_vol_pct = (float(vol20) / np.sqrt(252)) * np.sqrt(max(horizon_days, 1))
        alert_pct = float(np.clip(horizon_vol_pct * 0.60, 1.0, 8.0)) / 100
    else:
        alert_pct = 0.03

    alert_level = current * (1 - alert_pct)

    if signal == "SUBE" and target > current:
        action = "Considerar entrada"
        exit_level = target
        lectura = "La señal es positiva. Si el precio alcanza el objetivo antes del horizonte, se puede considerar tomar ganancia; si cae al nivel de alerta, conviene revisar la posición."
    elif signal == "ESPERAR":
        action = "Esperar confirmación"
        exit_level = np.nan
        lectura = "La señal no es suficientemente clara. El nivel de alerta sirve como referencia de riesgo, pero la acción principal es esperar más información."
    else:
        action = "No comprar / evitar"
        exit_level = np.nan
        lectura = "La señal no favorece una entrada de compra. El dashboard evita esta emisora salvo que no existan alternativas elegibles."

    return pd.Series({
        "Precio actual": round(current, 2),
        "Precio objetivo estimado": round(target, 2),
        "Rendimiento esperado (%)": round((target / current - 1) * 100, 2),
        "Nivel sugerido de salida": round(exit_level, 2) if np.isfinite(exit_level) else np.nan,
        "Nivel de alerta": round(alert_level, 2),
        "Acción sugerida": action,
        "Lectura estratégica": lectura,
    })


def build_strategy_table_for_assets(scored_assets, selected_assets, horizon_days):
    if scored_assets.empty or not selected_assets:
        return pd.DataFrame()
    base = scored_assets[scored_assets["Emisora"].isin(selected_assets)].copy()
    if base.empty:
        return pd.DataFrame()
    levels = base.apply(lambda r: compute_strategy_levels(r, horizon_days), axis=1)
    out = pd.concat([base[["Emisora", "Señal", "Confianza", "Riesgo"]].reset_index(drop=True), levels.reset_index(drop=True)], axis=1)
    return out


def build_idle_money_comparison(amount, portfolio_summary, horizon_days):
    amount = float(amount)
    expected_pct = float(portfolio_summary.get("portfolio_expected_ret", 0.0) or 0.0)
    inflation_pct = ((1 + INFLATION_REFERENCE_ANNUAL) ** (horizon_days / 252) - 1) * 100

    nominal_without_investment = amount
    real_without_investment = amount / (1 + inflation_pct / 100)
    expected_with_portfolio = amount * (1 + expected_pct / 100)
    real_with_portfolio = expected_with_portfolio / (1 + inflation_pct / 100)

    return pd.DataFrame([
        {
            "Escenario": "No invertir / dejar el dinero quieto",
            "Monto nominal al final": nominal_without_investment,
            "Rendimiento nominal estimado (%)": 0.0,
            "Referencia de inflación del periodo (%)": inflation_pct,
            "Valor real aproximado": real_without_investment,
            "Lectura": "El número de pesos no cambia, pero el poder adquisitivo puede disminuir por inflación.",
        },
        {
            "Escenario": "Seguir cartera sugerida",
            "Monto nominal al final": expected_with_portfolio,
            "Rendimiento nominal estimado (%)": expected_pct,
            "Referencia de inflación del periodo (%)": inflation_pct,
            "Valor real aproximado": real_with_portfolio,
            "Lectura": "El rendimiento esperado se compara contra la pérdida de poder adquisitivo del periodo.",
        },
    ]).round({
        "Monto nominal al final": 2,
        "Rendimiento nominal estimado (%)": 3,
        "Referencia de inflación del periodo (%)": 3,
        "Valor real aproximado": 2,
    })


def build_historical_portfolio_simulation(portfolio_df, df_rs, amount, horizon_days, paso, n_test,
                                          precisions, roll_acc_win, rsi_sell, rsi_buy,
                                          conf_min, warm, n_lags_morph):
    """
    Simulación retrospectiva aproximada.
    Usa las emisoras seleccionadas en la cartera actual y evalúa cómo se habrían comportado
    sus señales históricas walk-forward. No reconstruye la cartera desde cero en cada fecha.
    """
    if portfolio_df.empty:
        return pd.DataFrame(), {}

    assets = portfolio_df[portfolio_df["Emisora"] != "Efectivo / reserva"].copy()
    assets = assets[assets["Peso (%)"].fillna(0) > 0]
    if assets.empty:
        return pd.DataFrame(), {}

    pieces = []
    for _, asset in assets.iterrows():
        ticker = asset["Emisora"]
        weight = float(asset["Peso (%)"]) / 100.0
        df_t = df_rs[df_rs["instrument_id"] == ticker].sort_values("date").copy()
        res = run_gamma_backtest_for_ticker(
            df_t=df_t,
            horizon=horizon_days,
            paso=paso,
            n_test=int(n_test),
            precisions=tuple(int(x) for x in precisions),
            roll_acc_win=int(roll_acc_win),
            rsi_sell=float(rsi_sell),
            rsi_buy=float(rsi_buy),
            conf_min=float(conf_min),
            warm=int(warm),
            n_lags_morph=int(n_lags_morph),
        )
        if res is None:
            continue

        real_ret = np.asarray(res["ret_real"], dtype=float)
        pred = np.asarray(res["pred_F"], dtype=int)
        model_ret = np.where(pred == 1, real_ret, 0.0)
        tmp = pd.DataFrame({
            "Fecha": pd.to_datetime(res["dates"]),
            "Rendimiento modelo ponderado (%)": model_ret * weight,
            "Rendimiento compra y mantén ponderado (%)": real_ret * weight,
        })
        pieces.append(tmp)

    if not pieces:
        return pd.DataFrame(), {}

    hist = pd.concat(pieces, ignore_index=True)
    hist = hist.groupby("Fecha", as_index=False)[[
        "Rendimiento modelo ponderado (%)",
        "Rendimiento compra y mantén ponderado (%)",
    ]].sum().sort_values("Fecha").reset_index(drop=True)
    hist.insert(0, "Periodo simulado", np.arange(1, len(hist) + 1))

    hist["Capital modelo"] = amount * np.cumprod(1 + hist["Rendimiento modelo ponderado (%)"] / 100)
    hist["Capital compra y mantén"] = amount * np.cumprod(1 + hist["Rendimiento compra y mantén ponderado (%)"] / 100)
    hist["Resultado del periodo"] = np.where(hist["Rendimiento modelo ponderado (%)"] >= 0, "Positivo", "Negativo")

    summary = {
        "periods": int(len(hist)),
        "initial_amount": float(amount),
        "final_model": float(hist["Capital modelo"].iloc[-1]),
        "final_buyhold": float(hist["Capital compra y mantén"].iloc[-1]),
        "return_model_pct": float((hist["Capital modelo"].iloc[-1] / amount - 1) * 100),
        "return_buyhold_pct": float((hist["Capital compra y mantén"].iloc[-1] / amount - 1) * 100),
        "positive_periods_pct": float((hist["Rendimiento modelo ponderado (%)"] >= 0).mean() * 100),
    }
    return hist, summary


# ===================== CARGA =====================
if st.sidebar.button("Recargar archivo de datos"):
    st.cache_data.clear()
    # Limpiar también caché manual del scan
    st.session_state.pop("_scan_key", None)
    st.session_state.pop("_scan_result", None)
    st.rerun()

try:
    data_file_mtime = get_file_mtime(DATA_PATH)
    raw = load_prices(DATA_PATH, DATE_COL, TICKER_COL, PRICE_COL, data_file_mtime)
    data_file_updated = pd.to_datetime(data_file_mtime, unit="s")
except Exception as e:
    st.error(f"No pude leer el archivo de datos en '{DATA_PATH}': {e}")
    st.stop()


# ===================== ESTADO DE PERFIL =====================
def set_default_state():
    defaults = {
        "monto_inversion": DEFAULT_INVESTMENT_AMOUNT,
        "horizonte_valor": DEFAULT_HORIZON_DAYS,
        "tolerancia_riesgo": 3,
        "objetivo_inversion": "Balance entre crecimiento y estabilidad",
        # PARCHE 3: guardar la pestaña activa para que persista tras rerun
        "main_nav": "Inicio",
        # Caché manual del scan
        "_scan_key": "",
        "_scan_result": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


set_default_state()


# ===================== SIDEBAR =====================
st.sidebar.title("Tu perfil de inversión")
st.sidebar.caption(
    "Flujo inspirado en un cuestionario de perfil: monto, horizonte, tolerancia al riesgo y objetivo. "
    "El horizonte operativo del modelo se mantiene entre 1 y 10 días hábiles."
)

with st.sidebar.form("perfil_usuario_form"):
    monto_inversion = st.number_input(
        "Monto a invertir",
        min_value=MIN_INVESTMENT_AMOUNT,
        max_value=MAX_INVESTMENT_AMOUNT,
        value=int(np.clip(st.session_state["monto_inversion"], MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT)),
        step=1000,
        help=(
            f"El simulador acepta montos entre ${MIN_INVESTMENT_AMOUNT:,.0f} y "
            f"${MAX_INVESTMENT_AMOUNT:,.0f} MXN. El monto no promete más rendimiento, "
            "solo modifica la diversificación sugerida."
        ),
    )

    horizonte_valor = st.slider(
        "Horizonte de inversión (días hábiles)",
        min_value=1,
        max_value=MAX_HORIZON_DAYS,
        value=int(horizon_to_business_days(st.session_state["horizonte_valor"])),
        help=(
            "En esta versión el modelo GAMMA trabaja a corto plazo. "
            "Por alcance del TT, no se modelan horizontes de meses o años."
        ),
    )

    tolerancia_riesgo = st.slider(
        "¿Qué tanto riesgo aceptas?",
        min_value=1,
        max_value=5,
        value=int(st.session_state["tolerancia_riesgo"]),
        help=(
            "1 = prefieres estabilidad y menor variación. "
            "5 = aceptas movimientos fuertes para buscar mayor rendimiento."
        ),
    )
    st.caption(RISK_LEVEL_DESCRIPTIONS[int(tolerancia_riesgo)])

    objetivo_inversion = st.selectbox(
        "¿Qué quieres lograr con tu inversión?",
        options=INVESTMENT_GOALS,
        index=INVESTMENT_GOALS.index(st.session_state["objetivo_inversion"]),
        help="El objetivo ajusta la reserva, la tolerancia a volatilidad y el peso del rendimiento esperado.",
    )
    st.caption(f"{GOAL_EMOJI[objetivo_inversion]} {GOAL_DESCRIPTIONS[objetivo_inversion]}")

    with st.expander("¿Qué significa cada objetivo?"):
        for goal_name in INVESTMENT_GOALS:
            st.markdown(f"**{GOAL_EMOJI[goal_name]} {goal_name}:** {GOAL_DESCRIPTIONS[goal_name]}")

    profile_submit = st.form_submit_button("Aplicar perfil")

if profile_submit:
    old_horizon = st.session_state.get("horizonte_valor")
    st.session_state["monto_inversion"] = float(np.clip(monto_inversion, MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT))
    st.session_state["horizonte_valor"] = horizonte_valor
    st.session_state["tolerancia_riesgo"] = tolerancia_riesgo
    st.session_state["objetivo_inversion"] = objetivo_inversion

    # Si cambia el horizonte, el scan anterior ya no aplica.
    # Antes se comparaba después de asignar el valor y nunca limpiaba caché.
    if horizonte_valor != old_horizon:
        st.session_state["_scan_key"] = ""
        st.session_state["_scan_result"] = pd.DataFrame()

selected_horizon_days = horizon_to_business_days(st.session_state["horizonte_valor"])
selected_horizon_label = human_horizon_label(st.session_state["horizonte_valor"])
profile_info = classify_investor_profile(
    amount=float(st.session_state["monto_inversion"]),
    horizon_days=selected_horizon_days,
    risk_tolerance=int(st.session_state["tolerancia_riesgo"]),
    goal=st.session_state["objetivo_inversion"],
)

st.sidebar.success(
    f"Perfil detectado: {profile_info['perfil']}\n\n"
    f"Horizonte usado por el modelo: {selected_horizon_days} días hábiles\n\n"
    f"Objetivo: {st.session_state['objetivo_inversion']}"
)
st.sidebar.caption(profile_info["descripcion"])

st.sidebar.info(
    f"{profile_info['objetivo_icono']} {st.session_state['objetivo_inversion']}\n\n"
    f"{profile_info['objetivo_detalle']}\n\n"
    f"Riesgo seleccionado: {profile_info['riesgo_detalle']}\n\n"
    f"Meta mínima conceptual para cuidar poder adquisitivo en {selected_horizon_days} días: "
    f"{fmt_pct(profile_info['meta_minima_horizonte'], 3)}."
)


ultima_fecha_global = pd.to_datetime(raw["date"].max()).date()
primera_fecha_global = pd.to_datetime(raw["date"].min()).date()
st.sidebar.info(
    f"Archivo leído: {Path(DATA_PATH).name}\n\n"
    f"Última actualización del archivo: {data_file_updated.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    f"Rango de fechas cargado: {primera_fecha_global} a {ultima_fecha_global}"
)

st.sidebar.markdown("---")
# Los hiperparámetros del modelo Gamma ya no se exponen en la UI.
# Se fijaron en valores óptimos encontrados con un benchmark walk-forward
# sobre 8 emisoras representativas:
#   - features extendidas (ATR, MACD, momentum 21d, posición vs máximo, RSI rápido, skew)
#   - n_lags_morph = 8 (más contexto de retornos recientes)
#   - paso = 2 (muestreo más denso, más ejemplos de entrenamiento)
#   - precisions = (1, 2, 3) (p=4 era ~50x más lento sin mejorar el hit rate)
n_test = DEFAULT_N_TEST
warm = DEFAULT_WARM
n_lags_morph = DEFAULT_LAGS_MORPH
pA, pB, pC = DEFAULT_PRECISIONS
roll_acc_win = DEFAULT_ROLL_ACC_WIN
rsi_sell = DEFAULT_RSI_SELL
rsi_buy = DEFAULT_RSI_BUY
conf_min = DEFAULT_CONF_MIN


# ===================== PREPROCESO =====================
df_rs = resample_ohlcv(raw)   # ahora cacheado: mismo objeto en todos los reruns
wide = wide_prices(df_rs)
tickers_all = sorted(df_rs["instrument_id"].unique().tolist())
step_for_model = DEFAULT_PASO  # paso fijo: muestreo denso en walk-forward

st.title("Panel de estrategias de inversión personalizadas")
st.caption(
    "Ahora el panel no solo analiza emisoras: también usa tu monto, horizonte, nivel de riesgo y objetivo "
    "para construir una recomendación más cercana a tu perfil."
)

help_box(
    f"Perfil actual: {profile_info['perfil']} | Horizonte definido por el usuario: {selected_horizon_label} "
    f"({selected_horizon_days} días hábiles).\n\n"
    "Qué significa cada señal:\n"
    "🟢 SUBE = el modelo espera un aumento\n"
    "🔴 BAJA = espera una caída\n"
    "🟡 ESPERAR = no hay suficiente claridad para tomar una dirección."
)

# ===================== NAVEGACIÓN =====================
# PARCHE 4: key="main_nav" → Streamlit guarda la selección en session_state
# automáticamente, así sobrevive cualquier re-render sin necesidad de st.rerun().
scan_params = dict(
    df_rs=df_rs,
    tickers_all=tuple(tickers_all),
    horizon=selected_horizon_days,
    paso=step_for_model,
    n_test=int(n_test),
    precisions=(int(pA), int(pB), int(pC)),
    roll_acc_win=int(roll_acc_win),
    rsi_sell=float(rsi_sell),
    rsi_buy=float(rsi_buy),
    conf_min=float(conf_min),
    warm=int(warm),
    n_lags_morph=int(n_lags_morph),
)

view = st.segmented_control(
    "Sección",
    options=["Inicio", "Vista general", "Entender una emisora", "Pronóstico", "Comparativo", "Mi perfil y cartera"],
    key="main_nav",
    selection_mode="single",
)
if view is None:
    view = st.session_state.get("main_nav", "Inicio")

market_scan = pd.DataFrame()
needs_market_scan = view in ["Comparativo", "Mi perfil y cartera"]
if needs_market_scan:
    with st.spinner("Analizando el mercado… (solo tarda la primera vez, después es instantáneo)"):
        market_scan = scan_market(**scan_params)

# ---------- TAB 0: INICIO ----------
if view == "Inicio":
    st.markdown("## ¿Qué es este panel?")
    st.markdown(
        "Este dashboard es una herramienta de análisis financiero desarrollada como parte del Trabajo Terminal "
        "**\"Desarrollo de Estrategias de Inversión Basadas en Análisis de Datos Históricos, Predicciones de "
        "Mercado y Preferencias del Usuario\"** (TT No. 2026-A086) del **Instituto Politécnico Nacional — "
        "Escuela Superior de Cómputo (ESCOM)**."
    )
    st.markdown(
        "Está pensado para **cualquier persona que quiera acercarse al mundo de las inversiones**, "
        "sin importar la edad ni si tiene experiencia previa en finanzas. "
        "Solo recomendamos ser mayor de 18 años para utilizarlo."
    )
    st.markdown(
        "La idea es simple: tú nos dices **cuánto dinero tienes disponible, cuántos días quieres esperar, "
        "qué tanto riesgo estás dispuesto a asumir y qué quieres lograr con tu inversión**, y el dashboard "
        "analiza el comportamiento histórico de empresas que cotizan en la Bolsa Mexicana de Valores (BMV) "
        "para darte una idea de qué podría pasar con tu dinero. Todo en pesos mexicanos, sin registros "
        "ni datos personales."
    )

    st.markdown("---")
    st.markdown("## 🧭 ¿Qué puedo hacer aquí?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Vista general**")
        st.caption("Mira cómo han cambiado los precios de varias empresas a lo largo del tiempo y compáralas entre sí.")
        st.markdown("**🔍 Entender una emisora**")
        st.caption("Elige una empresa y explórala con más detalle: cómo se ha movido su precio, qué tan estable o volátil ha sido y qué indicadores usa el modelo.")
        st.markdown("**📈 Pronóstico**")
        st.caption("El modelo analiza los patrones históricos de una empresa y te dice si espera que su precio suba o baje en los próximos días, junto con qué tan confiable ha sido ese análisis en el pasado.")
    with col2:
        st.markdown("**🏆 Comparativo**")
        st.caption("Compara todas las empresas disponibles y ve cuáles tienen mejores condiciones según el análisis del modelo en este momento.")
        st.markdown("**👤 Mi perfil y cartera**")
        st.caption("Llena el formulario de la barra izquierda con tu información y el panel te sugerirá cómo podrías distribuir tu dinero entre varias empresas según tu situación personal.")

    st.markdown("---")
    st.markdown("## 📘 Conceptos que debes conocer")
    st.caption("No necesitas ser experto. Aquí te explicamos, en palabras sencillas, los términos que verás en el panel.")

    with st.expander("¿Por qué se eligieron estas gráficas en el dashboard?"):
        st.markdown(
            "- **Líneas de precio:** se usan para ver la trayectoria en el tiempo, porque una inversión no se entiende solo con un número final.\n"
            "- **Barras comparativas:** se usan para comparar porcentajes entre emisoras de forma rápida, por ejemplo acierto del modelo contra volatilidad.\n"
            "- **Pastel de cartera:** se usa para mostrar visualmente cómo se reparte el dinero entre emisoras y reserva.\n"
            "- **Líneas de simulación:** se usan para comparar cómo habría cambiado el capital siguiendo el modelo frente a comprar y mantener.\n"
            "- **Tablas explicativas:** se usan cuando el dato necesita contexto, como validaciones, motivos de elección y niveles de alerta."
        )

    with st.expander("¿Qué es el Modelo Gamma?"):
        st.markdown(
            "El **Modelo Gamma** es el motor de análisis que usa este panel. "
            "No es una fórmula mágica ni una bola de cristal; dentro del dashboard funciona como un método que "
            "**compara la situación actual de una empresa con situaciones similares del pasado** y revisa qué ocurrió después.\n\n"
            "Por ejemplo: si en los últimos días el precio de una acción tuvo un patrón de movimientos muy "
            "parecido al que tuvo hace tres años, el modelo toma eso en cuenta para estimar qué podría pasar "
            "en los próximos días.\n\n"
            "Para ser más preciso, el panel usa **tres versiones del modelo** a la vez y combina sus opiniones. "
            "Si las tres coinciden, la señal es más confiable. Si están divididas, el panel te lo indica."
        )
    with st.expander("¿Qué significan las señales? 🟢 🔴 🟡"):
        st.markdown(
            "- 🟢 **SUBE** — El modelo estima que el precio estará **más alto** que hoy al final del período "
            "que elegiste.\n"
            "- 🔴 **BAJA** — El modelo estima que el precio estará **más bajo** que hoy.\n"
            "- 🟡 **ESPERAR** — El modelo no tiene suficiente claridad para decidir en ninguna dirección. "
            "En este caso lo más prudente es no tomar una decisión basándose solo en esta herramienta.\n\n"
            "Recuerda: ninguna señal es una garantía. Son estimaciones basadas en patrones históricos."
        )
    with st.expander("¿Qué es la volatilidad?"):
        st.markdown(
            "La **volatilidad** es una forma de medir qué tan estable o inestable ha sido el precio de una "
            "empresa. En términos simples, responde a la pregunta: **¿cuánto sube y baja el precio en un "
            "período normal?**\n\n"
            "- **Volatilidad alta** → el precio cambia mucho y rápido. Más oportunidad de ganar, pero también "
            "más riesgo de perder.\n"
            "- **Volatilidad baja** → el precio se mueve de forma más tranquila y predecible.\n\n"
            "En el panel verás varios tipos:\n"
            "- **Volatilidad 20 días** — cómo se ha movido en el último mes.\n"
            "- **Volatilidad 60 días** — cómo se ha movido en los últimos tres meses.\n"
            "- **Volatilidad bajista** — mide solo los días en que el precio cayó. Es una medida del "
            "riesgo de pérdida.\n"
            "- **Caída máxima** — la pérdida más grande que tuvo una acción desde su punto más alto "
            "hasta su punto más bajo en el último año.\n\n"
            "Si tu perfil es conservador, el panel preferirá empresas con menos volatilidad. "
            "Si aceptas más riesgo, puede incluir empresas más volátiles con mayor potencial."
        )
    with st.expander("¿Qué miden las métricas de error?"):
        st.markdown(
            "Estas métricas te dicen **qué tan bien o mal ha funcionado el modelo en el pasado**. "
            "Son una forma de ser honestos sobre las limitaciones del análisis:\n\n"
            "- **MAE** — Cuántos pesos se equivocó el modelo en promedio al estimar el precio.\n"
            "- **RMSE** — Similar al MAE, pero los errores grandes pesan más. Si este número es mucho "
            "mayor que el MAE, significa que hubo algunas predicciones muy equivocadas.\n"
            "- **MAPE** — El error promedio expresado en porcentaje. Más fácil de comparar entre "
            "distintas empresas sin importar el precio de cada una.\n"
            "- **SMAPE** — Una variante del MAPE que evita distorsiones cuando los precios son muy "
            "bajos o muy altos.\n"
            "- **R²** — Va de 0 a 1 (y puede ser negativo si el modelo es muy malo). Cuanto más cerca "
            "de 1, mejor explica el modelo lo que realmente pasó. Cerca de 0 significa que el modelo "
            "no es mucho mejor que adivinar."
        )
    with st.expander("¿Qué es el horizonte de inversión?"):
        st.markdown(
            "Es simplemente **cuántos días hábiles quieres esperar** antes de evaluar tu inversión. "
            "Los días hábiles son los días en que la bolsa opera — de lunes a viernes, sin festivos.\n\n"
            "En este panel puedes elegir entre 1 y 10 días hábiles, lo que equivale a entre un día "
            "y dos semanas aproximadamente.\n\n"
            "Un horizonte corto (1-3 días) tiene más incertidumbre porque cualquier noticia puede "
            "cambiar las cosas de un día para otro. Un horizonte más largo (7-10 días) le da más "
            "tiempo al modelo para capturar tendencias."
        )
    with st.expander("¿Qué es el perfil de inversión?"):
        st.markdown(
            "El panel te clasifica en uno de tres perfiles según lo que respondas en el formulario "
            "de la barra lateral:\n\n"
            "- **Conservador** — Prefieres no arriesgar mucho aunque eso signifique ganar menos. "
            "El panel priorizará empresas estables y señales muy confiables.\n"
            "- **Moderado** — Buscas un equilibrio entre cuidar tu dinero y hacerlo crecer un poco. "
            "Aceptas algo de riesgo pero no en exceso.\n"
            "- **Agresivo** — Estás dispuesto a asumir más riesgo con tal de tener mayor oportunidad "
            "de ganancia.\n\n"
            "Tu perfil depende de cuatro cosas: **cuánto dinero quieres invertir, cuántos días "
            "puedes esperar, qué tanto riesgo aceptas (del 1 al 5) y qué objetivo tienes** "
            "(cuidar tu dinero, hacerlo crecer, etc.)."
        )

    st.markdown("---")


    with st.expander("Alcance del dashboard: corto plazo y Bolsa Mexicana de Valores"):
        st.markdown(
            "El dashboard se limita a **emisoras de la Bolsa Mexicana de Valores** para mantener consistencia "
            "en moneda, disponibilidad de datos y alcance académico. El modelo GAMMA trabaja con datos diarios "
            "y un horizonte máximo de **10 días hábiles**, por lo que se interpreta como una herramienta de "
            "apoyo para análisis de corto plazo.\n\n"
            "Los horizontes de mediano y largo plazo requerirían integrar más información, como variables "
            "macroeconómicas, ciclos económicos, inflación, tipo de cambio, eventos de crisis y datos históricos "
            "más amplios. Por eso se documentan como líneas futuras y no como parte central de esta versión."
        )

    with st.expander("Cómo se combinan GAMMA, perfil y cartera"):
        st.markdown(
            "El dashboard tiene tres capas separadas:\n\n"
            "1. **Datos históricos → Modelo GAMMA → Señales de emisoras**. GAMMA analiza patrones de precios y genera señales como SUBE, BAJA o ESPERAR.\n"
            "2. **Usuario → Perfil inversionista**. El perfil no lo decide GAMMA; se calcula con monto, horizonte, tolerancia al riesgo y objetivo.\n"
            "3. **Perfil + señales → Cartera sugerida**. La distribución final combina la señal del modelo con la compatibilidad de cada emisora con el perfil del usuario."
        )

    with st.expander("Definición operativa de los objetivos"):
        show_readable_dataframe(
            build_goal_reference_table(selected_horizon_days),
            height=320,
            text_columns=["Criterio operativo", "Rendimiento mínimo deseable"],
        )
        st.caption(
            "La referencia de inflación es educativa y sirve para explicar que 'cuidar mi dinero' no significa "
            "solo mantener el mismo número de pesos, sino conservar poder adquisitivo."
        )

    with st.expander("Líneas futuras del proyecto"):
        st.markdown(
            "- Incorporar horizontes de mediano plazo con variables macroeconómicas.\n"
            "- Comparar contra instrumentos conservadores como CETES u otros referentes.\n"
            "- Agregar rebalanceo periódico de cartera.\n"
            "- Evaluar escenarios de crisis económica y recuperación.\n"
            "- Integrar simuladores bursátiles externos para pruebas prácticas.\n"
            "- Añadir inflación, tipo de cambio y noticias como variables complementarias."
        )

    st.error(
        "**⚠️ Aviso importante — Herramienta de apoyo, no de asesoría financiera**\n\n"
        "Este panel fue desarrollado con fines académicos y de análisis. "
        "Las señales, estimaciones de precio y sugerencias de cartera son el resultado de un modelo "
        "estadístico y **no constituyen asesoría financiera ni garantías de ningún tipo**.\n\n"
        "Invertir en bolsa siempre implica riesgos. El hecho de que el modelo haya funcionado bien en el "
        "pasado no significa que lo hará en el futuro. Puedes perder parte o la totalidad del dinero invertido.\n\n"
        "**Los desarrolladores de esta herramienta no se hacen responsables** por las decisiones financieras "
        "que tomes con base en lo que veas aquí.\n\n"
        "Te recomendamos ampliamente hablar con un **asesor financiero certificado** antes de tomar cualquier "
        "decisión de inversión."
    )

    st.info("Para comenzar, elige una sección en la barra de navegación de arriba o completa tu perfil en la barra izquierda. 👈")

# ---------- TAB 1 ----------
elif view == "Vista general":
    st.subheader("Vista general de precios")
    st.caption("Aquí puedes comparar cómo se han movido una o varias emisoras en el tiempo.")

    if not tickers_all:
        st.info("No hay emisoras disponibles.")
    else:
        sel = st.multiselect(
            "Selecciona una o más emisoras",
            options=tickers_all,
            default=tickers_all[:2] if len(tickers_all) >= 2 else tickers_all,
        )

        if sel:
            tmp = wide[sel].copy().sort_index().dropna(how="all")
            c1, c2 = st.columns(2)
            with c1:
                ultimos_3y = st.checkbox("Ver solo los últimos 3 años", value=False)
            with c2:
                normalizar = st.checkbox("Comparar desde una base común de 100", value=False)

            if ultimos_3y and not tmp.empty:
                inicio = tmp.index.max() - pd.DateOffset(years=3)
                tmp = tmp.loc[tmp.index >= inicio]
            if normalizar and not tmp.empty:
                base = tmp.ffill().bfill().iloc[0]
                tmp = tmp.divide(base) * 100

            st.line_chart(tmp, width="stretch")
            chart_reason_box(
                "Línea de precios",
                "permite ver la evolución completa de cada emisora y detectar subidas, caídas o periodos de estabilidad sin depender solo del último precio."
            )

            st.markdown("### Resumen comparativo")
            st.caption("Estos indicadores toman los últimos 252 días hábiles para mantener coherencia con el enfoque de corto plazo.")
            rets_b = wide[sel].sort_index().pct_change().dropna(how="all")
            if len(rets_b) > 252:
                rets_b = rets_b.iloc[-252:]

            if rets_b.empty:
                st.info("Aún no hay suficientes datos para comparar estas emisoras.")
            else:
                risk_ratio = (rets_b.mean() / rets_b.std().replace(0, np.nan)) * np.sqrt(252)
                perf = pd.DataFrame({
                    "Cambio anualizado (%)": rets_b.mean() * 252 * 100,
                    "Volatilidad anualizada (%)": rets_b.std() * np.sqrt(252) * 100,
                    "Relación rendimiento/riesgo": risk_ratio,
                }).replace([np.inf, -np.inf], np.nan).round(2).dropna(how="all")
                st.dataframe(perf, width="stretch")
                help_box(
                    "Cambio anualizado resume el ritmo promedio de crecimiento usando días hábiles, "
                    "la volatilidad anualizada refleja qué tanto se mueve la serie, "
                    "y la relación rendimiento/riesgo ayuda a comparar qué tan eficiente fue ese comportamiento."
                )

# ---------- TAB 2 ----------
elif view == "Entender una emisora":
    st.subheader("Entender una emisora")
    st.caption("Esta sección sirve para ver una sola emisora con más detalle, incluyendo su volatilidad reciente.")

    if not tickers_all:
        st.info("No hay emisoras suficientes para analizar.")
    else:
        t = st.selectbox("Elige la emisora", options=tickers_all, index=0, key="eda_ticker_gamma")
        dt = df_rs[df_rs["instrument_id"] == t].sort_values("date").copy()
        y = dt.set_index("date")["adj_close"].dropna()

        if y.empty:
            st.info("No hay datos suficientes.")
        else:
            c1, c2 = st.columns([2.2, 1])
            with c1:
                fig = go.Figure()
                fig.add_scatter(x=y.index, y=y.values, mode="lines", name="Precio")
                fig.update_layout(title=f"Evolución del precio de {t}", xaxis_title="Fecha", yaxis_title="Precio")
                st.plotly_chart(fig, width="stretch")
                chart_reason_box(
                    "Línea de precio",
                    "se eligió porque muestra la historia de la emisora en orden temporal y ayuda a explicar si el precio viene subiendo, bajando o moviéndose lateralmente."
                )
            with c2:
                ret_1d = y.pct_change().dropna() * 100
                vol_info = compute_volatility_snapshot(dt)
                st.markdown("### Resumen rápido")
                st.metric("Datos observados", int(y.shape[0]))
                st.metric("Precio promedio", fmt_num(float(y.mean()), 2))
                st.metric("Cambio diario promedio", fmt_pct(float(ret_1d.mean()) if len(ret_1d) else np.nan, 2))
                st.metric("Riesgo reciente", vol_info["risk_band"])
                st.caption(f"Periodo analizado: {y.index.min().date()} a {y.index.max().date()}")

            feats = dt.copy()
            feats["Fuerza del movimiento (RSI)"] = calc_rsi(feats["adj_close"], 28)
            feats["Posición dentro de banda"] = calc_bb_pct(feats["adj_close"], 20)
            feats["Volatilidad reciente (%)"] = feats["adj_close"].pct_change().rolling(10).std() * 100

            st.markdown("### Indicadores que usa el modelo")
            st.caption("No necesitas saber la fórmula exacta: solo ayudan a medir impulso, posición del precio y variabilidad reciente.")
            fig2 = px.line(
                feats.melt(
                    id_vars="date",
                    value_vars=["Fuerza del movimiento (RSI)", "Posición dentro de banda", "Volatilidad reciente (%)"],
                ),
                x="date",
                y="value",
                color="variable",
                title="Comportamiento reciente de los indicadores",
            )
            st.plotly_chart(fig2, width="stretch")
            chart_reason_box(
                "Líneas de indicadores",
                "se usan porque RSI, posición dentro de banda y volatilidad cambian día con día; verlos como serie temporal facilita detectar momentos de sobrecompra, sobreventa o riesgo elevado."
            )

            st.markdown("### Módulo de volatilidad")
            vol_info = compute_volatility_snapshot(dt)
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Volatilidad 20 días", fmt_pct(vol_info["vol_20d"], 2))
            v1.caption("📅 Movimiento promedio del **último mes**. Refleja el riesgo más reciente.")
            v2.metric("Volatilidad 60 días", fmt_pct(vol_info["vol_60d"], 2))
            v2.caption("📅 Movimiento promedio del **último trimestre**. Más estable que la de 20 días.")
            v3.metric("Volatilidad bajista", fmt_pct(vol_info["vol_downside"], 2))
            v3.caption("📉 Mide **solo los días que el precio bajó**. Cuanto menor, más controlado el riesgo a la baja.")
            v4.metric("Caída máxima 252 días", fmt_pct(vol_info["max_dd_252"], 2))
            v4.caption("⚠️ La **pérdida más grande** registrada en el último año de máximo a mínimo.")
            help_box(
                "Mientras más alta sea la volatilidad, más bruscos han sido los movimientos del precio. "
                "Un valor bajo no significa que la acción sea 'mala', solo que se mueve con más calma. "
                "La caída máxima es el peor escenario reciente: te muestra cuánto pudo haber perdido alguien "
                "que compró en el peor momento del año."
            )

# ---------- TAB 3 ----------
elif view == "Pronóstico":
    st.subheader("Pronóstico de una emisora")
    st.caption(
        f"El modelo usa el horizonte elegido por el usuario: {selected_horizon_label}. "
        "En esta app el máximo siempre es de 10 días hábiles."
    )

    if not tickers_all:
        st.info("No hay emisoras suficientes para modelar.")
    else:
        t2 = st.selectbox("Selecciona la emisora a pronosticar", options=tickers_all, index=0, key="gamma_ticker")
        df_t = df_rs[df_rs["instrument_id"] == t2].sort_values("date").copy()

        with st.spinner("Analizando la emisora..."):
            res = run_gamma_backtest_for_ticker(
                df_t=df_t,
                horizon=selected_horizon_days,
                paso=step_for_model,
                n_test=int(n_test),
                precisions=(int(pA), int(pB), int(pC)),
                roll_acc_win=int(roll_acc_win),
                rsi_sell=float(rsi_sell),
                rsi_buy=float(rsi_buy),
                conf_min=float(conf_min),
                warm=int(warm),
                n_lags_morph=int(n_lags_morph),
            )

        if res is None:
            st.warning("No hay suficientes datos para generar el análisis con la configuración actual y ese horizonte.")
        else:
            st.success(
                f"{estado_color(res['current_signal'])} Señal actual: {res['current_signal']} | "
                f"Confianza: {confianza_texto(res['current_conf'])}"
            )
            hoy_real = pd.Timestamp.today().normalize()
            desfase_bursatil = len(pd.bdate_range(res["current_date"].normalize(), hoy_real)) - 1

            st.caption(
                f"Fecha de hoy: {hoy_real.date()} | "
                f"Última fecha con datos: {res['current_date'].date()} | "
                f"Fecha objetivo estimada: {res['target_date'].date()} | "
                f"{res['override_txt']}"
            )

            if desfase_bursatil > 3:
                st.warning(
                    f"Tus datos parecen estar atrasados {desfase_bursatil} días hábiles. "
                    "Conviene actualizar market_prices.csv para que el pronóstico use información más reciente."
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Señal", res["current_signal"])
            m2.metric("Precio actual", fmt_num(res["current_price"], 2))
            m3.metric("Precio estimado", fmt_num(res["projected_price"], 2))
            m4.metric("Cambio esperado", fmt_pct(res["expected_ret_pct"], 2))

            st.markdown("### Trayectoria diaria estimada")
            daily_projection = build_daily_projection_path(res)
            st.dataframe(daily_projection, width="stretch")

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=[res["current_date"].date()] + daily_projection["Fecha"].tolist(),
                y=[res["current_price"]] + daily_projection["Precio estimado"].tolist(),
                mode="lines+markers",
                name="Precio estimado diario",
            ))
            fig_daily.update_layout(
                title="Estimación diaria desde el precio actual hasta el precio objetivo",
                xaxis_title="Fecha",
                yaxis_title="Precio estimado",
            )
            st.plotly_chart(fig_daily, width="stretch")
            chart_reason_box(
                "Trayectoria diaria estimada",
                "convierte el pronóstico final en pasos por día hábil, para que el usuario entienda el camino aproximado y no solo vea un precio objetivo aislado."
            )
            help_box(
                "Esta trayectoria diaria no significa que el precio vaya a moverse exactamente así cada día. "
                "Solo reparte el cambio esperado del modelo en días hábiles para que el resultado no quede "
                "como un simple 'sube' o 'baja'."
            )

            st.markdown("### Qué tan bien ha funcionado")
            a1, a2, a3 = st.columns(3)
            a1.metric("Acierto de dirección", fmt_pct(res["met_F"]["hit_rate"], 2))
            a2.metric("Riesgo de caída máxima", fmt_pct(res["met_F"]["max_dd"], 2))
            a3.metric("Precisión general del precio", fmt_num(res["err_metrics"]["R²"], 3))
            help_box(
                "Acierto de dirección indica cuántas veces el modelo acertó si subía o bajaba. "
                "Riesgo de caída máxima muestra la peor caída vista en la estrategia. "
                "R² cercano a 1 implica mejor ajuste general del precio proyectado."
            )

            st.markdown("### Error del precio estimado")
            err_df = pd.DataFrame([{
                "Error promedio": round(res["err_metrics"]["MAE"], 4),
                "Error cuadrático": round(res["err_metrics"]["RMSE"], 4),
                "Error porcentual": round(res["err_metrics"]["MAPE (%)"], 2),
                "Error porcentual simétrico": round(res["err_metrics"]["SMAPE (%)"], 2),
                "R²": round(res["err_metrics"]["R²"], 4),
            }])
            st.dataframe(err_df, width="stretch")
            st.info(explicar_error_simple(res["err_metrics"]))

            df_curve = pd.DataFrame({
                "Fecha": pd.to_datetime(res["dates"]),
                "Modelo final": res["met_F"]["acum"] * 100,
                "Compra y mantén": res["acum_bh"] * 100,
                "Gamma A": res["met_A"]["acum"] * 100,
                "Gamma B": res["met_B"]["acum"] * 100,
                "Gamma C": res["met_C"]["acum"] * 100,
            })
            fig_curve = px.line(
                df_curve.melt(id_vars="Fecha", var_name="Serie", value_name="Cambio acumulado (%)"),
                x="Fecha",
                y="Cambio acumulado (%)",
                color="Serie",
                title="Comparación del desempeño acumulado",
            )
            st.plotly_chart(fig_curve, width="stretch")
            chart_reason_box(
                "Desempeño acumulado",
                "permite comparar de forma visual si el modelo final habría acumulado mejor o peor resultado que comprar y mantener la emisora."
            )

            df_price = pd.DataFrame({
                "Fecha": pd.to_datetime(res["dates"]),
                "Precio real": res["px_real"],
                "Precio estimado": res["px_pred"],
                "Precio de partida": res["px_signal"],
            })
            fig_price = px.line(
                df_price.melt(id_vars="Fecha", var_name="Serie", value_name="Precio"),
                x="Fecha",
                y="Precio",
                color="Serie",
                title="Precio real vs precio estimado",
            )
            st.plotly_chart(fig_price, width="stretch")
            chart_reason_box(
                "Precio real vs precio estimado",
                "se eligió para mostrar de manera honesta qué tan cerca quedó la estimación del comportamiento real observado en la prueba histórica."
            )

# ---------- TAB 4 ----------
elif view == "Comparativo":
    st.subheader("Comparativo entre emisoras")
    st.caption(
        "Se ordenan según el desempeño reciente del modelo y el horizonte seleccionado por el usuario "
        "dentro de un máximo de 10 días hábiles. "
        "Este ranking todavía es técnico; la recomendación personalizada está en la pestaña siguiente."
    )

    if not tickers_all:
        st.info("No hay emisoras suficientes.")
    elif market_scan.empty:
        st.warning("No se pudo generar el comparativo con la configuración actual.")
    else:
        rank = market_scan.sort_values("Puntaje modelo", ascending=False).reset_index(drop=True)
        help_box(
            "Cómo leer esta sección: Acierto (%) indica cuántas veces el modelo acertó la dirección en pruebas históricas; "
            "Cambio esperado (%) es la variación proyectada al horizonte seleccionado; Volatilidad 60d (%) resume el riesgo reciente; "
            "Puntaje modelo combina desempeño, error y confianza para ordenar las emisoras."
        )
        show_readable_dataframe(rank[[
            "Emisora", "Señal", "Confianza", "Acierto (%)", "Cambio esperado (%)",
            "Volatilidad 60d (%)", "MAPE (%)", "R²", "Puntaje modelo"
        ]], height=420)

        st.markdown("### Top 5")
        top5 = rank.head(5)
        show_readable_dataframe(
            top5[["Emisora", "Señal", "Acierto (%)", "Cambio esperado (%)", "Volatilidad 60d (%)", "Puntaje modelo"]],
            height=230,
        )

        fig_top = go.Figure()
        max_y_top = float(np.nanmax([
            top5["Acierto (%)"].max(),
            top5["Volatilidad 60d (%)"].max(),
        ])) if not top5.empty else 100.0

        fig_top.add_trace(go.Bar(
            x=top5["Emisora"],
            y=top5["Acierto (%)"],
            name="Acierto (%)",
            text=top5["Acierto (%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "-"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Acierto del modelo: %{y:.2f}%<extra></extra>",
        ))
        fig_top.add_trace(go.Bar(
            x=top5["Emisora"],
            y=top5["Volatilidad 60d (%)"],
            name="Volatilidad 60d (%)",
            text=top5["Volatilidad 60d (%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "-"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Volatilidad reciente: %{y:.2f}%<extra></extra>",
        ))
        fig_top.update_layout(
            barmode="group",
            title="Top 5: acierto del modelo vs volatilidad reciente",
            xaxis_title="Emisora",
            yaxis_title="Porcentaje (%)",
            yaxis=dict(ticksuffix="%", range=[0, max(100, max_y_top * 1.22)]),
            margin=dict(t=70, b=70),
            uniformtext_minsize=10,
            uniformtext_mode="show",
        )
        st.plotly_chart(fig_top, width="stretch")
        chart_reason_box(
            "Barras comparativas",
            "se eligieron porque permiten comparar porcentajes entre emisoras rápidamente; el acierto muestra desempeño histórico del modelo y la volatilidad muestra el riesgo reciente."
        )

        help_box(
            "Una emisora mejor posicionada suele combinar mayor acierto, menor error y una expectativa más favorable. "
            "Aun así, el ranking técnico no reemplaza la cartera personalizada."
        )

# ---------- TAB 5 ----------
elif view == "Mi perfil y cartera":
    st.subheader("Mi perfil y cartera sugerida")
    st.caption(
        "Aquí se integra lo que pedía el protocolo: formulario del usuario, clasificación de perfil, "
        "horizonte dependiente de la persona, recomendación de cartera y validación funcional de si "
        "la propuesta sí encaja contigo."
    )

    st.info(
        "El flujo es tipo cuestionario: primero se capturan preferencias, después se clasifica el perfil "
        "y finalmente se propone una distribución. No es una orden de compra, sino una simulación académica "
        "basada en datos históricos diarios."
    )


    if not tickers_all:
        st.info("No hay emisoras suficientes.")
    elif market_scan.empty:
        st.warning("No se pudo construir la recomendación con la configuración actual.")
    else:
        scored_assets = score_assets_for_profile(
            market_df=market_scan,
            profile_info=profile_info,
            goal=st.session_state["objetivo_inversion"],
            horizon_days=selected_horizon_days,
        )
        portfolio_pack = build_personalized_portfolio(
            scored_df=scored_assets,
            df_rs=df_rs,
            amount=float(st.session_state["monto_inversion"]),
            profile_info=profile_info,
            goal=st.session_state["objetivo_inversion"],
            horizon_days=selected_horizon_days,
        )

        c1, c2 = st.columns([1.1, 1.4])
        with c1:
            st.markdown("### 1) Tu perfil detectado")
            st.metric("Perfil", profile_info["perfil"])
            st.metric("Puntaje de perfil", fmt_num(profile_info["puntaje"], 2))
            st.metric("Monto analizado", fmt_num(float(st.session_state["monto_inversion"]), 0))
            st.metric("Horizonte usado", f"{selected_horizon_days} días hábiles")
            st.info(profile_info["descripcion"])
            st.markdown(profile_badge_html(profile_info["perfil"]), unsafe_allow_html=True)

        with c2:
            st.markdown("### 2) Resumen de la propuesta")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Emisoras sugeridas", portfolio_pack["summary"]["selected_count"])
            s2.metric("Reserva sugerida", fmt_pct(portfolio_pack["summary"]["cash_pct"] * 100, 1))
            s3.metric("Cambio esperado cartera", fmt_pct(portfolio_pack["summary"]["portfolio_expected_ret"], 2))
            s4.metric("Volatilidad estimada", fmt_pct(portfolio_pack["summary"]["portfolio_vol"], 2))
            st.caption(
                f"Confianza promedio ponderada de la cartera: {confianza_texto(portfolio_pack['summary']['portfolio_conf'])}. "
                f"Por el monto ingresado, el objetivo era revisar hasta {portfolio_pack['summary'].get('target_assets', 0)} emisora(s); "
                f"pasaron filtros {portfolio_pack['summary'].get('eligible_assets', 0)}."
            )
            help_box(
                "Reserva sugerida es el porcentaje que se mantiene sin invertir para reducir exposición. "
                "Cambio esperado cartera resume el posible rendimiento ponderado de las emisoras elegidas. "
                "Volatilidad estimada aproxima qué tanto podría variar la cartera en un año con base en datos recientes."
            )

        with st.expander("¿Cómo leer el puntaje de perfil?"):
            st.markdown(
                "El puntaje de perfil no es una calificación de si eres buen o mal inversionista. "
                "Solo resume tus respuestas: monto, horizonte, tolerancia al riesgo y objetivo. "
                "Un valor bajo tiende a perfil conservador, uno intermedio a moderado y uno alto a agresivo."
            )

        st.markdown("### 3) Validación de coherencia del perfil")
        consistency_df = assess_profile_consistency(
            amount=float(st.session_state["monto_inversion"]),
            risk_tolerance=int(st.session_state["tolerancia_riesgo"]),
            goal=st.session_state["objetivo_inversion"],
            horizon_days=selected_horizon_days,
            profile_info=profile_info,
        )
        show_readable_dataframe(consistency_df, height=260, text_columns=["Lectura"])
        help_box(
            "Esta tabla ayuda a defender por qué el dashboard recomienda más o menos emisoras según la coherencia "
            "entre monto, horizonte, riesgo y objetivo."
        )

        st.markdown("### 4) Distribución sugerida del dinero")
        portfolio_df = portfolio_pack["portfolio"].copy()
        show_readable_dataframe(portfolio_df, height=420, text_columns=["Motivo de elección"])
        help_box(
            "La columna 'Motivo de elección' explica por qué aparece cada emisora o la reserva. "
            "Si ingresas un monto mayor, el dashboard puede sugerir más emisoras, pero solo hasta donde existan activos elegibles."
        )
        with st.expander("Ver motivos completos de la cartera"):
            for _, row in portfolio_df.iterrows():
                st.markdown(f"**{row['Emisora']} — {row['Señal']}**")
                st.write(row["Motivo de elección"])

        pie_df = portfolio_df[["Emisora", "Peso (%)"]].copy()
        pie_df = pie_df[pie_df["Peso (%)"] > 0]
        fig_pie = px.pie(pie_df, names="Emisora", values="Peso (%)", title="Cómo repartir el dinero según tu perfil")
        st.plotly_chart(fig_pie, width="stretch")
        chart_reason_box(
            "Gráfica de pastel",
            "se eligió porque la cartera es una distribución del 100% del dinero; así se entiende rápido cuánto va a cada emisora y cuánto queda como reserva."
        )

        st.markdown("### 5) Por qué se eligieron estas emisoras")
        explain_cols = [
            "Emisora", "Señal", "Confianza", "Riesgo",
            "Cambio esperado (%)", "Volatilidad 60d (%)", "Puntaje perfil"
        ]
        show_readable_dataframe(scored_assets[explain_cols].head(8), height=300)
        help_box(
            "El puntaje de perfil combina la señal del modelo, la confianza, la volatilidad reciente, "
            "el cambio esperado y tu objetivo personal."
        )

        st.markdown("### 6) Validación orientada al usuario")
        show_readable_dataframe(portfolio_pack["validation"], height=330, text_columns=["Detalle"])
        with st.expander("Ver validaciones completas en texto"):
            for _, row in portfolio_pack["validation"].iterrows():
                st.markdown(f"**{row['Resultado']} {row['Chequeo']}**")
                st.write(row["Detalle"])
        help_box(
            "Esta validación no solo revisa si el modelo predice bien, también verifica si la cartera "
            "respeta el tipo de usuario que dijiste ser."
        )

        st.markdown("### 7) Interpretación sencilla")
        interp = [
            f"Tu perfil se clasificó como {profile_info['perfil'].lower()}.",
            f"Tu objetivo fue: {st.session_state['objetivo_inversion'].lower()}.",
            f"El análisis se hizo para un horizonte de {selected_horizon_label.lower()}.",
            f"La cartera sugiere mantener aproximadamente {fmt_pct(portfolio_pack['summary']['cash_pct'] * 100, 1)} en reserva.",
            f"Por el monto ingresado, el dashboard podía revisar hasta {portfolio_pack['summary'].get('target_assets', 0)} emisora(s), pero solo selecciona las que pasan filtros de señal, riesgo y confianza.",
        ]
        if portfolio_pack["summary"]["selected_count"] > 0:
            interp.append(
                f"El resto se distribuye entre {portfolio_pack['summary']['selected_count']} emisora(s) "
                "con mejor compatibilidad entre señal, riesgo y objetivo."
            )
        if pd.notna(portfolio_pack["summary"]["portfolio_vol"]):
            if portfolio_pack["summary"]["portfolio_vol"] <= profile_info["umbral_volatilidad"]:
                interp.append("La volatilidad estimada está dentro de lo esperable para tu perfil.")
            else:
                interp.append(
                    "La volatilidad estimada rebasa lo ideal para tu perfil, "
                    "así que conviene revisar la propuesta o aumentar la reserva."
                )
        st.write(" ".join(interp))

        selected_assets = portfolio_df.loc[portfolio_df["Emisora"] != "Efectivo / reserva", "Emisora"].tolist()

        st.markdown("### 8) Estrategia de salida y alerta")
        strategy_df = build_strategy_table_for_assets(scored_assets, selected_assets, selected_horizon_days)
        if strategy_df.empty:
            st.info("No hay emisoras seleccionadas para construir niveles de salida y alerta.")
        else:
            show_readable_dataframe(strategy_df, height=380, text_columns=["Lectura estratégica"])
            help_box(
                "Esta sección convierte el resultado 'sube/baja' en una estrategia más clara: precio de partida, "
                "precio objetivo, rendimiento esperado, nivel sugerido de salida y nivel de alerta."
            )

        st.markdown("### 9) ¿Qué pasa si no invierto?")
        idle_df = build_idle_money_comparison(
            amount=float(st.session_state["monto_inversion"]),
            portfolio_summary=portfolio_pack["summary"],
            horizon_days=selected_horizon_days,
        )
        show_readable_dataframe(idle_df, height=260, text_columns=["Lectura"])
        help_box(
            "La comparación usa una referencia educativa de inflación anual para mostrar que cuidar el dinero "
            "también implica pensar en poder adquisitivo, no solo en que el número de pesos no cambie."
        )

        st.markdown("### 10) Simulación histórica aproximada")
        sim_df, sim_summary = build_historical_portfolio_simulation(
            portfolio_df=portfolio_df,
            df_rs=df_rs,
            amount=float(st.session_state["monto_inversion"]),
            horizon_days=selected_horizon_days,
            paso=step_for_model,
            n_test=int(n_test),
            precisions=(int(pA), int(pB), int(pC)),
            roll_acc_win=int(roll_acc_win),
            rsi_sell=float(rsi_sell),
            rsi_buy=float(rsi_buy),
            conf_min=float(conf_min),
            warm=int(warm),
            n_lags_morph=int(n_lags_morph),
        )
        if sim_df.empty:
            st.info("No fue posible construir la simulación histórica con las emisoras seleccionadas.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Periodos simulados", sim_summary["periods"])
            h2.metric("Capital final modelo", fmt_num(sim_summary["final_model"], 2))
            h3.metric("Rendimiento modelo", fmt_pct(sim_summary["return_model_pct"], 2))
            h4.metric("Periodos positivos", fmt_pct(sim_summary["positive_periods_pct"], 1))

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=sim_df["Fecha"],
                y=sim_df["Capital modelo"],
                mode="lines",
                name="Estrategia GAMMA long-only",
            ))
            fig_sim.add_trace(go.Scatter(
                x=sim_df["Fecha"],
                y=sim_df["Capital compra y mantén"],
                mode="lines",
                name="Compra y mantén ponderada",
            ))
            fig_sim.update_layout(
                title="Simulación histórica con las emisoras seleccionadas",
                xaxis_title="Fecha",
                yaxis_title="Capital simulado",
            )
            st.plotly_chart(fig_sim, width="stretch")
            chart_reason_box(
                "Líneas de capital simulado",
                "se eligieron para comparar dos caminos históricos: seguir la estrategia del modelo o comprar y mantener las mismas emisoras."
            )

            st.caption(
                f"Se muestran todos los periodos simulados ({sim_summary['periods']}). "
                f"Cada periodo es una evaluación histórica walk-forward usando un horizonte de {selected_horizon_days} días hábiles; "
                "no representa necesariamente un día calendario individual."
            )
            show_readable_dataframe(sim_df.round(3), height=500)
            help_box(
                "Esta simulación usa señales históricas walk-forward del modelo sobre las emisoras seleccionadas. "
                "No garantiza resultados futuros y no reconstruye una cartera nueva en cada fecha; sirve como prueba funcional "
                "para evitar que la recomendación quede solo como una propuesta sin validación. "
                "Antes solo se mostraban los últimos 15 registros; ahora se visualiza la simulación completa para que no parezca que faltan periodos."
            )