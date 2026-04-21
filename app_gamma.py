import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
DEFAULT_LAGS_MORPH = 5
DEFAULT_CONF_MIN = 0.04

SEASONAL_PRIOR = {
    1: 0.0, 2: 0.0, 3: +0.10, 4: +0.20, 5: -0.10, 6: 0.0,
    7: 0.0, 8: -0.15, 9: -0.10, 10: -0.10, 11: +0.10, 12: 0.0,
}

# ===================== UTILIDADES BÁSICAS =====================
def get_file_mtime(path):
    return os.path.getmtime(path)


@st.cache_data
def load_prices(path, date_col, ticker_col, price_col, file_mtime):
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"No encontré la columna de fecha: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

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
    out = out.sort_values(["instrument_id", "date"]).drop_duplicates(subset=["instrument_id", "date"], keep="last")
    return out


def resample_ohlcv(df, freq="B"):
    # En esta app siempre trabajamos en días hábiles.
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


# ===================== MODELO GAMMA =====================
class GammaBinary:
    def __init__(self, precision=2):
        self.precision = precision
        self.max_int_vals = None
        self.rho = 0
        self.X_enc = None
        self.y_train = None
        self.classes = None
        self.n_cls = None

    def _encode(self, v, em):
        iv = int(round(np.clip(float(v), 0.0, 1.0) * (10 ** self.precision)))
        em = int(em)
        enc = np.zeros(em, dtype=np.int8)
        enc[:min(iv, em)] = 1
        return enc

    def _encode_batch(self, X):
        rows = []
        for row in X:
            parts = [self._encode(v, self.max_int_vals[j]) for j, v in enumerate(row)]
            rows.append(np.concatenate(parts))
        return np.array(rows, dtype=np.int8)

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
        return self

    def predict_with_score(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        Xe = self._encode_batch(X_test)
        results = []
        for pat in Xe:
            winner = None
            last_scores = {c: 0.0 for c in self.classes}
            for theta in range(self.rho + 1):
                csums = {c: 0 for c in self.classes}
                start = 0
                for _, em in enumerate(self.max_int_vals):
                    end = start + int(em)
                    seg_tr = self.X_enc[:, start:end].astype(np.int16)
                    seg_te = pat[start:end].astype(np.int16)
                    dists = np.sum(np.abs(seg_tr - seg_te), axis=1)
                    ok = dists <= theta
                    for c in self.classes:
                        csums[c] += np.sum(ok & (self.y_train == c))
                    start = end
                scores = {c: csums[c] / self.n_cls[c] for c in self.classes}
                last_scores = scores
                ms = max(scores.values())
                cands = [c for c, s in scores.items() if s == ms and s > 0]
                if len(cands) == 1:
                    winner = cands[0]
                    break
            if winner is None:
                winner = max(last_scores, key=last_scores.get)
            sv = sorted(last_scores.values(), reverse=True)
            conf = (sv[0] - sv[1]) if len(sv) >= 2 and sv[0] > 0 else 0.0
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
    df["ret_fwd"] = df["adj_close"].pct_change(horizon).shift(-horizon) * 100
    df["target"] = (df["ret_fwd"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    contexto = ["high_low_pct", "vol_real", "day_of_week", "bb_pct", "rsi_28", "ret_5d", "vol_ratio_5"]
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


def robust_scale_train_test(X, split_idx):
    p2 = np.percentile(X[:split_idx], 2, axis=0)
    p98 = np.percentile(X[:split_idx], 98, axis=0)
    denom = np.where((p98 - p2) == 0, 1.0, (p98 - p2))
    Xc = np.clip(X, p2, p98)
    Xs = (Xc - p2) / denom
    Xs[:split_idx] = np.clip(Xs[:split_idx], 0, 1)
    Xs[split_idx:] = np.clip(Xs[split_idx:], 0, 1)
    return Xs


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
        Xtr = X_sc[:idx_train]
        ytr = y[:idx_train]
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

        prior = SEASONAL_PRIOR.get(int(months[idx_train]), 0.0)
        vf = dict(va)
        vf[1] += prior
        vf[0] -= prior
        ens_final = 1 if vf[1] >= vf[0] else 0

        if rsi_arr[idx_train] < rsi_sell:
            ens_final = 0
        elif rsi_arr[idx_train] > rsi_buy:
            ens_final = 1

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

        ret_hist = np.array(ret_fwd_arr[:idx_train])
        y_hist = np.array(y[:idx_train])
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

    clf_fa = GammaBinary(pA).fit(X_sc, y)
    clf_fb = GammaBinary(pB).fit(X_sc, y)
    clf_fc = GammaBinary(pC).fit(X_sc, y)
    ra_f = clf_fa.predict_with_score([X_sc[-1]])[0]
    rb_f = clf_fb.predict_with_score([X_sc[-1]])[0]
    rc_f = clf_fc.predict_with_score([X_sc[-1]])[0]

    w_a_f = max(np.mean(roll_correct_A[-roll_acc_win:]), 0.1) if len(roll_correct_A) else 1.0
    w_b_f = max(np.mean(roll_correct_B[-roll_acc_win:]), 0.1) if len(roll_correct_B) else 1.0
    w_c_f = max(np.mean(roll_correct_C[-roll_acc_win:]), 0.1) if len(roll_correct_C) else 1.0

    vf = {0: 0.0, 1: 0.0}
    vf[ra_f[0]] += w_a_f * (1 + ra_f[1])
    vf[rb_f[0]] += w_b_f * (1 + rb_f[1])
    vf[rc_f[0]] += w_c_f * (1 + rc_f[1])

    mes_hoy = int(months[-1])
    prior_hoy = SEASONAL_PRIOR.get(mes_hoy, 0.0)
    vf[1] += prior_hoy
    vf[0] -= prior_hoy

    ens_f = 1 if vf[1] >= vf[0] else 0
    conf_f = abs(vf[1] - vf[0]) / (vf[1] + vf[0] + 1e-9)

    rsi_hoy = float(rsi_arr[-1])
    override_txt = f"RSI-28 = {rsi_hoy:.1f} (zona normal)"
    if rsi_hoy < rsi_sell:
        ens_f = 0
        override_txt = f"RSI bajo: {rsi_hoy:.1f} < {rsi_sell}"
    elif rsi_hoy > rsi_buy:
        ens_f = 1
        override_txt = f"RSI alto: {rsi_hoy:.1f} > {rsi_buy}"

    precio_hoy = float(px_signal[-1])
    fecha_hoy = pd.to_datetime(fechas[-1])
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
    horizon_score = 0.0 if horizon_days <= 3 else 0.6 if horizon_days <= 7 else 1.0
    goal_score = {
        "Cuidar mi dinero": -1,
        "Balance entre crecimiento y estabilidad": 0,
        "Hacer crecer mi inversión": 1,
        "Buscar una oportunidad más agresiva": 2,
    }[goal]
    total = (risk_tolerance - 1) * 1.6 + horizon_score + goal_score

    if total <= 2.5:
        profile = "Conservador"
        description = "Priorizas estabilidad, menor exposición y más protección del capital."
    elif total <= 5.5:
        profile = "Moderado"
        description = "Buscas crecer sin dejar de cuidar el riesgo."
    else:
        profile = "Agresivo"
        description = "Aceptas movimientos fuertes con tal de buscar mayor crecimiento."

    amount_note = ""
    if amount < 20000:
        amount_note = " Como el monto es relativamente pequeño, conviene evitar repartirlo en demasiadas emisoras."
    elif amount > 150000:
        amount_note = " Como el monto es más alto, sí tiene sentido diversificar un poco más."

    return {
        "perfil": profile,
        "puntaje": round(float(total), 2),
        "descripcion": description + amount_note,
        "horizonte_dias": int(horizon_days),
        "umbral_volatilidad": {"Conservador": 24, "Moderado": 34, "Agresivo": 50}[profile],
        "cash_base": {"Conservador": 0.30, "Moderado": 0.15, "Agresivo": 0.05}[profile],
        "max_peso": {"Conservador": 0.30, "Moderado": 0.40, "Agresivo": 0.50}[profile],
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
    if amount < 20000:
        return min(2, base_n)
    if amount < 60000:
        return max(2, base_n)
    return base_n + 1


@st.cache_data(show_spinner=False)
def scan_market(df_rs, tickers_all, horizon, paso, n_test, precisions, roll_acc_win, rsi_sell, rsi_buy, conf_min, warm, n_lags_morph):
    rows = []
    for ticker in tickers_all:
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
            continue

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
        rows.append({
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
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Puntaje modelo", ascending=False).reset_index(drop=True)


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
            },
        }

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

    n_assets = infer_asset_count(amount, profile_info["n_base"])
    cap_weight = profile_info["max_peso"] * (1 - cash_pct)

    eligible = scored_df[scored_df["Elegible"]].copy()
    if eligible.empty:
        top = scored_df.head(1).copy()
        eligible = top.assign(Elegible=False)
        cash_pct = 1.0
        invest_pct = 0.0
    else:
        invest_pct = 1 - cash_pct
        eligible = eligible.head(n_assets).copy()

    if invest_pct > 0:
        weights = normalize_weights_with_cap(eligible["Puntaje perfil"], total_weight=invest_pct, cap=cap_weight)
        eligible["Peso"] = weights.values
    else:
        eligible["Peso"] = 0.0

    eligible["Monto sugerido"] = eligible["Peso"] * amount
    eligible["Señal visual"] = eligible["Señal"].map(signal_emoji) + " " + eligible["Señal"]

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
    }])

    portfolio = pd.concat([eligible, cash_row], ignore_index=True, sort=False)
    portfolio["Peso (%)"] = portfolio["Peso"] * 100
    portfolio = portfolio[[
        "Emisora", "Señal visual", "Confianza", "Riesgo", "Cambio esperado (%)",
        "Volatilidad 60d (%)", "Peso (%)", "Monto sugerido"
    ]].rename(columns={
        "Señal visual": "Señal",
        "Cambio esperado (%)": f"Cambio esperado al horizonte (%)",
        "Volatilidad 60d (%)": "Volatilidad reciente (%)",
    })

    asset_rows = portfolio[portfolio["Emisora"] != "Efectivo / reserva"].copy()
    selected_assets = asset_rows["Emisora"].tolist()

    portfolio_expected_ret = float(asset_rows[f"Cambio esperado al horizonte (%)"].fillna(0).mul(asset_rows["Peso (%)"] / 100).sum())
    portfolio_conf = float(scored_df.set_index("Emisora").loc[selected_assets, "Confianza num"].fillna(0).mul(asset_rows["Peso (%)"] / 100).sum()) if selected_assets else 0.0

    portfolio_vol = np.nan
    if selected_assets:
        wide = df_rs[df_rs["instrument_id"].isin(selected_assets)].pivot(index="date", columns="instrument_id", values="adj_close").sort_index()
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
    diversification_ok = len(selected_assets) >= min(2, infer_asset_count(amount, profile_info["n_base"])) or amount < 20000
    no_baja_ok = "BAJA" not in scored_df.set_index("Emisora").reindex(selected_assets)["Señal"].fillna("").tolist()

    validation = pd.DataFrame([
        {
            "Chequeo": "Riesgo acorde a tu perfil",
            "Resultado": "✅ Sí" if risk_ok else "⚠️ Revisar",
            "Detalle": f"Volatilidad estimada del portafolio: {fmt_pct(portfolio_vol, 2)}. Límite de referencia para tu perfil: {profile_info['umbral_volatilidad']}%."
        },
        {
            "Chequeo": "Reserva de efectivo suficiente",
            "Resultado": "✅ Sí" if cash_ok else "⚠️ Revisar",
            "Detalle": f"Reserva sugerida: {fmt_pct(cash_pct * 100, 1)} del total."
        },
        {
            "Chequeo": "Concentración razonable",
            "Resultado": "✅ Sí" if concentration_ok else "⚠️ Revisar",
            "Detalle": f"Peso máximo en una sola emisora: {fmt_pct(max_weight_pct, 1)}."
        },
        {
            "Chequeo": "Diversificación mínima",
            "Resultado": "✅ Sí" if diversification_ok else "⚠️ Revisar",
            "Detalle": f"Emisoras seleccionadas: {len(selected_assets)}."
        },
        {
            "Chequeo": "Evita señales claramente negativas",
            "Resultado": "✅ Sí" if no_baja_ok else "⚠️ Revisar",
            "Detalle": "La cartera propuesta evita activos con señal BAJA cuando fue posible."
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


def explicar_error_simple(m):
    return (
        f"Error promedio: {fmt_num(m['MAE'], 3)} | "
        f"Error porcentual: {fmt_pct(m['MAPE (%)'], 2)} | "
        f"Ajuste general (R²): {fmt_num(m['R²'], 3)}"
    )


def help_box(text):
    st.info(text)


# ===================== CARGA =====================
if st.sidebar.button("Recargar archivo de datos"):
    st.cache_data.clear()
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
        "monto_inversion": 50000,
        "horizonte_valor": DEFAULT_HORIZON_DAYS,
        "tolerancia_riesgo": 3,
        "objetivo_inversion": "Balance entre crecimiento y estabilidad",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


set_default_state()


# ===================== SIDEBAR =====================
st.sidebar.title("Tu perfil de inversión")
st.sidebar.caption("Llena este formulario para que el panel adapte el horizonte, el perfil y la cartera sugerida. El horizonte siempre será de 1 a 10 días hábiles.")

with st.sidebar.form("perfil_usuario_form"):
    monto_inversion = st.number_input("Monto a invertir", min_value=1000, value=int(st.session_state["monto_inversion"]), step=1000)
    horizonte_valor = st.slider(
        "Horizonte de inversión (días hábiles)",
        min_value=1,
        max_value=MAX_HORIZON_DAYS,
        value=int(horizon_to_business_days(st.session_state["horizonte_valor"])),
        help="En esta versión el modelo siempre trabaja con un máximo de 10 días hábiles, es decir hasta 2 semanas de mercado.",
    )
    tolerancia_riesgo = st.slider("¿Qué tanto riesgo aceptas?", min_value=1, max_value=5, value=int(st.session_state["tolerancia_riesgo"]), help="1 = muy poco, 5 = alto")
    objetivo_inversion = st.selectbox(
        "¿Qué quieres lograr con tu inversión?",
        options=[
            "Cuidar mi dinero",
            "Balance entre crecimiento y estabilidad",
            "Hacer crecer mi inversión",
            "Buscar una oportunidad más agresiva",
        ],
        index=[
            "Cuidar mi dinero",
            "Balance entre crecimiento y estabilidad",
            "Hacer crecer mi inversión",
            "Buscar una oportunidad más agresiva",
        ].index(st.session_state["objetivo_inversion"]),
    )
    profile_submit = st.form_submit_button("Aplicar perfil")

if profile_submit:
    st.session_state["monto_inversion"] = monto_inversion
    st.session_state["horizonte_valor"] = horizonte_valor
    st.session_state["tolerancia_riesgo"] = tolerancia_riesgo
    st.session_state["objetivo_inversion"] = objetivo_inversion

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

ultima_fecha_global = pd.to_datetime(raw["date"].max()).date()
primera_fecha_global = pd.to_datetime(raw["date"].min()).date()
st.sidebar.info(
    f"Archivo leído: {Path(DATA_PATH).name}\n\n"
    f"Última actualización del archivo: {data_file_updated.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    f"Rango de fechas cargado: {primera_fecha_global} a {ultima_fecha_global}"
)

st.sidebar.markdown("---")
st.sidebar.title("Configuración técnica")
st.sidebar.caption("Estos ajustes ayudan al modelo. Si no estás seguro, puedes dejarlos como están.")

with st.sidebar.expander("Ajustes avanzados del modelo"):
    n_test = st.number_input("Pruebas históricas", min_value=20, value=DEFAULT_N_TEST, step=5)
    warm = st.number_input("Datos mínimos para arrancar", min_value=60, value=DEFAULT_WARM, step=10)
    n_lags_morph = st.number_input("Cambios recientes usados por el modelo", min_value=2, value=DEFAULT_LAGS_MORPH, step=1)
    pA = st.number_input("Nivel Gamma A", min_value=1, value=1, step=1)
    pB = st.number_input("Nivel Gamma B", min_value=1, value=3, step=1)
    pC = st.number_input("Nivel Gamma C", min_value=1, value=4, step=1)
    roll_acc_win = st.number_input("Ventana de evaluación interna", min_value=3, value=10, step=1)
    rsi_sell = st.number_input("Límite RSI para señal de baja", min_value=1, max_value=99, value=22, step=1)
    rsi_buy = st.number_input("Límite RSI para señal de subida", min_value=1, max_value=99, value=72, step=1)
    conf_min = st.number_input("Confianza mínima para marcar 'esperar'", min_value=0.0, value=DEFAULT_CONF_MIN, step=0.01)


# ===================== PREPROCESO =====================
df_rs = resample_ohlcv(raw, freq=FIXED_FREQ)
wide = wide_prices(df_rs)
tickers_all = sorted(df_rs["instrument_id"].unique().tolist())
step_for_model = int(max(1, min(selected_horizon_days, 5)))

st.title("Panel de estrategias de inversión personalizadas")
st.caption(
    "Ahora el panel no solo analiza emisoras: también usa tu monto, horizonte, nivel de riesgo y objetivo para construir una recomendación más cercana a tu perfil."
)

help_box(
    f"Perfil actual: {profile_info['perfil']} | Horizonte definido por el usuario: {selected_horizon_label} "
    f"({selected_horizon_days} días hábiles).\n"
    "Qué significa cada señal: 🟢 SUBE = el modelo espera un aumento, 🔴 BAJA = espera una caída, "
    "🟡 ESPERAR = no hay suficiente claridad para tomar una dirección."
)

tabs = st.tabs(["Vista general", "Entender una emisora", "Pronóstico", "Comparativo", "Mi perfil y cartera"])

# ---------- TAB 1 ----------
with tabs[0]:
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
                    "Cambio anualizado resume el ritmo promedio de crecimiento usando días hábiles, la volatilidad anualizada refleja qué tanto se mueve la serie, "
                    "y la relación rendimiento/riesgo ayuda a comparar qué tan eficiente fue ese comportamiento."
                )

# ---------- TAB 2 ----------
with tabs[1]:
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

            st.markdown("### Módulo de volatilidad")
            vol_info = compute_volatility_snapshot(dt)
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Volatilidad 20 días", fmt_pct(vol_info["vol_20d"], 2))
            v2.metric("Volatilidad 60 días", fmt_pct(vol_info["vol_60d"], 2))
            v3.metric("Volatilidad bajista", fmt_pct(vol_info["vol_downside"], 2))
            v4.metric("Caída máxima 252 días", fmt_pct(vol_info["max_dd_252"], 2))
            help_box(
                "Aquí la volatilidad se usa como medida formal de riesgo reciente. Mientras más alta sea, más bruscos han sido los movimientos del precio."
            )

# ---------- TAB 3 ----------
with tabs[2]:
    st.subheader("Pronóstico de una emisora")
    st.caption(f"El modelo usa el horizonte elegido por el usuario: {selected_horizon_label}. En esta app el máximo siempre es de 10 días hábiles.")

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
            desfase_bursatil = len(pd.bdate_range(res['current_date'].normalize(), hoy_real)) - 1

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

            with st.expander("Ver detalle técnico de clasificaciones"):
                cls_df = pd.DataFrame({
                    "Fecha": pd.to_datetime(res["dates"]),
                    "Real": res["real_cls"],
                    "Gamma A": res["pred_A"],
                    "Gamma B": res["pred_B"],
                    "Gamma C": res["pred_C"],
                    "Modelo final": res["pred_F"],
                })
                st.caption("1 = subió, 0 = bajó")
                st.dataframe(cls_df.tail(20), width="stretch")

# ---------- TAB 4 ----------
with tabs[3]:
    st.subheader("Comparativo entre emisoras")
    st.caption(
        "Se ordenan según el desempeño reciente del modelo y el horizonte seleccionado por el usuario dentro de un máximo de 10 días hábiles. "
        "Este ranking todavía es técnico; la recomendación personalizada está en la pestaña siguiente."
    )

    if not tickers_all:
        st.info("No hay emisoras suficientes.")
    else:
        with st.spinner("Analizando emisoras..."):
            market_scan = scan_market(
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

        if market_scan.empty:
            st.warning("No se pudo generar el comparativo con la configuración actual.")
        else:
            rank = market_scan.sort_values("Puntaje modelo", ascending=False).reset_index(drop=True)
            st.dataframe(rank[[
                "Emisora", "Señal", "Confianza", "Acierto (%)", "Cambio esperado (%)",
                "Volatilidad 60d (%)", "MAPE (%)", "R²", "Puntaje modelo"
            ]], width="stretch")

            st.markdown("### Top 5")
            top5 = rank.head(5)
            st.dataframe(top5[["Emisora", "Señal", "Acierto (%)", "Cambio esperado (%)", "Volatilidad 60d (%)", "Puntaje modelo"]], width="stretch")

            fig_top = go.Figure()
            fig_top.add_bar(x=top5["Emisora"], y=top5["Acierto (%)"], name="Acierto (%)")
            fig_top.add_bar(x=top5["Emisora"], y=top5["Volatilidad 60d (%)"], name="Volatilidad 60d (%)")
            fig_top.update_layout(
                barmode="group",
                title="Top 5: acierto del modelo vs volatilidad reciente",
                xaxis_title="Emisora",
                yaxis_title="Valor",
            )
            st.plotly_chart(fig_top, width="stretch")

            help_box(
                "Una emisora mejor posicionada suele combinar mayor acierto, menor error y una expectativa más favorable. "
                "Aun así, el ranking técnico no reemplaza la cartera personalizada."
            )

# ---------- TAB 5 ----------
with tabs[4]:
    st.subheader("Mi perfil y cartera sugerida")
    st.caption(
        "Aquí se integra lo que pedía el protocolo: formulario del usuario, clasificación de perfil, horizonte dependiente de la persona, "
        "recomendación de cartera y validación funcional de si la propuesta sí encaja contigo."
    )

    if not tickers_all:
        st.info("No hay emisoras suficientes.")
    else:
        with st.spinner("Construyendo recomendación personalizada..."):
            market_scan = scan_market(
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

        if market_scan.empty:
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

            with c2:
                st.markdown("### 2) Resumen de la propuesta")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Emisoras sugeridas", portfolio_pack["summary"]["selected_count"])
                s2.metric("Reserva sugerida", fmt_pct(portfolio_pack["summary"]["cash_pct"] * 100, 1))
                s3.metric("Cambio esperado cartera", fmt_pct(portfolio_pack["summary"]["portfolio_expected_ret"], 2))
                s4.metric("Volatilidad estimada", fmt_pct(portfolio_pack["summary"]["portfolio_vol"], 2))
                st.caption(
                    f"Confianza promedio ponderada de la cartera: {confianza_texto(portfolio_pack['summary']['portfolio_conf'])}."
                )

            st.markdown("### 3) Distribución sugerida del dinero")
            portfolio_df = portfolio_pack["portfolio"].copy()
            st.dataframe(portfolio_df, width="stretch")

            pie_df = portfolio_df[["Emisora", "Peso (%)"]].copy()
            pie_df = pie_df[pie_df["Peso (%)"] > 0]
            fig_pie = px.pie(pie_df, names="Emisora", values="Peso (%)", title="Cómo repartir el dinero según tu perfil")
            st.plotly_chart(fig_pie, width="stretch")

            st.markdown("### 4) Por qué se eligieron estas emisoras")
            explain_cols = [
                "Emisora", "Señal", "Confianza", "Riesgo", "Cambio esperado (%)", "Volatilidad 60d (%)", "Puntaje perfil"
            ]
            st.dataframe(scored_assets[explain_cols].head(8), width="stretch")
            help_box(
                "El puntaje de perfil combina la señal del modelo, la confianza, la volatilidad reciente, el cambio esperado y tu objetivo personal."
            )

            st.markdown("### 5) Validación orientada al usuario")
            st.dataframe(portfolio_pack["validation"], width="stretch")
            help_box(
                "Esta validación no solo revisa si el modelo predice bien, también verifica si la cartera respeta el tipo de usuario que dijiste ser."
            )

            st.markdown("### 6) Interpretación sencilla")
            interp = [
                f"Tu perfil se clasificó como {profile_info['perfil'].lower()}.",
                f"El análisis se hizo para un horizonte de {selected_horizon_label.lower()}.",
                f"La cartera sugiere mantener aproximadamente {fmt_pct(portfolio_pack['summary']['cash_pct'] * 100, 1)} en reserva.",
            ]
            if portfolio_pack["summary"]["selected_count"] > 0:
                interp.append(
                    f"El resto se distribuye entre {portfolio_pack['summary']['selected_count']} emisora(s) con mejor compatibilidad entre señal, riesgo y objetivo."
                )
            if pd.notna(portfolio_pack["summary"]["portfolio_vol"]):
                if portfolio_pack["summary"]["portfolio_vol"] <= profile_info["umbral_volatilidad"]:
                    interp.append("La volatilidad estimada está dentro de lo esperable para tu perfil.")
                else:
                    interp.append("La volatilidad estimada rebasa lo ideal para tu perfil, así que conviene revisar la propuesta o aumentar la reserva.")
            st.write(" ".join(interp))
