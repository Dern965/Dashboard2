import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG PÁGINA =====================
st.set_page_config(
    page_title="Análisis simple de acciones",
    layout="wide"
)

# ===================== PARÁMETROS FIJOS (ruta/columnas) =====================
DATA_PATH = "datos/market_prices.csv"
DATE_COL = "date"
TICKER_COL = "instrument_id"
PRICE_COL = "adj_close"

# ===================== CONFIG GAMMA =====================
FIXED_FREQ = "B"
FIXED_HORIZON = 10   # 10 días hábiles ≈ 2 semanas bursátiles
FIXED_STEP = 10
DEFAULT_WARM = 210
DEFAULT_N_TEST = 50
DEFAULT_LAGS_MORPH = 5

SEASONAL_PRIOR = {
    1: 0.0, 2: 0.0, 3: +0.10, 4: +0.20, 5: -0.10, 6: 0.0,
    7: 0.0, 8: -0.15, 9: -0.10, 10: -0.10, 11: +0.10, 12: 0.0,
}

# ===================== UTILIDADES BÁSICAS =====================
@st.cache_data
def load_prices(path, date_col, ticker_col, price_col):
    df = pd.read_csv(path, parse_dates=[date_col])
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

    keep = ["date", "instrument_id", "adj_close", "high", "low", "volume"]
    return (
        df[keep]
        .dropna(subset=["date", "instrument_id", "adj_close"])
        .sort_values(["instrument_id", "date"])
    )

def resample_ohlcv(df, freq="D"):
    out = []
    rule = {"D": "D", "W": "W", "B": "B", "M": "M", "Q": "Q"}[freq]
    for ticker, g in df.groupby("instrument_id", sort=True):
        g = g.sort_values("date").set_index("date")
        if rule == "D":
            tmp = g.copy()
        else:
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
                for j, em in enumerate(self.max_int_vals):
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
    if len(df) < max(500, warm + horizon + 30):
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

    if len(X_rows) < 80:
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
try:
    raw = load_prices(DATA_PATH, DATE_COL, TICKER_COL, PRICE_COL)
except Exception as e:
    st.error(f"No pude leer el archivo de datos en '{DATA_PATH}': {e}")
    st.stop()

# ===================== CONTROLES =====================
st.sidebar.title("Configuración")
st.sidebar.caption("Puedes dejar estos valores como están si solo quieres explorar.")

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
    conf_min = st.number_input("Confianza mínima para marcar 'esperar'", min_value=0.0, value=0.04, step=0.01)

# ===================== PREPROCESO =====================
df_rs = resample_ohlcv(raw, freq=FIXED_FREQ)
wide = wide_prices(df_rs)
tickers_all = sorted(df_rs["instrument_id"].unique().tolist())

st.title("Panel simple de análisis bursátil")
st.caption(
    "Esta versión conserva el motor Gamma, pero presenta los resultados de forma más clara. "
    "El objetivo es que cualquier persona entienda qué pasó con el precio, qué estima el modelo y cómo comparar emisoras."
)

help_box(
    "Qué significa cada señal: 🟢 SUBE = el modelo espera un aumento, 🔴 BAJA = espera una caída, "
    "🟡 ESPERAR = no hay suficiente claridad para tomar una dirección."
)

tabs = st.tabs(["Vista general", "Entender una emisora", "Pronóstico", "Comparativo"])

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
            st.caption("Estos indicadores toman los últimos 3 años con datos mensuales.")
            wide_m = wide.resample("M").last()
            rets_m = wide_m[sel].pct_change().dropna()
            if len(rets_m) > 36:
                rets_m = rets_m.iloc[-36:]

            if rets_m.empty:
                st.info("Aún no hay suficientes datos para comparar estas emisoras.")
            else:
                ann = 12
                perf = pd.DataFrame({
                    "Cambio promedio anual": rets_m.mean() * ann * 100,
                    "Variación anual": rets_m.std() * np.sqrt(ann) * 100,
                    "Relación rendimiento/riesgo": rets_m.mean() / rets_m.std(),
                }).round(2).dropna()
                st.dataframe(perf, width="stretch")
                help_box(
                    "Cambio promedio anual muestra el crecimiento medio estimado, variación anual refleja qué tanto se mueve la serie, "
                    "y la relación rendimiento/riesgo ayuda a comparar qué tan eficiente fue ese desempeño."
                )

# ---------- TAB 2 ----------
with tabs[1]:
    st.subheader("Entender una emisora")
    st.caption("Esta sección sirve para ver una sola emisora con más detalle.")

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
                fig.update_layout(
                    title=f"Evolución del precio de {t}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio"
                )
                st.plotly_chart(fig, width="stretch")
            with c2:
                ret_1d = y.pct_change().dropna() * 100
                st.markdown("### Resumen rápido")
                st.metric("Datos observados", int(y.shape[0]))
                st.metric("Precio promedio", fmt_num(float(y.mean()), 2))
                st.metric("Cambio diario promedio", fmt_pct(float(ret_1d.mean()) if len(ret_1d) else np.nan, 2))
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
                    value_vars=["Fuerza del movimiento (RSI)", "Posición dentro de banda", "Volatilidad reciente (%)"]
                ),
                x="date",
                y="value",
                color="variable",
                title="Comportamiento reciente de los indicadores"
            )
            st.plotly_chart(fig2, width="stretch")

# ---------- TAB 3 ----------
with tabs[2]:
    st.subheader("Pronóstico de una emisora")
    st.caption("El modelo estima qué podría pasar en aproximadamente 2 semanas hábiles.")

    if not tickers_all:
        st.info("No hay emisoras suficientes para modelar.")
    else:
        t2 = st.selectbox("Selecciona la emisora a pronosticar", options=tickers_all, index=0, key="gamma_ticker")
        df_t = df_rs[df_rs["instrument_id"] == t2].sort_values("date").copy()

        with st.spinner("Analizando la emisora..."):
            res = run_gamma_backtest_for_ticker(
                df_t=df_t,
                horizon=FIXED_HORIZON,
                paso=FIXED_STEP,
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
            st.warning("No hay suficientes datos para generar el análisis con la configuración actual.")
        else:
            st.success(
                f"{estado_color(res['current_signal'])} Señal actual: {res['current_signal']} | "
                f"Confianza: {confianza_texto(res['current_conf'])}"
            )
            st.caption(
                f"Fecha actual: {res['current_date'].date()} | Fecha estimada: {res['target_date'].date()} | {res['override_txt']}"
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
                title="Comparación del desempeño acumulado"
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
                title="Precio real vs precio estimado"
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
    st.caption("Se ordenan según desempeño reciente del modelo y calidad del pronóstico.")

    if not tickers_all:
        st.info("No hay emisoras suficientes.")
    else:
        results = []
        progress = st.progress(0.0, text="Analizando emisoras...")

        for i, ticker in enumerate(tickers_all):
            df_t = df_rs[df_rs["instrument_id"] == ticker].sort_values("date").copy()
            res = run_gamma_backtest_for_ticker(
                df_t=df_t,
                horizon=FIXED_HORIZON,
                paso=FIXED_STEP,
                n_test=int(n_test),
                precisions=(int(pA), int(pB), int(pC)),
                roll_acc_win=int(roll_acc_win),
                rsi_sell=float(rsi_sell),
                rsi_buy=float(rsi_buy),
                conf_min=float(conf_min),
                warm=int(warm),
                n_lags_morph=int(n_lags_morph),
            )

            if res is not None:
                score = (
                    res["met_F"]["hit_rate"]
                    + res["met_F"]["sharpe"]
                    - 0.25 * res["err_metrics"]["SMAPE (%)"]
                    - 0.10 * res["err_metrics"]["MAPE (%)"]
                )
                results.append({
                    "Emisora": ticker,
                    "Señal": res["current_signal"],
                    "Confianza": confianza_texto(res["current_conf"]),
                    "Acierto (%)": round(res["met_F"]["hit_rate"], 2),
                    "Caída máxima (%)": round(res["met_F"]["max_dd"], 2),
                    "Cambio esperado (%)": round(res["expected_ret_pct"], 2),
                    "Precio estimado": round(res["projected_price"], 2),
                    "Error porcentual (%)": round(res["err_metrics"]["MAPE (%)"], 2),
                    "R²": round(res["err_metrics"]["R²"], 3),
                    "Score": round(score, 3),
                })

            progress.progress((i + 1) / len(tickers_all), text=f"Analizando emisoras... {i + 1}/{len(tickers_all)}")

        progress.empty()

        if not results:
            st.warning("No se pudo generar el comparativo con la configuración actual.")
        else:
            rank = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
            st.dataframe(rank, width="stretch")

            st.markdown("### Top 5")
            top5 = rank.head(5)
            st.dataframe(top5, width="stretch")

            fig_top = go.Figure()
            fig_top.add_bar(x=top5["Emisora"], y=top5["Acierto (%)"], name="Acierto (%)")
            fig_top.add_bar(x=top5["Emisora"], y=top5["Error porcentual (%)"], name="Error (%)")
            fig_top.update_layout(
                barmode="group",
                title="Top 5: acierto del modelo vs error del pronóstico",
                xaxis_title="Emisora",
                yaxis_title="Valor"
            )
            st.plotly_chart(fig_top, width="stretch")

            help_box(
                "Una emisora mejor posicionada suele combinar mayor acierto, menor error y una expectativa más favorable. "
                "Aun así, el ranking es una ayuda de análisis, no una garantía de inversión."
            )
