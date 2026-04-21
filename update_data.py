"""
update_data.py
==============
Script de actualización INCREMENTAL del archivo datos/market_prices.csv.

- Lee el CSV existente para saber qué tickers hay y cuál es la última fecha.
- Solo descarga los días que faltan desde esa fecha hasta hoy (no re-descarga todo).
- Añade las filas nuevas, elimina duplicados y sobreescribe el CSV.
- Diseñado para correr en GitHub Actions o manualmente.

Uso:
    python update_data.py
    python update_data.py --csv datos/market_prices.csv --log update.log
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────
DEFAULT_CSV  = "datos/market_prices.csv"
DEFAULT_LOG  = "datos/update.log"
EXCHANGE_SFX = ".MX"
PAUSE_SEC    = 0.8   # Pausa entre tickers para no saturar la API de yfinance
MAX_RETRIES  = 3     # Reintentos por ticker ante error de red

# Columnas del CSV de salida (mismo orden que el original)
OUTPUT_COLS = ["instrument_id", "date", "open", "high", "low",
               "close", "adj_close", "volume", "source"]


# ──────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────
def setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def instrument_to_yf(instrument_id: str) -> str:
    """Convierte 'AC_MX' → 'AC.MX', 'BIMBOA_MX' → 'BIMBOA.MX', etc."""
    # Quitar el sufijo _MX y añadir .MX
    base = instrument_id.removesuffix("_MX")
    return f"{base}{EXCHANGE_SFX}"


def fetch_incremental(ticker_yf: str, start_date: str) -> pd.DataFrame:
    """
    Descarga OHLCV desde start_date hasta hoy.
    Devuelve DataFrame vacío si falla o no hay datos nuevos.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker_yf)
            df = t.history(
                start=start_date,
                auto_adjust=False,
                actions=False,
            )
            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()

            # Normalizar nombres de columnas (yfinance puede variar)
            col_map = {
                "Date":      "date",
                "Open":      "open",
                "High":      "high",
                "Low":       "low",
                "Close":     "close",
                "Adj Close": "adj_close",
                "Volume":    "volume",
            }
            df = df.rename(columns=col_map)

            # Asegurar que existan todas las columnas necesarias
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns:
                    if col == "adj_close":
                        df["adj_close"] = df["close"]
                    elif col == "volume":
                        df["volume"] = 0
                    else:
                        df[col] = np.nan

            df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.date
            df["date"] = pd.to_datetime(df["date"])

            # Redondear precios a 4 decimales
            for col in ["open", "high", "low", "close", "adj_close"]:
                df[col] = df[col].astype(float).round(4)
            df["volume"] = df["volume"].fillna(0).astype("int64")

            return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

        except Exception as e:
            logging.warning(f"  [{ticker_yf}] Intento {attempt}/{MAX_RETRIES} fallido: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────
def update_prices(csv_path: str) -> dict:
    """
    Actualiza el CSV de precios de forma incremental.
    Devuelve un resumen con estadísticas de la actualización.
    """
    csv_path = Path(csv_path)

    # ── 1. Leer CSV existente ──────────────────────────────────────
    if not csv_path.exists():
        logging.error(f"CSV no encontrado: {csv_path}")
        sys.exit(1)

    logging.info(f"Leyendo CSV existente: {csv_path}")
    df_existing = pd.read_csv(csv_path, parse_dates=["date"])
    df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.normalize()

    tickers = sorted(df_existing["instrument_id"].unique().tolist())
    logging.info(f"Tickers detectados: {len(tickers)}")

    # ── 2. Calcular fecha de inicio para cada ticker ───────────────
    last_dates = (
        df_existing.groupby("instrument_id")["date"]
        .max()
        .to_dict()
    )

    today = pd.Timestamp.now(tz="America/Mexico_City").normalize().tz_localize(None)
    logging.info(f"Fecha hoy (CDMX): {today.date()}")

    # ── 3. Descargar datos nuevos por ticker ───────────────────────
    new_frames = []
    stats = {"updated": 0, "skipped": 0, "failed": 0, "total_rows": 0}

    for instrument_id in tickers:
        yf_ticker = instrument_to_yf(instrument_id)
        last_date = last_dates.get(instrument_id, pd.Timestamp("2000-01-01"))

        # Empezar desde el día SIGUIENTE a la última fecha conocida
        start_dt = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Si ya tenemos datos de hoy, saltar
        if last_date >= today:
            logging.info(f"  [{instrument_id}] Ya está al día ({last_date.date()}) — omitido")
            stats["skipped"] += 1
            continue

        logging.info(f"  [{instrument_id}] Descargando desde {start_dt} → {yf_ticker}")
        df_new = fetch_incremental(yf_ticker, start_dt)

        if df_new.empty:
            logging.warning(f"  [{instrument_id}] Sin datos nuevos")
            stats["failed"] += 1
            time.sleep(PAUSE_SEC)
            continue

        # Añadir columnas de identificación
        df_new["instrument_id"] = instrument_id
        df_new["source"]        = "yfinance"

        rows_added = len(df_new)
        new_frames.append(df_new[OUTPUT_COLS])
        stats["updated"]    += 1
        stats["total_rows"] += rows_added
        logging.info(f"  [{instrument_id}] +{rows_added} filas nuevas ✓")

        time.sleep(PAUSE_SEC)

    # ── 4. Combinar y guardar ──────────────────────────────────────
    if not new_frames:
        logging.info("No hay datos nuevos. CSV sin cambios.")
        return stats

    df_new_all = pd.concat(new_frames, ignore_index=True)

    # Asegurarse de que las columnas coinciden antes de concatenar
    for col in OUTPUT_COLS:
        if col not in df_existing.columns:
            df_existing[col] = np.nan

    df_combined = pd.concat(
        [df_existing[OUTPUT_COLS], df_new_all[OUTPUT_COLS]],
        ignore_index=True
    )

    # Eliminar duplicados (mantener el más reciente por si hubo correcciones)
    df_combined["date"] = pd.to_datetime(df_combined["date"])
    df_combined = (
        df_combined
        .sort_values(["instrument_id", "date"])
        .drop_duplicates(subset=["instrument_id", "date"], keep="last")
        .reset_index(drop=True)
    )

    # Guardar
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(csv_path, index=False)
    total_rows = len(df_combined)
    logging.info(
        f"\n✅ CSV actualizado: {csv_path}  "
        f"({total_rows:,} filas totales, +{stats['total_rows']} nuevas)"
    )

    return stats


# ──────────────────────────────────────────────────────────────────
# ENTRADA
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Actualización incremental de market_prices.csv")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Ruta al CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help=f"Ruta al archivo de log (default: {DEFAULT_LOG})")
    args = parser.parse_args()

    setup_logging(args.log)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logging.info("=" * 60)
    logging.info(f"  ACTUALIZACIÓN DIARIA DE DATOS — {run_ts}")
    logging.info("=" * 60)

    stats = update_prices(args.csv)

    logging.info("\n── Resumen ──────────────────────────────────────")
    logging.info(f"  Tickers actualizados: {stats['updated']}")
    logging.info(f"  Tickers sin cambios:  {stats['skipped']}")
    logging.info(f"  Tickers con error:    {stats['failed']}")
    logging.info(f"  Filas nuevas totales: {stats['total_rows']}")
    logging.info("─" * 50)

    # Código de salida 0 = éxito, 1 = errores parciales
    sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
