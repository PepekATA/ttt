# app.py
"""
Trading Dashboard — auto-guardado de API keys, lectura de activos 24/7 (crypto) desde Alpaca,
dashboard con tendencia, probabilidad y duración estimada en múltiples horizontes.
Heurística simple: EMA crossover + RSI -> probabilidad y duración.
NOTAS:
 - Para producción y seguridad, usa Streamlit Secrets (no guardar claves en disco).
 - Algunos símbolos FX pueden no estar disponibles en Alpaca; puedes añadir símbolos manualmente.
"""
import os
import json
import time
from datetime import datetime, timedelta
import math

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Try to import alpaca libraries (compatible con alpaca-trade-api)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_LIB = "tradeapi"
except Exception:
    try:
        # alternativa: alpaca-py (si está instalada)
        from alpaca.data.rest import REST
        from alpaca.trading.client import TradingClient
        ALPACA_LIB = "alpaca-py"
    except Exception:
        ALPACA_LIB = None

st.set_page_config(layout="wide", page_title="QuickTrend 24/7 Dashboard")
st.title("QuickTrend 24/7 — Señales rápidas (EMA+RSI)")

CRED_FILE = "credentials.json"

# ---------------------------
# Util: guardar / cargar credenciales localmente
# ---------------------------
def save_credentials(key, secret):
    data = {"ALPACA_API_KEY": key, "ALPACA_API_SECRET": secret}
    with open(CRED_FILE, "w") as f:
        json.dump(data, f)

def load_credentials():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    return None

creds = load_credentials()

# UI para credenciales (solo si no existen)
if not creds:
    st.info("Introduce tus claves Alpaca (solo la primera vez se guardan localmente).")
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("ALPACA API KEY", type="password")
    with col2:
        api_secret = st.text_input("ALPACA API SECRET", type="password")
    if st.button("Guardar y conectar"):
        if api_key and api_secret:
            save_credentials(api_key.strip(), api_secret.strip())
            st.experimental_rerun()
        else:
            st.error("Debes ingresar ambas claves.")
else:
    api_key = creds.get("ALPACA_API_KEY")
    api_secret = creds.get("ALPACA_API_SECRET")

# Si no hay librería, mostrar instrucción
if ALPACA_LIB is None:
    st.error("Instala 'alpaca-trade-api' o 'alpaca-py' en el entorno. (requirements.txt incluido).")
    st.stop()

# Iniciar cliente Alpaca
def create_alpaca_client(api_key, api_secret):
    if ALPACA_LIB == "tradeapi":
        return tradeapi.REST(api_key, api_secret, api_version='v2')
    else:
        # alpaca-py case - using REST from data package for market data (not trading)
        return REST(api_key, api_secret)

client = create_alpaca_client(api_key, api_secret)

# ---------------------------
# Assets: obtener activos "24/7" (crypto) y permitir lista manual
# ---------------------------
@st.cache_data(ttl=300)
def fetch_crypto_assets():
    """Obtiene lista de activos crypto desde Alpaca y filtra activos activos/tradables."""
    assets = []
    try:
        if ALPACA_LIB == "tradeapi":
            raw = client.list_assets()
            for a in raw:
                # attribute 'class' may be asset_class in other libs
                cls = getattr(a, "asset_class", None) or getattr(a, "class", None) or ""
                status = getattr(a, "status", None)
                symbol = getattr(a, "symbol", None)
                if cls and "crypto" in str(cls).lower() and status == "active":
                    assets.append(symbol)
        else:
            raw = client.get_all_assets()
            for a in raw:
                if a.asset_class and a.asset_class.lower() == "crypto" and a.status == "active":
                    assets.append(a.symbol)
    except Exception as e:
        st.warning(f"No se pudo obtener lista desde Alpaca: {e}")
    # fallback: lista pequeña común
    if not assets:
        assets = ["BTCUSD", "ETHUSD", "SOLUSD"]
    return sorted(list(set(assets)))

crypto_assets = fetch_crypto_assets()

st.sidebar.header("Configuración")
manual_add = st.sidebar.text_input("Añadir símbolo (ej: EURUSD, GBPUSD, AAPL) — separado por comas")
refresh_secs = st.sidebar.number_input("Refrescar cada (segundos)", min_value=5, max_value=300, value=10, step=5)
display_limit = st.sidebar.number_input("Máx activos en dash", min_value=10, max_value=500, value=100, step=10)

manual_list = [s.strip().upper() for s in manual_add.split(",") if s.strip()]
all_symbols = crypto_assets + manual_list
all_symbols = sorted(list(dict.fromkeys(all_symbols)))  # unique preserve order

if len(all_symbols) == 0:
    st.error("No hay símbolos para monitorear. Añade símbolos manuales o verifica tu cuenta Alpaca.")
    st.stop()

# ---------------------------
# Funciones técnicas: indicadores y heurística
# ---------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def heuristic_score(close_series):
    """Devuelve dirección, probabilidad (1-99) y magnitud diff para uso en duración."""
    if len(close_series) < 3:
        return {"direction":"N/A","prob":50,"diff":0,"rsi":50}
    short = ema(close_series, span=8).iloc[-1]
    long = ema(close_series, span=21).iloc[-1]
    diff = (short - long) / (long + 1e-9)
    r = float(rsi(close_series).iloc[-1])
    # base probability from diff
    base = 50 + diff * 2000  # escala ajustable
    # combine with RSI bias (r centered at 50)
    prob = 0.7 * base + 0.3 * (100 - abs(50 - r))
    prob = max(1, min(99, prob))
    dirc = "Subiendo" if diff > 0 else "Bajando"
    return {"direction": dirc, "prob": int(prob), "diff": diff, "rsi": int(r)}

def estimate_duration_minutes(diff_abs):
    """Traduce magnitud diff a minutos estimados (heurística)."""
    m = int(max(0, min(60*24*7, diff_abs * 2000)))  # to cap, 1 week max for safety
    # make sure small diffs still give small minutes
    if m == 0:
        m = 1
    return m

# ---------------------------
# Lectura de velas (adaptable a alpaca-trade-api o alpaca-py)
# ---------------------------
def fetch_bars(symbol, timeframe="1Min", limit=200):
    """
    timeframe: "1Min","5Min","15Min","1H","1D"
    limit: cuantas velas traer
    """
    try:
        # map to alpaca format
        if ALPACA_LIB == "tradeapi":
            # alpaca-trade-api get_barset uses '1Min' as '1Min' etc via TimeFrame not required here - use get_barset
            tf_map = {"1Min":"1Min","5Min":"5Min","15Min":"15Min","1H":"1Hour","1D":"1Day"}
            # get_barset expects timeframe in minutes or string depending; we'll use get_bars via data API if exists
            # Attempt using client.get_bars if present
            try:
                bars = client.get_bars(symbol, timeframe=timeframe, limit=limit)
                # bars is list-like or DataFrame depending on version; try to coerce
                if hasattr(bars, "df"):
                    df = bars.df
                    if isinstance(df.columns, pd.MultiIndex):
                        # flatten for symbol
                        df = df.xs(symbol, axis=1, level=0)
                    df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                    df.index = pd.to_datetime(df.index)
                    return df
                else:
                    # older wrapper
                    df = pd.DataFrame([{"t":b.t,"o":b.o,"h":b.h,"l":b.l,"c":b.c,"v":b.v} for b in bars])
                    df = df.set_index(pd.to_datetime(df["t"]))
                    df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                    return df
            except Exception:
                # fallback: try get_barset
                barset = client.get_barset(symbol, timeframe.lower(), limit=limit)
                arr = barset[symbol]
                df = pd.DataFrame([{"t":b.t, "o":b.o, "h":b.h, "l":b.l, "c":b.c, "v":b.v} for b in arr])
                df = df.set_index(pd.to_datetime(df["t"]))
                df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                return df
        else:
            # alpaca-py REST client get_bars
            tf_map = {"1Min":"1Min","5Min":"5Min","15Min":"15Min","1H":"1Hour","1D":"1Day"}
            resp = client.get_bars(symbol, timeframe=timeframe, limit=limit)
            df = resp.df
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(symbol, axis=1, level=0)
            df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
            df.index = pd.to_datetime(df.index)
            return df
    except Exception as e:
        # devuelve None si hay error
        return None

# ---------------------------
# Predicción multi-horizonte y tabla resumen
# ---------------------------
time_horizons = [
    ("30s", "1Min"),   # 30s aproximado con 1Min
    ("1m", "1Min"),
    ("2m", "1Min"),
    ("5m", "5Min"),
    ("10m", "5Min"),
    ("15m", "15Min"),
    ("1h", "1H"),
    ("2h", "1H"),
    ("1d", "1D"),
    ("2d", "1D"),
]

st.sidebar.write(f"Símbolos detectados: {len(all_symbols)}")
limited_symbols = all_symbols[:int(display_limit)]

# Panel principal: tabla resumen + selección detalle
st.markdown("## Dashboard de tendencias (solo muestra: dirección, % prob, duración estimada)")
col_table, col_detail = st.columns([2,1])

# Build summary dataframe
summary_rows = []
progress = st.progress(0)
for i, sym in enumerate(limited_symbols):
    # For each horizon compute the heuristics; we will use the shortest timeframe available for core signal (1Min or 5Min)
    # Prefer 1Min if available
    try:
        bars_1m = fetch_bars(sym, timeframe="1Min", limit=120)
        # if 1Min not available, try 5Min
        if bars_1m is None or bars_1m.empty:
            bars_1m = fetch_bars(sym, timeframe="5Min", limit=120)
    except Exception:
        bars_1m = None

    if bars_1m is None or bars_1m.empty:
        # symbol sin data
        summary_rows.append({
            "symbol": sym,
            "direction": "N/D",
            "probability": 0,
            "duration_min": 0,
            "rsi": None
        })
    else:
        base = heuristic_score(bars_1m["close"])
        dur = estimate_duration_minutes(abs(base["diff"]))
        summary_rows.append({
            "symbol": sym,
            "direction": base["direction"],
            "probability": base["prob"],
            "duration_min": dur,
            "rsi": base.get("rsi", None)
        })
    progress.progress((i+1)/len(limited_symbols))

df_summary = pd.DataFrame(summary_rows)

# Color rows: verde si Subiendo, rojo si Bajando
def color_row(row):
    if row["direction"] == "Subiendo":
        return ['background-color: #e6ffed']*len(row)
    elif row["direction"] == "Bajando":
        return ['background-color: #ffecec']*len(row)
    else:
        return ['']*len(row)

with col_table:
    st.dataframe(df_summary.style.apply(color_row, axis=1), height=600)

# Detalle: elegir símbolo para ver gráfico y multi-horizonte exacto
with col_detail:
    st.markdown("### Detalle por símbolo")
    selected = st.selectbox("Elige símbolo", limited_symbols, index=0)
    st.write("Última actualización:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    if selected:
        # traer velas múltiples timeframes
        bars_1m = fetch_bars(selected, timeframe="1Min", limit=300) or pd.DataFrame()
        bars_5m = fetch_bars(selected, timeframe="5Min", limit=300) or pd.DataFrame()
        bars_15m = fetch_bars(selected, timeframe="15Min", limit=300) or pd.DataFrame()
        bars_1h = fetch_bars(selected, timeframe="1H", limit=300) or pd.DataFrame()
        bars_1d = fetch_bars(selected, timeframe="1D", limit=300) or pd.DataFrame()

        # Gráfica principal: usar 1m si está, sino 5m, etc
        main_bars = bars_1m if not bars_1m.empty else (bars_5m if not bars_5m.empty else (bars_15m if not bars_15m.empty else bars_1h))
        if main_bars is None or main_bars.empty:
            st.warning("No hay velas para este símbolo.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=main_bars.index, open=main_bars["open"], high=main_bars["high"],
                                         low=main_bars["low"], close=main_bars["close"], name=selected))
            fig.add_trace(go.Scatter(x=main_bars.index, y=ema(main_bars["close"],8), name="EMA8", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=main_bars.index, y=ema(main_bars["close"],21), name="EMA21", line=dict(width=1)))
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # calcular predicciones por cada horizonte
            st.markdown("#### Predicción por horizontes")
            ph_rows = []
            for label, tf in time_horizons:
                # map tf to available bars
                if tf == "1Min":
                    bars = bars_1m
                elif tf == "5Min":
                    bars = bars_5m
                elif tf == "15Min":
                    bars = bars_15m
                elif tf == "1H":
                    bars = bars_1h
                elif tf == "1D":
                    bars = bars_1d
                else:
                    bars = main_bars

                if bars is None or bars.empty:
                    ph_rows.append({"horizonte": label, "tendencia": "N/D", "prob": 0, "dur_min": 0})
                    continue

                h = heuristic_score(bars["close"])
                dur = estimate_duration_minutes(abs(h["diff"]))
                # Ajustar duración dentro del horizonte (ej: para 1m horizonte -> dur en minutos <= 60 por ejemplo)
                # También convertimos dur a unitades razonables según label
                if label == "30s":
                    dur_adj = max(1, min(5, int(dur/5)))
                elif label in ("1m","2m","5m","10m","15m"):
                    dur_adj = max(1, min(180, int(dur)))
                elif label in ("1h","2h"):
                    dur_adj = max(1, min(24*60, int(dur)))
                else:
                    dur_adj = max(1, min(60*24*30, int(dur)))  # días máximo 30 días
                ph_rows.append({"horizonte": label, "tendencia": h["direction"], "prob": h["prob"], "dur_min": dur_adj})

            df_ph = pd.DataFrame(ph_rows)
            st.table(df_ph)

st.write("---")
st.info("Notas: 1) Heurística EMA+RSI — es un punto de partida. 2) Para predicciones más precisas usar modelos ML/entrenamiento con datos históricos. 3) Para 30s el sistema aproxima con 1Min porque los endpoints suelen no proveer 30s.")
# Auto-refresh (simple)
st.write(f"Refrescando cada {refresh_secs} segundos. Si desplegas en Streamlit Cloud, configura el runtime apropiado para mantenerlo vivo 24/7.")
time.sleep(1)
st.experimental_rerun()
