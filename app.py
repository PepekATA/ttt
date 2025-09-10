# app.py
"""
QuickTrend 24/7 — Dashboard de señales (EMA + RSI + heurística)
- Guarda credenciales localmente (credentials.json) la primera vez.
- Extrae activos 'crypto' de Alpaca (24/7) y permite añadir manualmente.
- Muestra tabla con color (verde/subida, rojo/bajada), probabilidad (%) y duración estimada.
- Gráfica interactiva por símbolo con EMAs.
- NOTA: Reemplaza las credenciales placeholder por tus claves reales.
"""

import os
import json
import time
from datetime import datetime, timedelta
import math
import concurrent.futures

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Alpaca client imports (compatibility)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_LIB = "tradeapi"
except Exception:
    try:
        from alpaca.data.rest import REST as AlpacaREST
        ALPACA_LIB = "alpaca-py"
    except Exception:
        ALPACA_LIB = None

# -----------------------
# ============== USER CONFIG (REEMPLAZA ESTAS CLAVES) ==============
# -----------------------
# PON TUS CLAVES AQUÍ PARA PRUEBAS (NO DEJES ESTAS KEYS EN REPO PÚBLICO)
ALPACA_API_KEY = "PUT_YOUR_ALPACA_API_KEY_HERE"
ALPACA_API_SECRET = "PUT_YOUR_ALPACA_API_SECRET_HERE"
# Base URL (paper/live)
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # dejar así para paper trading

# Archivo donde guardamos credenciales para no pedirlas cada vez (pruebas)
CRED_FILE = "credentials.json"

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(layout="wide", page_title="QuickTrend 24/7", initial_sidebar_state="expanded")
st.title("🔮 QuickTrend 24/7 — Señales rápidas (EMA + RSI)")

# -----------------------
# Util: guardar / cargar credenciales localmente
# -----------------------
def save_credentials(key, secret, base_url=ALPACA_BASE_URL):
    data = {"ALPACA_API_KEY": key, "ALPACA_API_SECRET": secret, "ALPACA_BASE_URL": base_url}
    with open(CRED_FILE, "w") as f:
        json.dump(data, f)

def load_credentials():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# Si no hay credenciales, usamos las del archivo más arriba (usuario pidió ponerlas en el código).
creds = load_credentials()
if creds is None:
    # Si las claves inline están vacías, pedimos al usuario en UI
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        st.warning("Introduce tus claves Alpaca en la interfaz (o en el código).")
        col1, col2 = st.columns(2)
        with col1:
            input_key = st.text_input("ALPACA API KEY", type="password")
        with col2:
            input_secret = st.text_input("ALPACA API SECRET", type="password")
        if st.button("Guardar credenciales (prueba)"):
            if input_key and input_secret:
                save_credentials(input_key.strip(), input_secret.strip())
                st.experimental_rerun()
            else:
                st.error("Ambas claves son necesarias.")
    else:
        # Guardar las credenciales que están en el archivo (inline)
        save_credentials(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)
        creds = load_credentials()
else:
    # ya cargadas desde archivo
    pass

# Asegurarnos de tener credenciales luego
creds = load_credentials()
if not creds:
    st.stop()

API_KEY = creds.get("ALPACA_API_KEY")
API_SECRET = creds.get("ALPACA_API_SECRET")
BASE_URL = creds.get("ALPACA_BASE_URL", ALPACA_BASE_URL)

# -----------------------
# Crear cliente Alpaca (compatible)
# -----------------------
def create_alpaca_client(api_key, api_secret, base_url=BASE_URL):
    if ALPACA_LIB == "tradeapi":
        try:
            return tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        except Exception as e:
            st.error(f"Error creando cliente alpaca-trade-api: {e}")
            return None
    elif ALPACA_LIB == "alpaca-py":
        try:
            return AlpacaREST(api_key, api_secret, base_url=base_url)
        except Exception as e:
            st.error(f"Error creando cliente alpaca-py: {e}")
            return None
    else:
        st.error("Instala 'alpaca-trade-api' o 'alpaca-py' en el entorno.")
        return None

client = create_alpaca_client(API_KEY, API_SECRET)

if client is None:
    st.stop()

# -----------------------
# Helper: indicadores y heurística
# -----------------------
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
    """
    Heurística combinada que produce: direction, probability (1-99), diff (relative) y rsi.
    """
    if close_series is None or len(close_series) < 5:
        return {"direction":"N/D","prob":50,"diff":0.0,"rsi":50}
    # EMAs
    short = ema(close_series, span=8).iloc[-1]
    long = ema(close_series, span=21).iloc[-1]
    diff = (short - long) / (long + 1e-9)
    r = float(rsi(close_series).iloc[-1])
    # slope (momentum) via linear regression on last N
    y = np.log(close_series[-20:].values + 1e-9)
    x = np.arange(len(y))
    if len(y) >= 3:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        m = 0.0
    # volatility (recent)
    vol = np.std(np.log(close_series.pct_change().dropna()+1e-9)) if len(close_series)>5 else 0.0

    # Compose probability (heurística)
    # base from diff and slope
    base = 50 + diff * 2000 + m * 1000 - vol * 50
    # RSI proximity: push probability away from 50 if extreme
    r_bias = (50 - abs(50 - r))  # close to 50 -> reduces confidence
    prob = 0.7 * base + 0.3 * r_bias
    prob = max(1, min(99, int(round(prob))))
    direction = "Subiendo" if diff > 0 or m > 0 else "Bajando"
    return {"direction": direction, "prob": prob, "diff": diff, "rsi": int(round(r)), "slope": m, "vol": vol}

def estimate_duration_minutes(diff_abs):
    """Convierte magnitud diff a minutos estimados (heurística)."""
    minutes = int(max(1, min(60*24*30, diff_abs * 2000)))  # cap 30 días
    # hacer que pequeñas diferencias produzcan tiempos cortos
    if minutes < 1:
        minutes = 1
    return minutes

# -----------------------
# Fetch bars robusto (compatibilidad alpaca-trade-api y alpaca-py)
# -----------------------
def fetch_bars(symbol, timeframe="1Min", limit=200):
    """
    timeframe: "1Min","5Min","15Min","1H","1D"
    limit: number of bars (max)
    Returns pandas.DataFrame with columns: open, high, low, close, volume and datetime index
    """
    try:
        # alpaca-trade-api path
        if ALPACA_LIB == "tradeapi":
            # try get_bars (newer versions)
            try:
                bars = client.get_bars(symbol, timeframe=timeframe, start=None, end=None, limit=limit)
                # bars may be a Bars object with .df property
                if hasattr(bars, "df"):
                    df = bars.df
                    # if multiindex (symbol, col), flatten
                    if isinstance(df.columns, pd.MultiIndex):
                        if symbol in df.columns.levels[0]:
                            df = df.xs(symbol, axis=1, level=0)
                    df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                    df.index = pd.to_datetime(df.index)
                    # ensure numeric
                    for col in ["open","high","low","close","volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
                else:
                    # fallback: bars is iterable
                    items = []
                    for b in bars:
                        try:
                            items.append({"t": getattr(b, "t", getattr(b, "timestamp", None)),
                                          "o": getattr(b, "o", b.o if hasattr(b,'o') else None),
                                          "h": getattr(b, "h", None),
                                          "l": getattr(b, "l", None),
                                          "c": getattr(b, "c", None),
                                          "v": getattr(b, "v", None)})
                        except Exception:
                            pass
                    if not items:
                        return None
                    df = pd.DataFrame(items).set_index(pd.to_datetime(pd.Series([it["t"] for it in items])))
                    df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                    for col in ["open","high","low","close","volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
            except Exception:
                # try legacy get_barset
                try:
                    barset = client.get_barset(symbol, timeframe.lower(), limit=limit)
                    arr = barset[symbol]
                    if not arr:
                        return None
                    df = pd.DataFrame([{"t":b.t, "o":b.o, "h":b.h, "l":b.l, "c":b.c, "v":b.v} for b in arr])
                    df = df.set_index(pd.to_datetime(df["t"]))
                    df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                    for col in ["open","high","low","close","volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
                except Exception:
                    return None
        elif ALPACA_LIB == "alpaca-py":
            try:
                bars = client.get_bars(symbol, timeframe=timeframe, start=None, end=None, limit=limit)
                df = bars.df
                if isinstance(df.columns, pd.MultiIndex):
                    if symbol in df.columns.levels[0]:
                        df = df.xs(symbol, axis=1, level=0)
                df = df.rename(columns={"c":"close","o":"open","h":"high","l":"low","v":"volume"})
                df.index = pd.to_datetime(df.index)
                for col in ["open","high","low","close","volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None

# -----------------------
# Obtener lista de activos crypto 24/7 de Alpaca
# -----------------------
@st.cache_data(ttl=300)
def fetch_crypto_assets():
    assets = []
    try:
        if ALPACA_LIB == "tradeapi":
            raw = client.list_assets()
            for a in raw:
                cls = getattr(a, "asset_class", None) or getattr(a, "class", None) or ""
                status = getattr(a, "status", None)
                symbol = getattr(a, "symbol", None)
                if symbol and cls and "crypto" in str(cls).lower() and status == "active":
                    assets.append(symbol)
        else:
            raw = client.get_all_assets()
            for a in raw:
                if getattr(a, "asset_class", "").lower() == "crypto" and getattr(a, "status","")=="active":
                    assets.append(a.symbol)
    except Exception as e:
        # en caso de error devolvemos una lista por defecto útil
        st.warning(f"No se pudo listar activos desde Alpaca: {e}")
    # fallback
    if not assets:
        assets = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "DOGEUSD"]
    return sorted(list(set(assets)))

# -----------------------
# Sidebar: configuraciones
# -----------------------
st.sidebar.header("Configuración")
st.sidebar.markdown("Ajusta refresco, cantidad de símbolos mostrados y añade símbolos manuales.")

refresh_secs = int(st.sidebar.number_input("Refrescar cada (segundos)", min_value=5, max_value=300, value=12, step=1))
display_limit = int(st.sidebar.number_input("Máx símbolos en dashboard", min_value=20, max_value=1000, value=200, step=10))
manual_input = st.sidebar.text_input("Símbolos manuales (coma-separados, ej: EURUSD,GBPUSD)")
show_only_active_24_7 = st.sidebar.checkbox("Sólo activos 24/7 (crypto)", value=True)

# Obtener lista
crypto_symbols = fetch_crypto_assets()  # lista grande
manual_list = [s.strip().upper() for s in manual_input.split(",") if s.strip()]
symbols = (crypto_symbols + manual_list) if not manual_list==[] else crypto_symbols
# Deduplicate and limit
symbols = list(dict.fromkeys(symbols))  # preserve order unique
symbols = symbols[:display_limit]

# Botón para forzar refresh inmediato
if st.sidebar.button("Refrescar ahora"):
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Seguridad: si vas a dejar las claves en producción usa Streamlit Secrets o variables de entorno.")
st.sidebar.write(f"Activos detectados: {len(crypto_symbols)} (crypto). Mostrando {len(symbols)} símbolos.")

# -----------------------
# Función para procesar un símbolo -> retorna summary dict
# -----------------------
def process_symbol(sym):
    """Fetch bars (1Min preferred) and compute heuristic summary. Maneja errores."""
    try:
        bars = fetch_bars(sym, timeframe="1Min", limit=150)
        if bars is None or bars.empty:
            # intentar 5Min
            bars = fetch_bars(sym, timeframe="5Min", limit=120)
        if bars is None or bars.empty:
            return {"symbol": sym, "direction": "N/D", "probability": 0, "duration_min": 0, "rsi": None, "has_data": False}
        hs = heuristic_score(bars["close"])
        dur = estimate_duration_minutes(abs(hs["diff"]))
        return {"symbol": sym, "direction": hs["direction"], "probability": hs["prob"], "duration_min": dur, "rsi": hs.get("rsi", None), "has_data": True}
    except Exception as e:
        return {"symbol": sym, "direction": "Err", "probability": 0, "duration_min": 0, "rsi": None, "has_data": False, "error": str(e)}

# -----------------------
# Ejecutar procesamiento concurrente para muchos símbolos (mejor rendimiento)
# -----------------------
st.markdown("## Resumen de señales (tabla interactiva)")
progress_text = st.empty()
progress_bar = st.progress(0)

results = []
# Limit batch size to avoid rate limit; procesar en hilos pequeños
batch = symbols
total = len(batch)
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_symbol, s): s for s in batch}
    completed = 0
    for fut in concurrent.futures.as_completed(futures):
        sym = futures[fut]
        try:
            res = fut.result()
        except Exception as e:
            res = {"symbol": sym, "direction": "Err", "probability": 0, "duration_min": 0, "rsi": None, "has_data": False, "error": str(e)}
        results.append(res)
        completed += 1
        progress_text.text(f"Procesados {completed}/{total} símbolos")
        progress_bar.progress(completed/total)

# Construir DataFrame ordenado por probabilidad desc (o por symbol)
df = pd.DataFrame(results)
# Ordenar por probabilidad descendente para resaltar señales más fuertes
df_sorted = df.sort_values(by="probability", ascending=False).reset_index(drop=True)

# Color styling: verde si Subiendo, rojo si Bajando, gris si N/D
def color_style(row):
    if row["direction"] == "Subiendo":
        return ["background-color: #eaffea"] * len(row)
    elif row["direction"] == "Bajando":
        return ["background-color: #fff0f0"] * len(row)
    elif row["direction"] == "Err":
        return ["background-color: #fff4e6"] * len(row)
    else:
        return [""] * len(row)

# Mostrar tabla con colores y columnas importantes
st.write("### Tabla de señales")
# Reorder columns for clarity
cols_order = ["symbol", "direction", "probability", "duration_min", "rsi", "has_data"]
present_df = df_sorted[[c for c in cols_order if c in df_sorted.columns]]
present_df = present_df.rename(columns={"duration_min":"duración (min)", "probability":"prob (%)", "has_data":"datos"})

# Use st.dataframe with styled HTML via pandas style -> convert to HTML then st.write may display but st.dataframe is interactive
st.dataframe(present_df.style.apply(color_style, axis=1), height=600)

# -----------------------
# Selección de símbolo para detalle y gráfico
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Detalle gráfico")
selected_symbol = st.sidebar.selectbox("Selecciona símbolo para ver detalle", options=present_df["symbol"].tolist(), index=0)

if selected_symbol:
    st.markdown(f"## Detalle: {selected_symbol}")
    # Traer multiple TFs
    bars_1m = fetch_bars(selected_symbol, timeframe="1Min", limit=300) or pd.DataFrame()
    bars_5m = fetch_bars(selected_symbol, timeframe="5Min", limit=300) or pd.DataFrame()
    bars_15m = fetch_bars(selected_symbol, timeframe="15Min", limit=300) or pd.DataFrame()
    bars_1h = fetch_bars(selected_symbol, timeframe="1H", limit=500) or pd.DataFrame()
    bars_1d = fetch_bars(selected_symbol, timeframe="1D", limit=500) or pd.DataFrame()

    # Decide which to use for main display (prefer 1m)
    main_bars = bars_1m if not bars_1m.empty else (bars_5m if not bars_5m.empty else (bars_15m if not bars_15m.empty else (bars_1h if not bars_1h.empty else bars_1d)))
    if main_bars is None or main_bars.empty:
        st.warning("No hay datos para este símbolo.")
    else:
        # Calculate indicators
        main_bars["EMA8"] = ema(main_bars["close"], 8)
        main_bars["EMA21"] = ema(main_bars["close"], 21)
        main_bars["RSI"] = rsi(main_bars["close"])

        # Heuristic for the selected symbol
        hs = heuristic_score(main_bars["close"])
        dur = estimate_duration_minutes(abs(hs["diff"]))

        # Layout: graph + side info
        col1, col2 = st.columns([3,1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=main_bars.index, open=main_bars["open"], high=main_bars["high"],
                                         low=main_bars["low"], close=main_bars["close"], name=selected_symbol))
            fig.add_trace(go.Scatter(x=main_bars.index, y=main_bars["EMA8"], name="EMA8", line=dict(width=1.2)))
            fig.add_trace(go.Scatter(x=main_bars.index, y=main_bars["EMA21"], name="EMA21", line=dict(width=1.2)))
            # Color background depending on direction
            if hs["direction"] == "Subiendo":
                fig.update_layout(plot_bgcolor="rgba(233, 255, 235, 0.4)")
            else:
                fig.update_layout(plot_bgcolor="rgba(255, 230, 230, 0.4)")
            fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Card-like info
            st.markdown("### Señal rápida")
            trend_color = "green" if hs["direction"] == "Subiendo" else "red"
            st.markdown(f"**Tendencia:** <span style='color:{trend_color};font-weight:700'>{hs['direction']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Probabilidad:** {hs['prob']}%")
            st.markdown(f"**Duración estimada:** {dur} minutos")
            st.markdown(f"**RSI (últ):** {hs.get('rsi', 'N/A')}")
            st.markdown(f"**Slope:** {hs.get('slope', 0.0):.6f}")
            st.markdown(f"**Vol:** {hs.get('vol', 0.0):.6f}")
            st.markdown("---")
            st.markdown("Horizontes aproximados (30s,1m,2m,5m,10m,15m,1h,2h,1d)")
            # compute multi-horizon quick table
            horizons = [("30s","1Min"),("1m","1Min"),("2m","1Min"),("5m","5Min"),("10m","5Min"),("15m","15Min"),("1h","1H"),("2h","1H"),("1d","1D")]
            rows = []
            for label, tf in horizons:
                bars = fetch_bars(selected_symbol, timeframe=tf, limit=200)
                if bars is None or bars.empty:
                    rows.append([label, "N/D", 0, 0])
                else:
                    h = heuristic_score(bars["close"])
                    dur_h = estimate_duration_minutes(abs(h["diff"]))
                    rows.append([label, h["direction"], h["prob"], dur_h])
            df_h = pd.DataFrame(rows, columns=["horizonte","tendencia","prob(%)","dur_min"])
            st.table(df_h)

# -----------------------
# Footer + auto refresh
# -----------------------
st.markdown("---")
st.write(f"Última actualización: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.write("Refrescando cada", refresh_secs, "segundos. Si despliegas en hosting con límites, ajusta el refresh para evitar rate limits.")
time.sleep(0.5)
# reload page after refresh_secs
st.experimental_rerun()
