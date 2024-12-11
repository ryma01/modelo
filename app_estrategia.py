import os
import logging
import ccxt
import pandas as pd
import pandas_ta as ta
from flask import Flask, render_template, request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
from sklearn.ensemble import RandomForestClassifier
import joblib

# ============================================================================
# Configuración de Logging
# ============================================================================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# ============================================================================
# Configuración del Entorno
# ============================================================================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
EMA_PERIOD = int(os.getenv("EMA_PERIOD", 200))
CLOSENESS_THRESHOLD = float(os.getenv("CLOSENESS_THRESHOLD", 0.01))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", 100_000))
RETRY_LIMIT = int(os.getenv("RETRY_LIMIT", 3))
CACHE_EXPIRATION = int(os.getenv("CACHE_EXPIRATION", 300))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
MIN_RSI = float(os.getenv("MIN_RSI", 30))
MAX_RSI = float(os.getenv("MAX_RSI", 70))
USE_MACD_FILTER = os.getenv("USE_MACD_FILTER", "true").lower() == "true"
USE_ATR_FILTER = os.getenv("USE_ATR_FILTER", "true").lower() == "true"
PROB_SUBIDA_THRESHOLD = float(os.getenv("PROB_SUBIDA_THRESHOLD", 0.7))

HISTORICAL_CSV = "historico.csv"
MODEL_FILE = "modelo_prob_subida.pkl"

# ============================================================================
# Inicialización del Exchange
# ============================================================================
exchange = getattr(ccxt, EXCHANGE_ID)({'enableRateLimit': True})

# ============================================================================
# Inicialización de la Aplicación Flask
# ============================================================================
app = Flask(__name__)

# ============================================================================
# Configuración de la Caché
# ============================================================================
cache_data = {"timestamp": None, "resultados": []}

# ============================================================================
# Cargar o Entrenar Modelo
# ============================================================================
def cargar_o_entrenar_modelo():
    if os.path.exists(MODEL_FILE):
        logging.info("Cargando modelo desde disco...")
        return joblib.load(MODEL_FILE)
    logging.error("No se encontró el modelo. Finalizando.")
    return None

model = cargar_o_entrenar_modelo()
if not model:
    exit("Modelo no disponible. Entrene el modelo antes de ejecutar la aplicación.")

# ============================================================================
# Funciones Auxiliares
# ============================================================================
def fetch_ohlcv_with_retry(symbol, timeframe, limit, retries=RETRY_LIMIT, delay=1):
    for attempt in range(1, retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            logging.warning(f"Error al obtener datos de {symbol}, intento {attempt}/{retries}: {e}")
            time.sleep(delay)
    logging.error(f"No se pudieron obtener datos de {symbol} tras {retries} reintentos.")
    return None

def obtener_pares_usdt():
    logging.debug("Cargando mercados del exchange...")
    markets = exchange.load_markets()
    return [m for m in markets.keys() if m.endswith("/USDT") and markets[m].get("active", True)]

def procesar_par(symbol):
    logging.debug(f"Obteniendo OHLCV para {symbol}")
    ohlcv = fetch_ohlcv_with_retry(symbol, timeframe=TIMEFRAME, limit=max(EMA_PERIOD + MACD_SLOW + 50, 300))
    if not ohlcv:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if len(df) < MACD_SLOW + 10:
        return None

    df["ema200"] = ta.ema(df["close"], length=EMA_PERIOD)
    df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df["macd"] = macd["MACD_12_26_9"] if macd is not None else None
    df["macd_signal"] = macd["MACDs_12_26_9"] if macd is not None else None
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)

    last_row = df.iloc[-1]
    required_cols = ["close", "ema200", "rsi", "volume", "macd", "macd_signal", "atr"]

    for col in required_cols:
        if last_row[col] is None or pd.isna(last_row[col]):
            logging.warning(f"Valor faltante para {symbol}: {col}")
            return None

    diff_ratio = abs(last_row["close"] - last_row["ema200"]) / last_row["ema200"]
    condiciones = [
        diff_ratio <= CLOSENESS_THRESHOLD,
        last_row["volume"] >= MIN_VOLUME,
        MIN_RSI < last_row["rsi"] < MAX_RSI
    ]

    if USE_MACD_FILTER:
        condiciones.append(last_row["macd"] > last_row["macd_signal"])
    if USE_ATR_FILTER:
        condiciones.append(last_row["atr"] > 0)

    if all(condiciones):
        return {
            "symbol": symbol,
            "current_price": last_row["close"],
            "ema200": last_row["ema200"],
            "diff_ratio": diff_ratio,
            "rsi": last_row["rsi"],
            "volume": last_row["volume"],
            "macd": last_row["macd"],
            "macd_signal": last_row["macd_signal"],
            "atr": last_row["atr"],
        }
    return None

def filtrar_pares_por_ema200_cercana():
    pares = obtener_pares_usdt()
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(procesar_par, par): par for par in pares}
        for future in as_completed(futures):
            result = future.result()
            if result:
                resultados.append(result)
    return resultados

# ============================================================================
# Rutas de la Aplicación Flask
# ============================================================================
@app.route("/estrategia/ema200-cercana")
def estrategia_ema200_cercana():
    resultados = filtrar_pares_por_ema200_cercana()
    for r in resultados:
        features = ["ema200", "rsi", "macd", "macd_signal", "atr", "volume"]
        X_live = [[r[f] for f in features]]
        r["prob_subida"] = model.predict_proba(X_live)[0][1] if model else None

    resultados_filtrados = [r for r in resultados if r["prob_subida"] and r["prob_subida"] >= PROB_SUBIDA_THRESHOLD]
    return render_template("resultados.html", resultados=resultados_filtrados, fecha=datetime.now())

# ============================================================================
# Ejecución de la Aplicación Flask
# ============================================================================
if __name__ == "__main__":
    logging.debug("Iniciando aplicación Flask en modo debug")
    app.run(host="0.0.0.0", port=5000, debug=True)