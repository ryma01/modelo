import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Configuraciones
EXCHANGE_ID = "binance"
SYMBOL = "BTC/USDT"        # Ajustar el par deseado
TIMEFRAME = "4h"         # Ejemplo: velas diarias
DAYS_BACK = 365            # Cuántos días hacia atrás quieres datos
OUTPUT_CSV = "historico.csv"

# Inicializar exchange
exchange = getattr(ccxt, EXCHANGE_ID)({
    'enableRateLimit': True
})

# Convertir DAYS_BACK a milisegundos:
# 1 día = 24 * 60 * 60 * 1000 ms
now = exchange.milliseconds()
since = now - DAYS_BACK * 24 * 60 * 60 * 1000

all_ohlcv = []
limit = 500   # Límite de velas por petición (depende del exchange)
since_param = since

print(f"Descargando datos de {SYMBOL} desde {DAYS_BACK} días atrás...")

while True:
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_param, limit=limit)
    if not ohlcv:
        break
    all_ohlcv.extend(ohlcv)
    print(f"Obtenidas {len(ohlcv)} velas. Total acumulado: {len(all_ohlcv)}")
    # La última vela cargada:
    last_timestamp = ohlcv[-1][0]
    # Siguiente request desde la última vela + 1 ms
    since_param = last_timestamp + 1
    # Si la última vela es cercana a 'now', paramos
    if last_timestamp > now - (24*60*60*1000):
        break
    time.sleep(exchange.rateLimit / 1000)  # Respeto al límite de tasa del exchange

# Crear DataFrame
df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df.sort_values("timestamp", inplace=True)
df.drop_duplicates(subset="timestamp", inplace=True)

print(f"Total de velas obtenidas: {len(df)}")

# Guardar en CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Datos guardados en {OUTPUT_CSV}")
