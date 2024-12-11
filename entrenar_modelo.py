import os
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Parámetros de los indicadores
EMA_PERIOD = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

# Archivo con datos históricos
HISTORICAL_CSV = "historico.csv"
MODEL_FILE = "modelo_prob_subida.pkl"

def cargar_datos(csv_path):
    """
    Carga el CSV con columnas: timestamp, open, high, low, close, volume.
    timestamp en milisegundos. Ordena por fecha.
    """
    if not os.path.exists(csv_path):
        logging.error(f"No se encontró el archivo {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if not {"timestamp","open","high","low","close","volume"}.issubset(df.columns):
        logging.error("El CSV no contiene las columnas requeridas: timestamp, open, high, low, close, volume")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df

def preparar_dataset(df):
    """
    Calcula los indicadores técnicos y crea la etiqueta future_up.
    future_up = 1 si la vela siguiente cierra arriba, 0 en caso contrario.
    """

    # Calculamos indicadores
    df["ema200"] = ta.ema(df["close"], length=EMA_PERIOD)
    df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)

    # Etiqueta: future_up
    df["future_close"] = df["close"].shift(-1)
    df["future_up"] = (df["future_close"] > df["close"]).astype(int)

    # Eliminamos filas con NaN
    df.dropna(subset=["ema200","rsi","macd","macd_signal","atr","volume","future_up"], inplace=True)

    return df

def entrenar_modelo(df):
    """
    Entrena un modelo usando Random Forest y RandomizedSearchCV, 
    lo evalúa y lo guarda en disco.
    """
    features = ["ema200", "rsi", "macd", "macd_signal", "atr", "volume"]
    X = df[features].values
    y = df["future_up"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Pipeline: escalado + RandomForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    # Búsqueda aleatoria de hiperparámetros
    param_distributions = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5]
    }

    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_distributions, 
        n_iter=10, 
        scoring="accuracy", 
        cv=3, 
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    logging.info("Iniciando búsqueda de hiperparámetros...")
    search.fit(X_train, y_train)
    logging.info(f"Mejores parámetros: {search.best_params_}")
    logging.info(f"Mejor score CV: {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Evaluar en test
    y_pred = best_model.predict(X_test)
    logging.info("Reporte de clasificación en el set de prueba:")
    logging.info("\n" + classification_report(y_test, y_pred))

    # Guardar modelo
    dump(best_model, MODEL_FILE)
    logging.info(f"Modelo guardado en {MODEL_FILE}")

    return best_model

def main():
    logging.info("Cargando datos...")
    df = cargar_datos(HISTORICAL_CSV)
    if df is None:
        return

    logging.info("Preparando dataset...")
    df = preparar_dataset(df)
    if len(df) < 500:  
        # Un mínimo arbitrario. Deberías tener más datos.
        logging.warning("Pocos datos para entrenar el modelo, podrían no ser representativos.")
    
    logging.info("Entrenando modelo...")
    model = entrenar_modelo(df)
    logging.info("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
