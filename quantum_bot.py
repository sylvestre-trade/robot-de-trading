import time
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot

# Infos Telegram
TELEGRAM_TOKEN = "8042921740:AAF-vcdq8WF_k9DT0e8ExeeA302_RFyaH0E"
TELEGRAM_CHAT_ID = "8041749003"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
SUCCESS_THRESHOLD = 0.8

bot = Bot(token=TELEGRAM_TOKEN)
models = {pair: RandomForestClassifier(n_estimators=150) for pair in PAIRS}
history = {pair: [] for pair in PAIRS}


def fetch_data(pair):
    symbol_mapping = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X"
    }
    symbol = symbol_mapping[pair]

    # Récupération des données
    data = yf.download(symbol, period="1d", interval="1m")
    data = data.reset_index()

    # Renommage des colonnes
    data = data.rename(columns={
        "Datetime": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    })
    return data[["time", "open", "high", "low", "close"]]


# ... [Les autres fonctions restent identiques à votre code initial] ...

def feature_engineering(df):
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma14"] = df["close"].rolling(14).mean()
    df["rsi14"] = compute_rsi(df["close"], 14)
    df["body"] = df["close"] - df["open"]
    df = df.dropna()
    return df


def compute_rsi(series, period: int):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def label_candle(df):
    df["color"] = np.where(df["close"].shift(-1) > df["open"].shift(-1), "verte", "rouge")
    df = df.dropna()
    return df


def send_signal(pair, pred_color, proba, accuracy):
    message = (
        f"🕗 {pd.Timestamp.now().strftime('%H:%M')}\n"
        f"Paire: {pair}\n"
        f"Couleur prédite prochaine bougie: *{pred_color.upper()}*\n"
        f"Taux confiance du modèle: {proba:.0%}\n"
        f"Précision IA glissante: {accuracy:.0%}"
    )
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')


def update_history(pair, prediction, real):
    history[pair].append(prediction == real)
    if len(history[pair]) > 100:
        history[pair] = history[pair][-100:]
    return sum(history[pair]) / len(history[pair]) if history[pair] else 0


while True:
    for pair in PAIRS:
        try:
            df = fetch_data(pair)

            if len(df) < 30:
                print(f"Données insuffisantes pour {pair} ({len(df)} bougies)")
                continue

            df = feature_engineering(df)
            df = label_candle(df)

            if len(df) < 1:
                print(f"Données insuffisantes après traitement pour {pair}")
                continue

            X = df[["ma7", "ma14", "rsi14", "body"]][:-1]
            y = df["color"][:-1]

            if len(X) < 1 or len(y) < 1:
                print(f"Données d'entraînement insuffisantes pour {pair}")
                continue

            X_pred = df[["ma7", "ma14", "rsi14", "body"]].tail(1)

            models[pair].fit(X, y)
            pred = models[pair].predict(X_pred)
            proba = max(models[pair].predict_proba(X_pred)[0])

            true_color = df["color"].iloc[-1]
            acc = update_history(pair, pred[0], true_color)

            if proba >= SUCCESS_THRESHOLD and acc >= SUCCESS_THRESHOLD:
                send_signal(pair, pred[0], proba, acc)
            else:
                print(f"Signal pas envoyé ({pair}) | Précision: {acc:.0%} | Confiance: {proba:.0%}")

        except Exception as e:
            print(f"Erreur {pair}: {str(e)}")

    next_minute = (int(time.time()) // 60 + 1) * 60
    sleep_time = next_minute - time.time()
    if sleep_time > 0:
        time.sleep(sleep_time)