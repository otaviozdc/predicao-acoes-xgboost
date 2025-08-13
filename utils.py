import pandas as pd

def calcular_features_tecnicas(dados: pd.DataFrame) -> pd.DataFrame:
    # Médias Móveis
    dados["MA_20"] = dados["Close"].rolling(window=20).mean()
    dados["MA_50"] = dados["Close"].rolling(window=50).mean()

    # RSI (Índice de Força Relativa)
    delta = dados["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    dados["RSI"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence) e linha de sinal
    exp12 = dados["Close"].ewm(span=12, adjust=False).mean()
    exp26 = dados["Close"].ewm(span=26, adjust=False).mean()
    dados["MACD"] = exp12 - exp26
    dados["Signal_Line"] = dados["MACD"].ewm(span=9, adjust=False).mean()

    # Bandas de Bollinger
    window = 20
    num_std = 2
    dados["SMA"] = dados["Close"].rolling(window=window).mean()
    dados["STD"] = dados["Close"].rolling(window=window).std()
    dados["Banda_Sup"] = dados["SMA"] + (dados["STD"] * num_std)
    dados["Banda_Inf"] = dados["SMA"] - (dados["STD"] * num_std)

    return dados
