# Arquivo de configuração para variáveis do projeto

# Ticker da ação a ser analisada
TICKER = "PETR4.SA" 

# Período de dados históricos
PERIOD = "5y"

# Lista de features (indicadores técnicos e dados brutos) a serem usados pelo modelo
FEATURES = ["Close", "MA_20", "MA_50", "RSI", "Volume", "MACD", "Signal_Line", "Banda_Sup", "Banda_Inf"]

# Grid de hiperparâmetros para o GridSearchCV
PARAM_GRID = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "colsample_bytree": [0.7, 0.8]
}

# Semente para reprodutibilidade dos resultados
RANDOM_STATE = 42