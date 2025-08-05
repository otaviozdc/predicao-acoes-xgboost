import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np

# Importa as configurações e as funções auxiliares
from config import TICKER, PERIOD, FEATURES, PARAM_GRID, RANDOM_STATE
from utils import calcular_features_tecnicas

# PREPARAÇÃO DOS DADOS
print("Baixando dados e calculando features...")
dados_brutos = yf.download(TICKER, period=PERIOD)

dados_com_features = calcular_features_tecnicas(dados_brutos.copy())
dados_com_features["target"] = (dados_com_features["Close"].shift(-1) > dados_com_features["Close"]).astype(int)

dados_limpos = dados_com_features.dropna()
X = dados_limpos[FEATURES]
y = dados_limpos["target"] 

# CONFIGURAÇÃO DO MODELO E BUSCA DE HIPERPARÂMETROS
tscv = TimeSeriesSplit(n_splits=5)
class_counts = y.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Calculando peso para a classe minoritária (alta): {scale_pos_weight:.2f}")

modelo_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    scale_pos_weight=scale_pos_weight
)

grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=PARAM_GRID,
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    scoring="f1"
)

print("Iniciando a busca pelos melhores hiperparâmetros para XGBoost...")
grid_search.fit(X, y)

print("\nMelhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

melhor_modelo = grid_search.best_estimator_

# AVALIAÇÃO FINAL DO MODELO
tamanho_treino = int(len(dados_limpos) * 0.8)
X_train = X[:tamanho_treino]
X_test = X[tamanho_treino:]
y_train = y[:tamanho_treino]
y_test = y[tamanho_treino:]

melhor_modelo.fit(X_train, y_train)
previsoes = melhor_modelo.predict(X_test)

print(f"\nAcurácia do modelo no conjunto de teste: {accuracy_score(y_test, previsoes):.2f}")
print("\nMatriz de Confusão do modelo:")
print(confusion_matrix(y_test, previsoes))
print("\nRelatório de Classificação do modelo:")
print(classification_report(y_test, previsoes))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, previsoes), display_labels=["Baixa", "Alta"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")

# PREVISÃO PARA AMANHÃ
X_hoje = dados_com_features[FEATURES].dropna().iloc[-1:].copy()

if X_hoje.empty or X_hoje.isnull().values.any():
    print("\nErro: Não foi possível obter dados válidos para previsão.")
    print("Verifique se os dados mais recentes do Yahoo Finance estão disponíveis.")
    print("\nÚltima linha disponível (com ou sem NaN):")
    print(dados_com_features[FEATURES].iloc[-1:])
else:
    previsao_amanha = melhor_modelo.predict(X_hoje)
    previsao_prob = melhor_modelo.predict_proba(X_hoje)

    print("\n--- Previsão para Amanhã ---")
    if previsao_amanha[0] == 1:
        print("📈 O modelo prevê que o preço de fechamento de amanhã será MAIOR que o de hoje.")
        print("Previsão: ALTA (Comprar)")
    elif previsao_amanha[0] == 0:
        print("📉 O modelo prevê que o preço de fechamento de amanhã será MENOR ou IGUAL ao de hoje.")
        print("Previsão: BAIXA (Não comprar)")

    print(f"Probabilidade da previsão (Baixa, Alta): {np.round(previsao_prob[0], 3)}")
    print("----------------------------")

plt.show()