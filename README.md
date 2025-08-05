# Modelo de Predição de Ações com XGBoost

Um projeto de Machine Learning em Python para prever o movimento de preços da ação **PETR4.SA**, utilizando indicadores técnicos e o algoritmo XGBoost.

---

### ⚠️ Aviso Importante ⚠️

**Este projeto é para fins exclusivamente educacionais e demonstrativos. Ele não deve ser usado como consultoria financeira ou base para decisões de investimento. O mercado de ações é altamente volátil e imprevisível, e o desempenho passado de um modelo não garante resultados futuros. O uso deste código para fins de investimento é de inteira responsabilidade do usuário.**

---

### Metodologia

Este modelo de classificação binária foi construído com as seguintes etapas:

* **Fonte de Dados:** Dados históricos da ação `PETR4.SA` (Petrobras) foram baixados diretamente do Yahoo Finance usando a biblioteca `yfinance`.
* **Engenharia de Features:** Diversos indicadores técnicos foram calculados e utilizados como features para o modelo, incluindo:
    * Médias Móveis (MA_20, MA_50)
    * Índice de Força Relativa (RSI)
    * MACD e Linha de Sinal
    * Bandas de Bollinger
    * Volume e Preço de Fechamento (`Close`)
* **Variável-Alvo (`target`):** O modelo tenta prever se o preço de fechamento de amanhã será maior (classe `1`) ou menor/igual (classe `0`) que o de hoje.
* **Modelo:** Um `XGBoost Classifier` foi escolhido por ser um modelo robusto e de alto desempenho para problemas de classificação.
    * O problema de desbalanceamento de classes foi tratado usando o parâmetro `scale_pos_weight`.
* **Otimização de Hiperparâmetros:** Os melhores parâmetros para o modelo foram encontrados usando `GridSearchCV` com uma validação cruzada específica para séries temporais (`TimeSeriesSplit`), garantindo que a validação ocorresse em ordem cronológica.

### Estrutura do Projeto

O projeto está organizado em três arquivos para melhor organização e escalabilidade:

* `main.py`: O script principal que orquestra todo o processo, desde o download dos dados até a previsão final.
* `config.py`: Contém todas as variáveis de configuração, como o ticker da ação, a lista de features e o grid de hiperparâmetros.
* `utils.py`: Armazena a função auxiliar para o cálculo dos indicadores técnicos.

### Como Usar

Para rodar o projeto localmente, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/otaviozdc/predicao-acoes-xgboost.git](https://github.com/otaviozdc/predicao-acoes-xgboost.git)
    cd predicao-acoes-xgboost
    ```
2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute o script principal:**
    ```bash
    python3 main.py
    ```

O script irá baixar os dados, treinar o modelo e exibir o relatório de avaliação e a previsão para o próximo dia de negociação.

### Resultados e Limitações

O modelo alcança uma acurácia em torno de 60% no conjunto de teste. É importante notar que a confiança nas previsões pode ser baixa, como evidenciado pela probabilidade de previsão próxima de 50%.

Os resultados apresentados são específicos para a ação `PETR4.SA`. Embora o código seja genérico e possa ser aplicado a outras ações, a performance pode variar significativamente. A alta liquidez e os padrões de negociação consistentes da PETR4.SA contribuem para a eficácia dos indicadores técnicos utilizados, fazendo com que o modelo demonstre seus melhores resultados com este ativo.

### Tecnologias Utilizadas

* Python 3.x
* `yfinance`
* `pandas`
* `scikit-learn`
* `xgboost`
* `matplotlib`

**Autor:** [Otávio Zucchetti Dalla Costa](https://github.com/otaviozdc)
