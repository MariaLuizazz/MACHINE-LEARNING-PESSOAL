# Modelo de Classificação com Random Forest — Breast Cancer Dataset

# Exploração dos Dados

A base utilizada corresponde ao Breast Cancer Dataset, amplamente utilizado em estudos de Machine Learning para diagnóstico de câncer de mama.
Cada linha representa uma amostra de tecido mamário, e cada coluna descreve características morfológicas das células, como raio, textura, perímetro, área, concavidade e simetria.
O objetivo é prever se o diagnóstico é benigno ou maligno.

🔍 Natureza dos dados

Tipo: dados tabulares
Total de amostras: 569 registros
Variável alvo (diagnosis):
Maligno (1)
Benigno (0)

Total de atributos: 30 variáveis numéricas contínuas

📊 Análise descritiva inicial

As variáveis numéricas apresentaram médias e desvios-padrão variados, refletindo diferentes escalas de medição.
Por exemplo:

radius_mean, area_mean e perimeter_mean possuem valores mais altos e correlação entre si;

- Variáveis como concave points_mean e concavity_mean estão fortemente associadas à probabilidade de malignidade.

📈 Visualizações sugeridas

Durante a etapa de exploração, histogramas e boxplots mostraram que os tumores malignos tendem a ter valores mais altos para medidas como área, raio e concavidade, enquanto os benignos exibem distribuições menores e mais uniformes.


# Pré-processamento

O pré-processamento envolveu limpeza, codificação e tratamento de valores ausentes.

A coluna id foi removida por não conter informação relevante para o modelo.

A variável alvo diagnosis foi codificada com LabelEncoder, onde:

M → 1 (Maligno)

B → 0 (Benigno)

As variáveis com valores ausentes (concavity_mean e concave points_mean) foram imputadas com a mediana de cada respectiva coluna, garantindo consistência sem distorcer a distribuição.

Todas as features numéricas foram mantidas em sua escala original, visto que a Random Forest não é sensível a normalização ou padronização.

Resultado: base limpa, numérica e pronta para o treino do modelo.


# Divisão dos Dados

O dataset foi dividido em:

70% para treino

30% para teste

A divisão utilizou o parâmetro stratify=y, garantindo que a proporção de diagnósticos malignos e benignos fosse preservada em ambas as amostras.
O parâmetro random_state=42 assegurou a reprodutibilidade dos resultados.


# Treinamento do  modelo

O modelo implementado foi o Random Forest Classifier, um ensemble de múltiplas árvores de decisão.
A configuração utilizada foi a seguinte:

Hiperparâmetro	Valor
n_estimators	100
max_depth	5
max_features	'sqrt'
random_state	42

Essas configurações equilibram precisão e interpretabilidade, evitando sobreajuste (overfitting) e mantendo uma boa capacidade de generalização.

Durante o treinamento, cada árvore foi construída a partir de um subconjunto aleatório de dados e variáveis, característica que torna o modelo robusto e estável frente a ruídos.


# Avaliação do Modelo
🎯 Métrica principal: Acurácia

✅ Precisão obtida: 0,9708

O modelo atingiu 97,08% de acurácia na base de teste, indicando excelente desempenho na classificação entre tumores benignos e malignos.

📊 Importância das Variáveis

A análise da importância das variáveis mostrou que o modelo se baseia fortemente em características geométricas e de textura das células.
As 10 variáveis mais relevantes foram:

Posição	Feature	Importância
1	area_worst	0.171
2	concave points_mean	0.108
3	concave points_worst	0.103
4	radius_worst	0.084
5	peripheral_worst	0.082
6	peripheral_mean	0.076
7	area_mean	0.060
8	concavity_mean	0.057
9	radius_mean	0.047
10	concavity_worst	0.029

🔍 Interpretação:
As variáveis relacionadas a área e concavidade são determinantes para o diagnóstico. Tumores malignos apresentam contornos mais irregulares e áreas maiores — o que justifica o peso elevado dessas variáveis.


=== "Random forest"

    ```python exec="1" html="true"
    --8<-- "docs/RANDOMFOREST/explo.py"
    ```

=== "Resultado"

    ```python exec="1" html="true"
    --8<-- "docs/RANDOMFOREST/explo.py"
    ```
=== "Code"

    ```python
    --8<-- "docs/RANDOMFOREST/explo.py"
    ``` 

# Relatório Final e Considerações
📋 Conclusões

O modelo de Random Forest apresentou excelente desempenho, com acurácia de 97%, interpretabilidade satisfatória e estabilidade nos resultados.
A importância das variáveis reforça a coerência clínica dos dados — características morfológicas das células são realmente indicativas da natureza do tumor.

💭 Possíveis melhorias

Ajuste de hiperparâmetros com GridSearchCV para buscar a combinação ótima de profundidade e número de árvores.

Inclusão de métricas adicionais (precisão, recall, F1-score e matriz de confusão) para análise mais detalhada do desempenho.

Aplicação de validação cruzada (cross-validation) para aumentar a confiabilidade dos resultados.

Criação de uma interface visual no dashboard (via MkDocs ou Flask) exibindo as árvores e a curva ROC para fins explicativos.