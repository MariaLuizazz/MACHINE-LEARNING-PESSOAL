# Exploração de Dados

!!! example "Descrição da base de dados e código de exploração"

O câncer de mama é o tipo de câncer mais comum entre mulheres em todo o mundo, responsável por aproximadamente 25% de todos os casos e afetando milhões de pessoas todos os anos. Ele se desenvolve quando células da mama começam a crescer de forma descontrolada, formando tumores que podem ser identificados por exames de imagem (raios-X) ou detectados como nódulos.

O principal desafio no diagnóstico é diferenciar corretamente os tumores malignos (cancerosos) dos benignos (não cancerosos). O objetivo deste projeto é desenvolver um modelo de classificação supervisionada capaz de prever, com base em atributos numéricos das células, se um tumor é maligno ou benigno.

!!! tip "Sobre o Dataset"

Total de registros: 569 amostras

Variável alvo: diagnosis (M = maligno, B = benigno)

Número de variáveis preditoras: 30 atributos numéricos relacionados ao tamanho, textura, formato e concavidade das células


=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Exploracaodedados.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/EXERCICIOAD/Exploracaodedados.py"
    ```

# Pré-Processamento

!!! example "Explicação dos processos realizados no pré-processamento"

Antes do treinamento do modelo, foi realizado um pré-processamento para garantir a qualidade e consistência dos dados:

Remoção de colunas irrelevantes – A coluna id foi descartada, pois não contribui para o aprendizado do modelo.

Tratamento de valores ausentes – Foram encontrados valores faltantes em algumas variáveis (concavity_worst e concave points_worst). Esses valores foram preenchidos utilizando a mediana, por ser uma técnica robusta contra outliers.

Codificação de variáveis categóricas – A variável alvo diagnosis foi transformada em valores numéricos por meio de Label Encoding (M = 1, B = 0), permitindo sua utilização pelo algoritmo de aprendizado.

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Preprocessamento.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/EXERCICIOAD/Treinamentodomodelo.py"
    ```

# Divisão de Dados

!!! example "Separação em treino e teste"

O dataset foi dividido em conjuntos de treino e teste para permitir a avaliação do modelo em dados não vistos durante o treinamento. Foram utilizadas duas proporções distintas:

Etapa I: 80% treino, 20% teste

Etapa II: 70% treino, 30% teste

Essa variação foi realizada para observar como a quantidade de dados de treino impacta o desempenho do modelo.

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Divisaodedados.py"
    ``` 

# Treinamento do modelo

!!! example "Etapas do treinamento"

Na etapa de treinamento, foi utilizado o algoritmo de Árvore de Decisão, por ser um método simples, interpretável e bastante utilizado em problemas de classificação inicial.

!!! example "ETAPA I:"

Divisão: 80% treino, 20% teste

Resultado: 93% de acurácia

!!! example "ETAPA II:"

Divisão: 70% treino, 30% teste

Resultado: 90% de acurácia

Os resultados mostram que pequenas variações na divisão dos dados afetam a acurácia final, embora o desempenho geral do modelo tenha se mantido satisfatório.

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Treinamentodomodelo.py"
    ``` 


# Avaliação do Modelo Final

Após os testes iniciais, foi feita a avaliação final do modelo. O foco desta etapa foi verificar o comportamento da árvore de decisão em termos de acurácia e complexidade.

A árvore gerada inicialmente se apresentou pequena, sugerindo que o modelo poderia estar simplificando demais os padrões dos dados (underfitting). Após ajustes na proporção de dados de treino, a árvore tornou-se mais consistente, refletindo melhor as relações entre as variáveis.
O modelo alcançou 90% de acurácia no conjunto de teste, isso significa que, a cada 100 diagnósticos, 90 foram corretos.

Embora existam modelos que possam alcançar valores um pouco maiores (como 95%+), a escolha dos 90% foi intencional:

Por que 90% foi considerado adequado?
- Balanceamento entre desempenho e generalização
- Acima de 90%, o modelo começava a apresentar sinais de overfitting.

Os 90% garantem que o modelo é confiável e generaliza melhor para novos pacientes.

- Cenário do dataset

O modelo de 90% apresentou bom equilíbrio entre acerto de benignos e malignos, o que é essencial em aplicações médicas.

- Importância clínica

No contexto de câncer de mama, evitar falsos negativos (não detectar um tumor maligno) é prioridade.
O modelo de 90% não só manteve acurácia alta, como também preservou um bom recall para a classe Maligna, reduzindo o risco de diagnósticos perigosamente errados.

!!! tip "Conclusão da Avaliação"
O modelo final com 90% de acurácia foi escolhido por representar o melhor equilíbrio entre desempenho, generalização e relevância prática para o contexto médico.

!!! example "Breast Cancer Dataset"

=== "decision tree"

    ```python exec="1" html="true"
    --8<-- "docs/EXERCICIOAD/Avaliacaodomodelo.py"
    ```


=== "code"

    ```python exec="0"
    --8<-- "docs/EXERCICIOAD/Avaliacaodomodelo.py"
    ```
---

# Relatório Final

!!! example "Resumo do Projeto"

Este projeto teve como objetivo aplicar técnicas de Machine Learning para criar um modelo capaz de prever se um tumor de mama é benigno ou maligno, utilizando o dataset Breast Cancer Wisconsin (Diagnostic).

As etapas seguidas foram:

Exploração de dados - Pré-processamento - Divisão de dados – separação em treino e teste -
Treinamento do modelo - Avaliação do modelo – análise do desempenho final com base na acurácia e na estrutura da árvore.


!!! success "Resultados Obtidos"
Acurácia variando entre 90% e 93%, dependendo da proporção de treino/teste utilizada.


!!! tip "Conclusão"
Mesmo com limitações, o projeto cumpriu seu objetivo: desenvolver um modelo de classificação supervisionada e aplicar todo o fluxo de pré-processamento, treino e avaliação, consolidando o  meu aprendizado sobre o processo de Machine Learning.



