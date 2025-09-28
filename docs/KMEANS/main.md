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






parte 2



=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/KMEANS/km.py"
    ```

=== "Code"

    ```python exec="0"
    --8<-- "docs/KMEANS/km.py"
    ```

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/KMEANS/kmeans.py"
    ```

Cada ponto azul/roxo/amarelo no gráfico é um paciente representado nessas duas dimensões condensadas.

Esses componentes carregam a maior parte da variabilidade dos dados originais (30 features)

Você aplicou PCA para reduzir a dimensionalidade dos dados.

O primeiro componente principal explica ≈44,3% da variância total dos dados.

O segundo componente principal explica ≈18,97% da variância.

Isso significa que, juntos, os dois componentes capturam ≈63,2% da informação original do dataset.