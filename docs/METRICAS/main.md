# Exploração de Dados

!!! example "Descrição da base de dados e código de exploração"

O câncer de mama é o tipo de câncer mais comum entre mulheres em todo o mundo, responsável por aproximadamente 25% de todos os casos e afetando milhões de pessoas todos os anos. Ele se desenvolve quando células da mama começam a crescer de forma descontrolada, formando tumores que podem ser identificados por exames de imagem (raios-X) ou detectados como nódulos.

O principal desafio no diagnóstico é diferenciar corretamente os tumores malignos (cancerosos) dos benignos (não cancerosos). O objetivo deste projeto é desenvolver um modelo de classificação supervisionada capaz de prever, com base em atributos numéricos das células, se um tumor é maligno ou benigno, e estabelecer um diagnóstico confiável.

!!! tip "Sobre o Dataset"

Total de registros: 569 amostras

Variável alvo: diagnosis (M = maligno, B = benigno)

Número de variáveis preditoras: 30 atributos numéricos relacionados ao tamanho, textura, formato e concavidade das células.


=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/KNN/exploracaodedados.py"
    ```


# Aplicação da Técnicas

!!! example "Implementação do KNN "


=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/METRICAS/knnM.py"
    ```
=== "Code"

    ```python
    --8<-- "docs/METRICAS/knnM.py"
    ``` 



!!! example "Implementação do KMEANS "



=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/METRICAS/kmeansM.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/METRICAS/kmeansM.py"
    ``` 

    cbq
    cb

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/METRICAS/teste.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/METRICAS/teste.py"
    ``` 
