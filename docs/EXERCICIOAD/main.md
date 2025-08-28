# Exploração de dados

!!! example "Explicação da base escolhida e codigo de exploração"

O câncer de mama é o câncer mais comum entre as mulheres do mundo. É responsável por 25% de todos os casos de câncer e afetou mais de 2,1 milhões de pessoas apenas em 2015. Começa quando as células da mama começam a crescer fora de controle. Essas células geralmente formam tumores que podem ser vistos via raios-X ou sentidos como nódulos na área da mama.

O principal desafio contra sua detecção é como classificar os tumores em malignos (cancerosos) ou benignos (não cancerosos), o intuito dessa entrega é criar um modelo que preveja a variavel target, classificada em tumores malignos ou benignos.


=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Exploracaodedados.py"
    ``` 

# Pré-Processamento

!!! example "Explicação dos processos realizados no pré-processamento"

Na etapa de pré-processamento, os dados do dataset de cancer de mama passaram por um processo de limpeza de dados, tratamento de valores ausentes e label encoding.
Colunas irrelevantes para o modelo foram retiradas por exemplo a coluna ['id'] , foi realizada a imputação com mediana de valores ausentes nas features concavity_worts e concavity points_worst e conversão de caracteres para números com labelEncoder na variavel target diagnostico.

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Preprocessamento.py"
    ``` 

# Divisão de Dados

!!! example "Explicação da base escolhida e codigo de exploração"

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Divisaodedados.py"
    ``` 

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/EXERCICIOAD/Divisaodedados.py"
    ```


# Treinamento do modelo

=== "Code"

    ```python
    --8<-- "docs/EXERCICIOAD/Treinamentodomodelo.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/EXERCICIOAD/Treinamentodomodelo.py"
    ```

# Avaliação do Modelo Final
ggjgvg

!!! example "Breast Cancer Dataset"

=== "decision tree"

    ```python exec="1" html="true"
    --8<-- "docs/EXERCICIOAD/Avaliacaodomodelo.py"
    ```

=== "dataset"

    ```python exec="on" html="0"
    --8<-- "docs/EXERCICIOAD/Avaliacaodomodelo.py"
    ```

=== "code"

    ```python exec="0"
    --8<-- "docs/EXERCICIOAD/Avaliacaodomodelo.py"
    ```
---

# Relatório Final


