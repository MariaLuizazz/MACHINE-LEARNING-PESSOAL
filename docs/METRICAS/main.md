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


1. Preparação dos Dados

Os dados foram pré-processados para garantir melhor qualidade nas previsões. As principais etapas incluíram:

Normalização das variáveis numéricas.

Aplicação do SMOTE (Synthetic Minority Oversampling Technique) para balanceamento das classes.

Utilização de PCA (Principal Component Analysis) para redução de dimensionalidade, garantindo menor complexidade e melhor desempenho computacional.

2. Treinamento dos Modelos

KNN (K-Nearest Neighbors): Algoritmo supervisionado baseado na proximidade dos vizinhos mais próximos.

KMeans (K-Means Clustering): Algoritmo não supervisionado de agrupamento, adaptado para a tarefa de classificação.

# Aplicação da Técnicas

Este projeto tem como objetivo avaliar e comparar o desempenho de dois algoritmos de Machine Learning – KNN (K-Nearest Neighbors) e KMeans (K-Means Clustering) – aplicados a um problema de classificação binária. A análise foi conduzida com foco em métricas de desempenho e matrizes de confusão, de forma a compreender vantagens e limitações de cada abordagem.

!!! example "Implementação do KNN "


=== "Resultado"

    ```python exec="on" html="1"
    --8<-- "docs/METRICAS/knnM.py"
    ```
=== "Code"

    ```python
    --8<-- "docs/METRICAS/knnM.py"
    ``` 



!!! example "Implementação do KMEANS "



=== "Resultado"

    ```python exec="on" html="1"
    --8<-- "docs/METRICAS/kmeansM.py"
    ```

=== "Code"

    ```python
    --8<-- "docs/METRICAS/kmeansM.py"
    ``` 





# Matrizes de Confusão

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/METRICAS/teste.py"
    ```



# Avaliação dos Modelos

=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/METRICAS/avaliacao.py"
    ```
=== "Code"

    ```python
    --8<-- "docs/METRICAS/avaliacao.py"
    ``` 


# Comparação dos Resultados

A avaliação dos dois modelos (KNN e KMeans) permitiu observar diferenças importantes em termos de desempenho, destacando pontos fortes e limitações de cada abordagem.

O modelo KNN apresentou uma acurácia de 91,2%, com métricas balanceadas entre precisão (0,877), recall (0,891) e F1-score (0,884). Isso indica que o algoritmo teve um bom equilíbrio entre identificar corretamente os casos benignos e malignos, sendo mais consistente no tratamento das duas classes. No entanto, sua precisão foi ligeiramente inferior, o que significa que, entre os casos previstos como malignos, houve uma proporção maior de falsos positivos em comparação ao KMeans.

Já o modelo KMeans, por se tratar de um algoritmo de aprendizado não supervisionado, surpreendeu ao alcançar uma acurácia de 90,2%, próxima à do KNN. Seu destaque foi a alta precisão (0,981), ou seja, quase todas as amostras classificadas como malignas realmente pertenciam a essa classe. Porém, essa alta precisão veio acompanhada de uma queda no recall (0,750), mostrando que o KMeans deixou de identificar corretamente uma parcela considerável dos casos malignos, classificando-os como benignos. Isso é crítico em contextos sensíveis, como o diagnóstico médico, em que falsos negativos podem trazer riscos significativos.

- De forma geral, pode-se afirmar que:

O KNN é mais equilibrado e confiável para cenários em que tanto falsos positivos quanto falsos negativos precisam ser controlados.

O KMeans é vantajoso em termos de precisão, mas pode não ser o mais indicado em situações em que a detecção completa dos casos positivos (alto recall) é essencial.

Portanto, a escolha entre os dois modelos dependerá diretamente do contexto de aplicação: se o objetivo for minimizar falsos negativos, o KNN se mostra mais adequado; se o objetivo for garantir maior segurança nas classificações positivas, o KMeans pode ser preferido.