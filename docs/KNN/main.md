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


# Pré processamento

!!! example "Explicação dos processos realizados no pré-processamento"

Antes do treinamento do modelo, foi realizado um pré-processamento para garantir a qualidade e consistência dos dados:

Remoção de colunas irrelevantes – A coluna id foi descartada, pois não contribui para o aprendizado do modelo.

Tratamento de valores ausentes – Foram encontrados valores faltantes em algumas variáveis (concavity_worst e concave points_worst). Esses valores foram preenchidos utilizando a mediana.

Codificação de variáveis categóricas – A variável alvo diagnosis foi transformada em valores numéricos por meio de Label Encoding (M = 1, B = 0), permitindo sua utilização pelo algoritmo de aprendizado.

!!! example "Todos os processos de pré-processamento feitos no algoritmo de árvore de decisão foram feitos também no algoritmo de KNN."


=== "Code"

    ```python exec="0"
    --8<-- "docs/KNN/prepro.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/KNN/prepro.py"
    ```


# Divisão dos dados, Treinamento do Modelo e Avaliação do Modelo.

!!! sucess "O dataset foi dividido em conjuntos de treino e teste com uma proporção de 80% treino e 20% teste."

O questionamento principal foi: Quais features são mais relevantes para o diagnóstico de câncer de mama de acordo com o que foi fornecido no meu dataset?

Ao analisar o dataset, foi bom lembrar que trata-se da previsão de um diagnóstico, é preciso entender a base e o que eu quero prever. Em uma pesquisa rápida para entender melhor, concluí que: para a escolha das minhas features eu deveria ficar atenta às minhas variáveis mais relevantes.

Se o nódulo é redondo, pequeno e com bordas suaves → mais provável benigno.
Se o nódulo é grande, irregular, com bordas cheias de reentrâncias → mais provável maligno.
Nódulos malignos costumam ser maiores, com contornos irregulares e não lisos, enquanto benignos tendem a ser mais arredondados e bem delimitados.

Portanto, as variáveis mais importantes do meu dataset para o diagnóstico seriam aquelas que especificam tamanho, formato e textura do nódulo. No caso, as variáveis escolhidas foram: texture_mean e radius_mean.

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/KNN/knn.py"
    ```

=== "Code"

    ```python exec="0"
    --8<-- "docs/KNN/knn.py"
    ```



# Relatorio final

- O modelo KNN com k=3 memoriza os dados de treino e usa distância para fazer previsões. Para cada novo tumor, ele encontra os 3 tumores mais similares no conjunto de treino e decide pela maioria.

- Sobre a Avaliação:
Após a etapa de treino e teste, o processo entregou uma acurácia de 86% que nos mostra que o modelo acerta 86 em cada 100 previsões. 

- Sobre a Visualização:
A fronteira de decisão mostra como o modelo separa tumores benignos de malignos. Áreas coloridas mostram onde o modelo prevê cada classe. Observando o modelo e a acurácia o modelo apresenta sinais de overfitting.




