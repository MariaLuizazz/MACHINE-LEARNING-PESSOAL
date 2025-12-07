
# Exploração dos dados
!!! example "Descrição da base de dados e código de exploração"

O câncer de mama é o tipo de câncer mais comum entre mulheres em todo o mundo, responsável por aproximadamente 25% de todos os casos e afetando milhões de pessoas todos os anos. Ele se desenvolve quando células da mama começam a crescer de forma descontrolada, formando tumores que podem ser identificados por exames de imagem (raios-X) ou detectados como nódulos.

O principal desafio no diagnóstico é diferenciar corretamente os tumores malignos (cancerosos) dos benignos (não cancerosos). O objetivo deste projeto é desenvolver um modelo de classificação supervisionada capaz de prever, com base em atributos numéricos das células, se um tumor é maligno ou benigno.

!!! tip "Sobre o Dataset"

Total de registros: 569 amostras

Variável alvo: diagnosis (M = maligno, B = benigno)

Número de variáveis preditoras: 30 atributos numéricos relacionados ao tamanho, textura, formato e concavidade das células.


=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/decision-tree/Exploracaodedados.py"
    ```


# pré - processamento

Antes do treinamento do modelo, foi realizado um pré-processamento para garantir a qualidade e consistência dos dados:

Remoção de colunas irrelevantes – A coluna id foi descartada, pois não contribui para o aprendizado do modelo.

Tratamento de valores ausentes – Foram encontrados valores faltantes em algumas variáveis (concavity_worst e concave points_worst). Esses valores foram preenchidos utilizando a mediana, por ser uma técnica robusta contra outliers.

Codificação de variáveis categóricas – A variável alvo diagnosis foi transformada em valores numéricos por meio de Label Encoding (M = 1, B = 0), permitindo sua utilização pelo algoritmo de aprendizado.

=== "Code"

    ```python
    --8<-- "docs/decision-tree/Preprocessamento.py"
    ``` 



# Divisão de Dados e Treinamneto do Modelo(SVM + PCA)

Divisão dos dados em treino e teste

O código separa o conjunto de dados em dois grupos:
um para treinar o modelo (x_train, y_train) e outro para avaliar seu desempenho (x_test, y_test).
A divisão usa 70% dos dados para treino e 30% para teste.

A opção stratify=y garante que a proporção entre as classes (benigno/maligno) seja mantida igual nos dois conjutos.
Isso evita que o modelo treine com mais exemplos de uma classe do que outra.

O parâmetro random_state=42 apenas fixa a semente aleatória, garantindo que a mesma divisão sempre seja reproduzida.

```python
# Divisão treino/teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
```

### Aplicação do PCA somente no conjunto de treino

O PCA (Principal Component Analysis) reduz a dimensionalidade dos dados de entrada para 2 componentes principais.
Essa redução serve para:

evitar sobreajuste

acelerar o treinamento

permitir visualização dos dados em 2D

O ponto crítico aqui é que o PCA é ajustado apenas no conjunto de treino, usando: x_train_pca = pca.fit_transform(x_train)

Isso significa que o PCA aprende sua transformação apenas com os dados que o modelo pode ver durante o treinamento — evitando vazamento de informação do teste.

Depois disso, o mesmo PCA transformado é aplicado ao conjunto de teste, sem novo ajuste: x_test_pca = pca.transform(x_test)


```python

# PCA treinado APENAS no conjunto de treino
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

```
# Avaliação do Modelo

| Kernel   | Acurácia |
|----------|----------|
| linear   | 0.9240   |
| sigmoid  | 0.8889   |
| poly     | 0.8480   |
| rbf      | 0.9123   |


### Representação

![alt text](image.png)

- Foram avaliadas quatro variantes do SVM, cada uma utilizando um kernel diferente: linear, sigmoid, polynomial (poly) e rbf.

1. Kernel Linear (0.9240) — Melhor desempenho
A alta acurácia indica que, após o PCA, os dados tornam-se essencialmente separáveis por uma fronteira linear. Isso sugere que a estrutura do problema é relativamente simples no espaço reduzido.

2. Kernel RBF (0.9123) — Segundo melhor
Mesmo sendo mais flexível, o RBF não superou o kernel linear, mostrando que a complexidade extra não traz ganho real neste cenário.

3. Kernel Sigmoid (0.8889) — Desempenho intermediário
O kernel sigmoid tende a ser menos estável e frequentemente oferece resultados inferiores em comparação a kernels mais robustos.

4. Kernel Poly (0.8480) — Pior desempenho
A fronteira polinomial acaba gerando uma complexidade que não reflete bem a estrutura dos dados, prejudicando a generalização.



# Relatório Final

Os experimentos mostraram que:

O PCA cumpriu sua função ao comprimir a variância dos 30 atributos para duas dimensões, permitindo uma separação clara entre as classes.

O SVM com kernel linear mostrou-se o mais adequado para o problema, fornecendo a melhor acurácia e generalização.

Kernels mais complexos, como RBF e Poly, não superaram o modelo linear, reforçando que a separação dos dados no espaço PCA é simples.

### Possíveis Melhorias:

- Avaliação de SVM sem PCA, para comparar o impacto da redução dimensional.

- Aplicação de normalização (StandardScaler) antes do PCA e do SVM.

- Ajuste de hiperparâmetros via GridSearchCV.

- Teste de outras métricas além da acurácia: F1-score, recall, matriz de confusão.

- Uso de métodos mais robustos a ruído, como Random Forest ou Gradient Boosting.