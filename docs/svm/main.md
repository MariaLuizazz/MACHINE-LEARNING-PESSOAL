
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



# Divisão de Dados e Treinamneto do Modelo.

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

![alt text](image.png)

- Foram avaliadas quatro variantes do SVM, cada uma utilizando um kernel diferente: linear, sigmoid, polynomial (poly) e rbf.

1. Kernel linear apresentou a melhor acurácia (0.9240).
Isso indica que a separação entre as classes do problema é essencialmente linear após a transformação via PCA. Em outras palavras, os dois componentes principais já organizam os dados de forma que uma fronteira linear é suficiente para discriminar benigno vs maligno com alta precisão.

2. Kernel rbf ficou muito próximo (0.9123).
O RBF é mais flexível e captura relações não lineares, mas sua vantagem não se manifesta aqui. Isso reforça a ideia de que o espaço reduzido pelo PCA já tem separação relativamente simples. Ainda assim, o desempenho está consistente e estável.

3. Kernel sigmoid teve desempenho intermediário (0.8889).
O sigmoid tende a ser instável e sensível a parâmetros; não é um kernel amplamente recomendado para classificação prática. A menor acurácia indica que ele não modela bem a estrutura dos dados após a redução dimensional.

4. Kernel poly foi o pior (0.8480).
O kernel polinomial pode criar fronteiras excessivamente complexas ou rígidas dependendo do grau padrão (degree=3). O desempenho mais baixo sugere que essa complexidade não representa bem a distribuição dos dados, gerando fronteiras que não generalizam tão bem no conjunto de teste.



# Relatório Final

Conclusão técnica

Os resultados mostram que o kernel linear é o mais adequado para este problema quando os dados são reduzidos a duas dimensões via PCA.
A separabilidade linear implica que:

- o PCA conseguiu capturar a maior parte da variabilidade relevante,

- as classes ficam relativamente bem separadas nesse novo espaço,

- e um modelo mais simples (linear) generaliza melhor do que kernels mais flexíveis.

Isso é desejável em termos práticos, pois modelos lineares são mais interpretáveis, mais rápidos e menos suscetíveis a sobreajuste nesse cenário.