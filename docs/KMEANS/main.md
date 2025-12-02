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
    --8<-- "docs/decision-tree/Exploracaodedados.py"
    ``` 
=== "Resultado"

    ```python exec="on" html="0"
    --8<-- "docs/decision-tree/Exploracaodedados.py"
    ```





Cada ponto azul/roxo/amarelo no gráfico é um paciente representado nessas duas dimensões condensadas.

Esses componentes carregam a maior parte da variabilidade dos dados originais (30 features)

Você aplicou PCA para reduzir a dimensionalidade dos dados.

O primeiro componente principal explica ≈44,3% da variância total dos dados.

O segundo componente principal explica ≈18,97% da variância.

Isso significa que, juntos, os dois componentes capturam ≈63,2% da informação original do dataset.

# Relatório do Modelo: K-Means com PCA

## 1. Descrição do Modelo
O modelo aplicado combina duas técnicas:

1. **PCA (Análise de Componentes Principais)**: usada para reduzir a dimensionalidade dos dados originais, mantendo a maior parte da variabilidade.  
2. **K-Means**: algoritmo de clustering que agrupa os dados em 3 clusters distintos com base na proximidade aos centróides.

O objetivo do modelo é identificar **agrupamentos naturais** nos dados de câncer de mama, considerando as características das células.

---

## 2. Variância Explicada pelos Componentes Principais

| Componente Principal | Variância Explicada | Variância Acumulada |
|---------------------|-------------------|-------------------|
| PC1                 | 0,44272           | 0,44272           |
| PC2                 | 0,189712          | 0,632432          |

- **PC1** explica aproximadamente **44,3%** da variância dos dados.  
- **PC2** explica aproximadamente **18,97%**.  
- **Variância total explicada pelos 2 componentes:** 63,24%.  


> **Interpretação:** Mais da metade da informação original dos dados é preservada nesses dois componentes, permitindo uma visualização e análise de clusters eficaz.

---

## 3. Centròides Finais dos Clusters

Os centróides identificados pelo K-Means no espaço reduzido (PCA) são:

[[ 2.67596132 3.31195566] → Cluster 1
[-2.3259273 -0.20749033] → Cluster 2
[ 4.90974577 -1.89255356]] → Cluster 3


- Cada centróide representa o **“ponto médio”** de cada cluster.  
- Cada ponto do dataset é atribuído ao cluster cujo centróide está mais próximo.

---

## 4. Inércia (WCSS)

- **Inércia final:** 3871,15  
- Representa a soma das distâncias quadráticas dos pontos aos seus centróides.  
- Quanto menor a inércia, mais compactos são os clusters.

> **Interpretação prática:** A dispersão dos clusters é moderada. Para otimizar o número de clusters, pode-se usar o **método do cotovelo**.

---

## 5. Gráfico dos Clusters


---

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/KMEANS/km.py"
    ```

=== "Code"

    ```python exec="0"
    --8<-- "docs/KMEANS/km.py"
    ```


## 6. Conclusão

- O modelo reduziu os dados para 2 dimensões mantendo **63% da variância**, facilitando visualização.  
- Foram identificados **3 clusters distintos**, cada um representado por um centróide.  
- O WCSS sugere que os clusters são relativamente coesos.


1. Explorar mais componentes do PCA para capturar mais variabilidade.  
2. Testar diferentes números de clusters (K) usando o método do cotovelo.  
3. Investigar quais características impactam mais a separação dos clusters.
