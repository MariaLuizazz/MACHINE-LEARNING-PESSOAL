# Machine Learning Project Template

- Este repositório fornece uma estrutura completa para desenvolver projetos de ciência de dados e machine learning, com foco em reprodutibilidade, organização de código, boas práticas e documentação.

# 🧠 Comparação de Algoritmos de Machine Learning — Câncer de Mama

Este projeto tem como objetivo **comparar o desempenho de diferentes algoritmos de Machine Learning** aplicados a um mesmo problema de classificação: a previsão de câncer de mama a partir de dados clínicos.

Todos os modelos utilizam **a mesma base de dados do Kaggle**, garantindo que a comparação seja justa e que as diferenças de resultado estejam relacionadas **apenas ao comportamento de cada algoritmo**, e não aos dados.

---

## 🎯 Objetivo do Projeto

O principal objetivo é:

* Avaliar como diferentes algoritmos se comportam no mesmo dataset
* Comparar métricas de desempenho como *accuracy, precision, recall, F1-score*, etc.
* Entender os pontos fortes e fracos de cada abordagem
* Criar uma base sólida de estudo sobre **modelos de classificação supervisionada**

Este projeto tem caráter **educacional e experimental**, mas segue uma estrutura organizada e replicável, semelhante ao que é feito em projetos reais de ciência de dados.

---

## 🗂️ Dataset

O dataset utilizado é o **Breast Cancer Dataset** disponível no Kaggle, contendo:

* Features numéricas extraídas de exames
* Uma variável alvo indicando se o tumor é **benigno ou maligno**

O mesmo conjunto de dados e o mesmo pré-processamento são usados para **todos os modelos**, garantindo consistência nos experimentos.

---

## 🤖 Algoritmos Testados

Neste projeto, são testados diferentes tipos de modelos, como por exemplo:

* Regressão Logística
* KNN (K-Nearest Neighbors)
* Árvore de Decisão
* Random Forest
* SVM
* (outros que venham a ser adicionados)

Cada algoritmo é:

* Treinado com os mesmos dados
* Avaliado com as mesmas métricas
* Comparado de forma objetiva com os demais

---

## 🔬 Metodologia

O fluxo de trabalho do projeto segue as etapas:

1. Entendimento do problema
2. Análise exploratória dos dados (EDA)
3. Pré-processamento e tratamento dos dados
4. Treinamento dos modelos
5. Avaliação e comparação dos resultados
6. Análise crítica do desempenho de cada algoritmo

---

## 📊 Resultados

Os resultados mostram claramente que:

* Diferentes algoritmos respondem de formas diferentes ao mesmo problema
* Alguns modelos têm melhor desempenho geral
* Outros podem ser mais simples, mais rápidos ou mais interpretáveis

A análise detalhada de cada modelo e suas métricas está documentada nas seções específicas deste projeto.

---

## 🏗️ Organização do Projeto

O projeto está estruturado de forma a separar:

* Dados
* Notebooks de análise
* Código reutilizável
* Modelos treinados
* Relatórios e visualizações
* Documentação (este site)

Isso facilita a manutenção, a reprodução dos experimentos e o entendimento do projeto.

---

## 💼 Por que este projeto é relevante?

Este tipo de comparação é **extremamente comum no mercado**, pois raramente sabemos de antemão qual algoritmo será o melhor.

O valor está justamente em:

> Testar, medir, comparar e decidir com base em evidência.

Este projeto demonstra não apenas o uso de modelos, mas **método científico aplicado à ciência de dados**.

---
??? info "Informações da Turma"
    - Curso: Ciência de Dados
    - Disciplina: Machine Learning
    - Semestre: 4º Semestre — 2025.2
    - Professor: Humberto Sandmann

---


## Template Pessoal

1. Maria Oliveira



!!! tip "Instruções"

    HUMBERRTOOO se você chegou a esse template a minha árvore de decisão está na aba de ATIVIDADESS.

## Entregas

- [x] Árvore de decisão - Data 29/08/2025
- [X] KNN - Data 16/09/2025
- [ ] Roteiro 3
- [ ] Roteiro 4
- [ ] Projeto

## Diagramas

Use o [Mermaid](https://mermaid.js.org/intro/){:target='_blank'} para criar os diagramas de documentação.

[Mermaid Live Editor](https://mermaid.live/){:target='_blank'}


``` mermaid
flowchart TD
    Deployment:::orange -->|defines| ReplicaSet
    ReplicaSet -->|manages| pod((Pod))
    pod:::red -->|runs| Container
    Deployment -->|scales| pod
    Deployment -->|updates| pod

    Service:::orange -->|exposes| pod

    subgraph  
        ConfigMap:::orange
        Secret:::orange
    end

    ConfigMap --> Deployment
    Secret --> Deployment
    classDef red fill:#f55
    classDef orange fill:#ffa500
```



## Códigos

=== "De um arquivo remoto"

    ``` { .yaml .copy .select linenums='1' title="main.yaml" }
    --8<-- "https://raw.githubusercontent.com/hsandmann/documentation.template/refs/heads/main/.github/workflows/main.yaml"
    ```

=== "Anotações no código"

    ``` { .yaml title="compose.yaml" }
    name: app

        db:
            image: postgres:17
            environment:
                POSTGRES_DB: ${POSTGRES_DB:-projeto} # (1)!
                POSTGRES_USER: ${POSTGRES_USER:-projeto}
                POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-projeto}
            ports:
                - 5432:5432 #(2)!
    ```

    1.  Caso a variável de ambiente `POSTGRES_DB` não exista ou seja nula - não seja definida no arquivo `.env` - o valor padrão será `projeto`. Vide [documentação](https://docs.docker.com/reference/compose-file/interpolation/){target='_blank'}.

    2. Aqui é feito um túnel da porta 5432 do container do banco de dados para a porta 5432 do host (no caso localhost). Em um ambiente de produção, essa porta não deve ser exposta, pois ninguém de fora do compose deveria acessar o banco de dados diretamente.


## Exemplo de vídeo

Lorem ipsum dolor sit amet

<iframe width="100%" height="470" src="https://www.youtube.com/embed/3574AYQml8w" allowfullscreen></iframe>


## Referências

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/){:target='_blank'}