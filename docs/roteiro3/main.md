
Running the code below in Browser (Woooooowwwwww!!!!!!). [^1]


``` pyodide install="pandas,ssl"
import ssl
import pandas as pd

df = pd.DataFrame()
df['AAPL'] = pd.Series([1, 2, 3])
df['MSFT'] = pd.Series([4, 5, 6])
df['GOOGL'] = pd.Series([7, 8, 9])

print(df)

```

[^1]: [Pyodide](https://pawamoy.github.io/markdown-exec/usage/pyodide/){target="_blank"}

# Exploração de dados

O câncer de mama é o câncer mais comum entre as mulheres do mundo. É responsável por 25% de todos os casos de câncer e afetou mais de 2,1 milhões de pessoas apenas em 2015. Começa quando as células da mama começam a crescer fora de controle. Essas células geralmente formam tumores que podem ser vistos via raios-X ou sentidos como nódulos na área da mama.

O principal desafio contra sua detecção é como classificar os tumores em malignos (cancerosos) ou benignos (não cancerosos), o intuito dessa entrega é criar um modelo que preveja a variavel target, classificada em tumores malignos ou benignos.

``` python exec="on" html="0"
--8<-- "./docs/EXERCICIOAD/Exploracaodedados.py"
```
