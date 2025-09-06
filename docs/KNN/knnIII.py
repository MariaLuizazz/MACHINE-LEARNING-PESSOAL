# Reimportando bibliotecas após reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Carrega o dataset
df = pd.read_csv("https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv")

# Remove a coluna id
df = df.drop(columns=["id"])

# Converte diagnosis (M = 1, B = 0)
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

# Gráfico comparando radius_mean vs concave points_mean
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df,
    x="radius_mean",
    y="concave points_mean",
    hue="diagnosis",
    style="diagnosis",
    palette="deep",
    s=100
)
plt.title("Relação entre Raio Médio e Pontos Côncavos Médios (Benigno vs Maligno)")
plt.xlabel("Radius Mean (Tamanho do Nódulo)")
plt.ylabel("Concave Points Mean (Irregularidade das Bordas)")
plt.legend(title="Diagnóstico", labels=["Benigno (0)", "Maligno (1)"])
plt.show()
