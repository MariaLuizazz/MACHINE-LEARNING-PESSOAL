import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


plt.figure(figsize=(12,10))

#carregamento da base
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

#Préprocess
#remoção da coluna id pois é irrelevante para o modelo
df = df.drop(columns=['id'])

#conversão de letra para número
label_encoder = LabelEncoder()  
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])


#imputação com mediana de valores ausentes nas features concavity_worts e concavity points_worst
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)

#escolha de features
X = df[['radius_mean', 'texture_mean']]
y = df['diagnosis']

#Separação de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Treianamento do KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


#Teste e validação
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


# Mapeia os rótulos: 0 -> Benigno, 1 -> Maligno
labels_map = {0: "Benigno", 1: "Maligno"}
y_labels = y.map(labels_map)


#Preparação para o gráfico da fronteira de decisão(malha de visualização)
h = 0.02
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

#Prevendo classe em cada ponto
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#gráfico final
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn_r, alpha=0.3)
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y_labels, style=y_labels, palette={'Benigno': 'green', 'Maligno': 'red'}, s=100) #motivooooooo do errroo
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.title("KNN Decision Boundary (k=3) -  Diagnóstico de Câncer")
plt.legend(title="Diagnóstico")  



#Exibição do gráfico
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())



