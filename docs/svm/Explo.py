import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabulate import tabulate  
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay



#carregamento da base
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')
print(df.sample(n=10, random_state=42).to_markdown(index=False))

# Remover coluna irrelevante
df = df.drop(columns=['id'])

# Conversão de letra para número (diagnosis: M=1, B=0)
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Imputação com mediana para valores nulos
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)

# Separação entre variáveis independentes e alvo
x = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Divisão treino/teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)


#IDENTIFICAR O MELHOR HIPERPLANO


#cada subplot é usado para exibir o SVM com um Kernel diferente
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}


# loop pra treinar e plotar os svm


for k, ax in kernels.items():
    svm = SVC(kernel=k, C=1)
    svm.fit(x, y)
    svm.score(x, y)


    DecisionBoundaryDisplay.from_estimator(
            svm,#argumento
            x,#argumento(variaveis independentes)
            response_method="predict",
            alpha=0.6,
            cmap="viridis",
            ax=ax
    )
    ax.scatter(
        x[:, 0],#feature 1 no eixo X
        x[:, 1], # feature 2 no eixo Y
        c=y, 
        s=20, edgecolors="k"
    )
    ax.set_title(k)
    ax.set_xticks([]) 
    ax.set_yticks([])


    predicted= svm.predict(x_test)

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
