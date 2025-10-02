import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA  

plt.figure(figsize=(12,10))

# Carregamento da base
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

# Pré-processamento
# Remoção da coluna id pois é irrelevante para o modelo
df = df.drop(columns=['id'])

# Conversão de letra para número
label_encoder = LabelEncoder()  
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Imputação com mediana de valores ausentes
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)

# Escolha de features - 7 features
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',  
        'smoothness_mean', 'compactness_mean', 'concavity_mean']]
y = df['diagnosis']

# NORMALIZAÇÃO das features (importante para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# REDUÇÃO DE DIMENSIONALIDADE com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variância explicada pelas componentes principais:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total: {pca.explained_variance_ratio_.sum():.3f}")

# Separação de treino e teste com dados transformados pelo PCA
X_train, X_test, y_train, y_test = train_test_split( X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Treinamento do KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_balanced, y_train_balanced) 

# Teste e validação
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Mapeia os rótulos: 0 -> Benigno, 1 -> Maligno
labels_map = {0: "Benigno", 1: "Maligno"}
y_labels = y.map(labels_map)

# PREPARAÇÃO PARA O GRÁFICO com dados PCA
h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# PREVISÕES na space do PCA (2D)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico final
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn_r, alpha=0.3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolor='k')
plt.xlabel(f"Primeira Componente Principal (PC1 - {pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"Segunda Componente Principal (PC2 - {pca.explained_variance_ratio_[1]:.1%})")
plt.title("KNN Decision Boundary (k=11) - Diagnóstico de Câncer\n(7 Features Reduzidas por PCA)")

# Adicionar legenda manualmente para as classes
import matplotlib.patches as mpatches
benign_patch = mpatches.Patch(color='green', label='Benigno')
malign_patch = mpatches.Patch(color='purple', label='Maligno')
plt.legend(handles=[benign_patch, malign_patch], title="Diagnóstico")

# Exibição do gráfico
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

# VALIDAÇÃO CRUZADA com dados originais (não PCA)
# Para validação, usar dados originais para melhor avaliação
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Pipeline completo com scaling e PCA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=11))
])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Validação Cruzada com PCA: {scores.mean():.3f} ± {scores.std():.3f}")

# Matriz de Confusão
from sklearn.metrics import confusion_matrix
print("Matriz de Confusão:")
print(confusion_matrix(y_test, predictions))