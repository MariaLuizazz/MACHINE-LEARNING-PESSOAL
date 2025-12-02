import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from scipy.stats import mode


df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

# Pré-processamento
df = df.drop(columns=['id'])
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Corrigir valores ausentes
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)

# Features selecionadas
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean']]
y = df['diagnosis']

# =========================
# Modelo 1: KNN
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Treinamento do KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_bal, y_train_bal)

# Predição
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)

# =========================
# Modelo 2: KMeans
# =========================
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

pca_full = PCA(n_components=2)
X_pca_full = pca_full.fit_transform(X_scaled_full)

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca_full)

# Mapear clusters para classes reais
mapping = {}
for cluster in np.unique(clusters):
    mask = clusters == cluster
    # mode agora precisa de indexação [0] para pegar o valor
    mapping[cluster] = mode(y[mask], keepdims=True).mode[0]

y_pred_kmeans = [mapping[c] for c in clusters]
cm_kmeans = confusion_matrix(y, y_pred_kmeans)

# =========================
# Impressão em Markdown
# =========================
def matriz_markdown(cm, labels, titulo):
    md = f"### {titulo}\n\n"
    md += f"|                 | Previsto {labels[0]} | Previsto {labels[1]} |\n"
    md += f"|-----------------|------------------|------------------|\n"
    md += f"| **Real {labels[0]}** | {cm[0,0]}              | {cm[0,1]}               |\n"
    md += f"| **Real {labels[1]}** | {cm[1,0]}              | {cm[1,1]}               |\n"
    return md

print(matriz_markdown(cm_knn, ["Benigno", "Maligno"], "Matriz de Confusão - KNN"))
print(matriz_markdown(cm_kmeans, ["Benigno", "Maligno"], "Matriz de Confusão - KMeans"))
