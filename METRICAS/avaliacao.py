import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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

#knn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


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

# Métricas KNN
acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)


# Modelo 2: KMeans
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
    mapping[cluster] = mode(y[mask], keepdims=True).mode[0]

y_pred_kmeans = [mapping[c] for c in clusters]
cm_kmeans = confusion_matrix(y, y_pred_kmeans)

# Métricas KMeans
acc_kmeans = accuracy_score(y, y_pred_kmeans)
prec_kmeans = precision_score(y, y_pred_kmeans)
rec_kmeans = recall_score(y, y_pred_kmeans)
f1_kmeans = f1_score(y, y_pred_kmeans)


# Impressão em Markdown
def matriz_markdown(cm, labels, titulo):
    md = f"### {titulo}\n\n"
    md += f"|                 | Previsto {labels[0]} | Previsto {labels[1]} |\n"
    md += f"|-----------------|------------------|------------------|\n"
    md += f"| **Real {labels[0]}** | {cm[0,0]}              | {cm[0,1]}               |\n"
    md += f"| **Real {labels[1]}** | {cm[1,0]}              | {cm[1,1]}               |\n"
    return md

print(matriz_markdown(cm_knn, ["Benigno", "Maligno"], "Matriz de Confusão - KNN"))
print(f"- Acurácia: {acc_knn:.3f}\n- Precisão: {prec_knn:.3f}\n- Recall: {rec_knn:.3f}\n- F1-score: {f1_knn:.3f}\n")

print(matriz_markdown(cm_kmeans, ["Benigno", "Maligno"], "Matriz de Confusão - KMeans"))
print(f"- Acurácia: {acc_kmeans:.3f}\n- Precisão: {prec_kmeans:.3f}\n- Recall: {rec_kmeans:.3f}\n- F1-score: {f1_kmeans:.3f}\n")


# Comparação lado a lado em Markdown
comparacao = f"""
### Comparação de Métricas

| Modelo   | Acurácia | Precisão | Recall | F1-score |
|----------|----------|----------|--------|----------|
| **KNN**  | {acc_knn:.3f} | {prec_knn:.3f} | {rec_knn:.3f} | {f1_knn:.3f} |
| **KMeans**| {acc_kmeans:.3f} | {prec_kmeans:.3f} | {rec_kmeans:.3f} | {f1_kmeans:.3f} |
"""
print(comparacao)
