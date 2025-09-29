import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# ==============================
# Carregamento da base
# ==============================
df = pd.read_csv("https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv")

# Remover id
df = df.drop(columns=["id"])

# Label encoding
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])  # M=1, B=0

# Features escolhidas
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean']]
y = df["diagnosis"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Treinamento KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_bal, y_train_bal)

# Predições
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)[:, 1]

# ==============================
# Métricas de avaliação
# ==============================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("Métricas KNN")
print(f"Acurácia:      {acc:.4f}")
print(f"Precisão:      {prec:.4f}")
print(f"Sensibilidade: {rec:.4f}")
print(f"Especificidade:{specificity:.4f}")
print(f"F1-Score:      {f1:.4f}")

# ==============================
# Matriz de Confusão
# ==============================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - KNN")
plt.show()

# ==============================
# Curva ROC
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdadeiros Positivos")
plt.title("Curva ROC - KNN")
plt.legend()
plt.show()

# ==============================
# Curva Precision-Recall
# ==============================
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recalls, precisions)

plt.figure(figsize=(6,5))
plt.plot(recalls, precisions, color="green", label=f"AUC-PR = {pr_auc:.4f}")
plt.xlabel("Recall (Sensibilidade)")
plt.ylabel("Precisão")
plt.title("Curva Precision-Recall - KNN")
plt.legend()
plt.show()
