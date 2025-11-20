# svm_kernels_pca_correct.py
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# 1) Carregamento e limpeza
# -------------------------
url = 'https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv'
df = pd.read_csv(url)

# Exibir amostra rápida (opcional)
print(df.sample(n=6, random_state=42).to_markdown(index=False))

# Remover coluna irrelevante
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Converter diagnosis (M/B) para 1/0
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# Imputação simples (se houver NaNs nas colunas citadas)
for col in ['concavity_mean', 'concave points_mean']:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# -------------------------
# 2) Separação X / y
# -------------------------
X_full = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# -------------------------
# 3) Divisão treino / teste
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.30, random_state=42, stratify=y
)

# -------------------------
# 4) PCA (treinar só no treino)
# -------------------------
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train)    # fit no treino
X_test_pca = pca.transform(X_test)          # transformar teste
X_pca_all = pca.transform(X_full)           # para plotar todas as amostras em 2D

# -------------------------
# 5) Treinar SVMs e plotar
# -------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
kernels = {
    'linear': ax1,
    'sigmoid': ax2,
    'poly': ax3,
    'rbf': ax4
}

results = []  # para coletar métricas

for kernel_name, ax in kernels.items():
    # Cria o classificador (ajuste básico; você pode tunar C, degree, gamma etc.)
    svm = SVC(kernel=kernel_name, C=1, probability=False, random_state=42)

    # Treina NO CONJUNTO DE TREINO (em PCA 2D)
    svm.fit(X_train_pca, y_train)

    # Avaliação no conjunto de teste (PCA 2D)
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['maligno (0)', 'benigno (1)'])
    cm = confusion_matrix(y_test, y_pred)

    # Guardar resultados
    results.append({
        'kernel': kernel_name,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm
    })

    # Plot da fronteira de decisão usando TODAS as amostras projetadas (X_pca_all)
    DecisionBoundaryDisplay.from_estimator(
        estimator=svm,
        X=X_pca_all,
        response_method="predict",
        alpha=0.6,
        cmap="viridis",
        ax=ax
    )

    # Plot dos pontos reais (todas as amostras projetadas)
    ax.scatter(
        X_pca_all[:, 0],
        X_pca_all[:, 1],
        c=y,
        s=20,
        edgecolors="k",
        linewidth=0.4
    )
    ax.set_title(f"SVM kernel = {kernel_name} — acc teste: {acc:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])

# Layout e exibição
plt.tight_layout()

# Salvar figura (opcional)
plt.savefig("svm_kernels_pca.png", dpi=300, bbox_inches='tight')

# Se quiser imprimir o SVG no console (como você fazia), descomente abaixo:
# buffer = StringIO()
# plt.savefig(buffer, format="svg", transparent=True)
# print(buffer.getvalue())

plt.show()

# -------------------------
# 6) Exibir métricas detalhadas
# -------------------------
for r in results:
    print("="*60)
    print(f"Kernel: {r['kernel']}")
    print(f"Acurácia (teste): {r['accuracy']:.4f}")
    print("Matriz de confusão (teste):")
    print(r['confusion_matrix'])
    print("Relatório de classificação (teste):")
    print(r['report'])
