import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import StringIO


df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')

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


# Criação e treino do modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(x_train, y_train)

# Avaliação
predictions = rf.predict(x_test)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions):.4f}")

# Importância das variáveis
importances = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\n📊 Importância das Features:")
print(importances.head(10))

# Plot de uma árvore individual
fn = list(x.columns)             # nomes das features
cn = ['Benigno', 'Maligno']      # classes do diagnóstico

fig, ax = plt.subplots(figsize=(20,10), dpi=150)
tree.plot_tree(
    rf.estimators_[0],
    feature_names=fn,
    class_names=cn,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)

# Plot das 5 primeiras árvores da floresta
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25,5), dpi=150)
for index in range(5):
    tree.plot_tree(
        rf.estimators_[index],
        feature_names=fn,
        class_names=cn,
        filled=True,
        ax=axes[index],
        fontsize=6
    )
    axes[index].set_title(f"Árvore {index+1}", fontsize=10)
plt.tight_layout()
plt.show()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.title("🌲 Árvore Individual da Random Forest")
plt.show()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

