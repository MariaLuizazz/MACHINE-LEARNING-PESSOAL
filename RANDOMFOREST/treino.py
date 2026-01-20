
rf = RandomForestClassifier(
    n_estimators=100,     # número de árvores
    max_depth=5,          # profundidade máxima
    max_features='sqrt',  # número de variáveis avaliadas por split
    random_state=42
)
rf.fit(x_train, y_train)
