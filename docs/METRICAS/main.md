# ✅ Métricas de avaliação CLÍNICAS
print("\n" + "="*50)
print("AVALIAÇÃO CLÍNICA DO MODELO")
print("="*50)

# Matriz de Confusão
cm = confusion_matrix(y_test, predictions)
print("\nMatriz de Confusão:")
print(cm)

# Relatório de Classificação com métricas por classe
print("\nRelatório de Classificação:")
print(classification_report(y_test, predictions, 
                          target_names=['Benigno', 'Maligno']))

# ✅ Métricas específicas para câncer
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"\nMétricas Clínicas:")
print(f"Sensibilidade (Recall Maligno): {sensitivity:.3f} → % de câncer detectado")
print(f"Especificidade: {specificity:.3f} → % de saudáveis corretamente identificados")
print(f"Falsos Negativos: {fn} → Casos de câncer NÃO detectados 🚨")
print(f"Falsos Positivos: {fp} → Saudáveis classificados como câncer")

# ✅ Validação Cruzada com foco em sensibilidade
print("\nValidação Cruzada (Acurácia):")
scores_acc = cross_val_score(knn, X_train_balanced, y_train_balanced, cv=5)
print(f"Acurácia: {scores_acc.mean():.3f} ± {scores_acc.std():.3f}")

# ✅ Testar com k maior para verificar generalização
knn_k11 = KNeighborsClassifier(n_neighbors=11)
knn_k11.fit(X_train_balanced, y_train_balanced)
print(f"K=11 Accuracy: {accuracy_score(y_test, knn_k11.predict(X_test)):.3f}")