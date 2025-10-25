from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')


df = df.drop(columns=['id'])
#conversão de letra para número
label_encoder = LabelEncoder()  
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

#features escolhidas, todas menos diagnosis e id
x = df.drop(columns=['diagnosis'])
y = df['diagnosis']

#imputação com mediana de valores ausentes nas features concavity_worts e concavity points_worst
df['concavity_mean'].fillna(df['concavity_mean'].median(), inplace=True)
df['concave points_mean'].fillna(df['concave points_mean'].median(), inplace=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#algoritmo de random forest
rf = RandomForestClassifier(n_estimators=100,   # Number of trees
                            max_depth=5,        # Max depth of trees  
                            max_features='sqrt', # Features per split
                            random_state=42)

rf.fit(x_train, y_train)

# Predict and evaluate
predictions = rf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Feature importances
print(f"Feature Importances: {rf.feature_importances_}")