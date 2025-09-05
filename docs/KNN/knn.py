import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd

plt.figure(figsize=(12,10))

#carregamento da base
df = pd.read_csv('https://raw.githubusercontent.com/MariaLuizazz/MACHINE-LEARNING-PESSOAL/refs/heads/main/dados/breast-cancer.csv')





#separação de treino e teste

#Treianamento do KNN

#teste e validação

#Preparação para o gráfico da fronteira de decisão

#Prevendo classe em cada ponto

#gráfico final
