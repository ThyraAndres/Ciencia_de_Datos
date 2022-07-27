# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:52:01 2022

@author: William Andres
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

filename = input('Ingrese el Dataframe: ')
if len(filename) < 1: filename = "Not Exist.txt"
filn = open(filename)
dataset = pd.read_csv(filn)
print(dataset.shape)

dataset.head()
from sklearn.model_selection import train_test_split

dataset.describe()
dataset.head()
# Seleccionamos las características para el modelo
data = dataset[:-1]
data.head()
# Información del dataset
data.info()
# Dividimos los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
# X son nuestras variables independientes
X = data.drop(["Outcome"],axis = 1)

# y es nuestra variable dependiente
y = data.Outcome

# División 75% de datos para entrenamiento, 25% de daatos para test
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


# Creaamos el modelo de Bosques Aleatorios (y configuramos el número de estimadores (árboles de decisión))
BA_model = RandomForestClassifier(n_estimators = 19, 
                                  random_state = 2016,
                                  min_samples_leaf = 8,)

# random_state ->fija una semilla para el generador de números aleatorios, lo que permite reproducir la función. 

# Entrenamiento
BA_model.fit(X_train, y_train)
# Accuracy promedio
c = BA_model.score(X_test, y_test)

print("score: ", c)
# Matriz de confusion
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Predicción del modelo usando los datos de prueba
y_pred = BA_model.predict(X_test)
matriz = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(conf_mat=matriz, figsize=(6,6), show_normed=False)
plt.tight_layout()
plt.show()