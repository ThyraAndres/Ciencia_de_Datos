
"""
Created on Sat May 20 14:50:01 2022

@author: William Andres
"""


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas 
import math

filename = input('Ingrese el Dataframe: ')
if len(filename) < 1: filename = "Not Exist.txt"
filn = open(filename)
dataset = pandas.read_csv(filn)
print(dataset.shape)
""" 
1er Forma
Muestreo con reemplazo en aleatoriedad  """
print(dataset.sample(frac=2/3, replace=True)) #fracciono los datos inicialmente eran 206, ahora: 137
print(dataset.sample(frac=2/3, replace=True)) # se vuelven a fraccionar pero en otra cantidad por el random
print(dataset.sample(frac=2/3, replace=True))   # replace= true. se asegura de que ningún elemento se produzca dos veces.
print(dataset.sample(frac=2/3, replace=True))
print(dataset.sample(frac=2/3, replace=True))

print(dataset.sample(frac=2/3, replace=True))
print(dataset.sample(frac=2/3, replace=True))
print(dataset.sample(frac=2/3, replace=True))

"""
2da Forma

"""
print("------------------ 2DA FORMA-------------------")

dataset.describe()
dataset.head()
# Información del dataset
dataset.info()

#dataset.drop(['Unnamed: 0'], axis=1)
from random import sample
#tercera forma de aleatorizacion para bosques aleatorios
#cv = math.sqrt(len(dataset.columns))

print("Numero de columnas: " ,len(dataset.columns))
cv = 4
print(dataset.columns[:-1], "\n")
print("sample: ", sample(set(dataset.columns[:-1]), cv))
#las caracteristicas a mostrar van a depender del total de columnas

from sklearn.ensemble import RandomForestClassifier

#Creacion del bosque, mostreo aleatorio. max_sampl para indicar q parametros vamos a mostrar
#cada arbol se crea usando 2/3 de los arboles. se estan creando 100 arboles
#oob signifca fuera de la bolsa, quiere decir se van a guardar 1/3 de los arboles, 
# el resto de instancias no seleccionadas son las oob. Ademas cada uno de esos arboles se evaluan con aquellas
# instancias q no entraron

#configuramos la creacion de bosques aleatorios
bosque = RandomForestClassifier(n_estimators=4,
                               criterion="gini",
                               max_features="sqrt",
                               bootstrap=True,
                               max_samples=2/3,
                               oob_score=True)

"""
normalmente las estimaciones son 100, los criterios son entropia "ganancia de informacion", o gini.
el gini es el area debajo de la curva lorence, usualmente calculado para analizar la distribucion de un income en poblacion
     lorence --> representación gráfica de la desigualdad en el reparto de la renta existente en un determinado territorio 
max_features es el numero maximo de caracteristicas = raiz
bootsrap por defecto esta habilitado.
max_samples es para mostrar el porcentaje a mostrar, en este caso 2/3 como en el ejemplo
oob_score, es para que guarde el 1/3. es una metrica especial de los bosques aleatorios.
"""

# le pasamos los datos del dataset.
bosque.fit(dataset[dataset.columns[:-1]].values, dataset["Outcome"].values)

# ejecutamos una prediccion, le pasamos una lista
print(bosque.predict([[1,89,66,23,94,2,0.167,21]]))

# score de evaluacion para instancias dentro de la bolsa. --> 2/3
print("Score : ", bosque.score(dataset[dataset.columns[:-1]].values, dataset["Outcome"].values))

# score de evaluacion para instancias fuera de la bolsa --> 1/3
print("instancias fuera de bolsa:  ", bosque.oob_score_)


import matplotlib.pyplot as plt
from sklearn import tree


# estimators_ es un atributo sobre el cual podemos iterar ( son los # de arboles que creamos.)
for arbol in bosque.estimators_:
    tree.plot_tree(arbol, feature_names=dataset.columns[:-1])
    plt.show()

