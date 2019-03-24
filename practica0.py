# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:27:36 2019

@author: Natalia Fernández Martínez
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

#EJERCICIO 1
print("Ejercicio 1")
#Lee la base de datos de iris
iris = datasets.load_iris()

#Obtiene las características y la clase
X = iris.data
y = iris.target

#Guarda las dos últimas columnas de las características
dos_ultimos = X[-2::]

#Scatter plot
plt.xlabel("Largo del sépalo")
plt.ylabel("Ancho del sépalo")
plt.scatter(X[0:50:,0],X[0:50:,1], label="Setosa", c="red")
plt.scatter(X[50:100:,0],X[50:100:,1], label="Versicolor", c="green")
plt.scatter(X[100::,0],X[100::,1], label="Virginica", c="blue")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente ejercicio")

#EJERCICIO 2
print("Ejercicio 2")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

input("Pulse una tecla para pasar al siguiente ejercicio")

#EJERCICIO 3
print("Ejercicio 3")
#100 valores equiespaciados entre 0 y 2π
valores = np.linspace(0, 2*math.pi, num=100)

#sin(x), cos(x) y sin(x)+cos(x)
seno = []
coseno = []
sincos = []
for i in valores:
    seno.append(math.sin(i))
    coseno.append(math.cos(i))
    sincos.append(math.sin(i) + math.cos(i))

#plot
plt.plot(valores, seno, 'r--', label="seno")
plt.plot(valores, coseno, 'k--', label="coseno")
plt.plot(valores, sincos, 'b--', label="seno + coseno")
plt.legend(loc="lower right")
