# -*- coding: utf-8 -*-

import numpy as np
import csv
from sklearn.feature_selection import SelectKBest, chi2

np.random.seed(2)

# Lectura de los datos, separación de estos en train y test, y separación de
    # características y etiquetas
trainx, trainy = [], []
testx, testy = [], []

with open('datos/optdigits.tra', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = list(map(int, row))
        trainx.append(row[:-1])
        val = [0] * 10
        val[row[-1]] = 1
        trainy.append(val)

with open('datos/optdigits.tes', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = list(map(int, row))
        testx.append(row[:-1])
        val = [0] * 10
        val[row[-1]] = 1
        testy.append(val)

# Eliminación de características que tienen el mismo valor para todas las instancias
datos = trainx + testx
caracteristicas_triviales = []
for i in range(64):
    columnai = [j[i] for j in datos]
    if len(set(columnai)) == 1: # set(columna) --> valores distintos que se encuentran en la lista
        caracteristicas_triviales.append(i) # si la longitud de la lista es 1, esa posición solo toma un valor

for i in caracteristicas_triviales:
    for j in trainx:
        del j[i]
    for j in testx:
        del j[i]

#Selección de los 20 mejores atributos
datos = trainx + testx
etiquetas = trainy + testy
datos = SelectKBest(chi2, k=20).fit_transform(datos, etiquetas)

trainx = datos[:len(trainx)]
testx = datos[len(trainx):]

#Generación de un valor aleatorio para la matriz de pesos y el vector de bias
W = np.random.uniform(size=(20,10))

#Función softmax
def softmax(y):
    y -= np.max(y)
    sol = (np.exp(y).T / np.sum(np.exp(y), axis=1)).T
    
    return sol

#Función que calcula y=Xw
def net(X, w):
    y_linear = np.dot(X, w)
    yhat = softmax(y_linear)
    return yhat

#Función que calcula el porcentaje de aciertos
def accuracy(X, y, w):
    num_aciertos = 0
    
    output = net(X,w)
    
    for i in range(len(output)):
        prediccion = list(output[i]).index(max(output[i]))
        if prediccion == y[i].index(max(y[i])): num_aciertos += 1

    return num_aciertos / len(X)

#Función que modifica los valores de W y b
def SGD(X, y, yhat, w, eta):
    gradiente = (-1/len(X)) * (np.dot(np.transpose(X), y-yhat))
    
    w -= gradiente * eta
    
    return w

epochs = 1000
eta = 0.01

#Regresión logística Multilabel con SGD
def RegLogMl(X, y, w):
    pesos = w
    iteraciones = 0
    
    X_aux, y_aux = np.copy(X), np.copy(y)
    
    while iteraciones < epochs:
        #Hace una permutación aleatoria en el orden de los datos
        #usando la variable estado para hacer la misma permutación en los dos vectores
        estado = np.random.get_state()
        np.random.shuffle(X_aux)
        np.random.set_state(estado)
        np.random.shuffle(y_aux)
        
        i = 64
        #recorre los datos de X en mini-batches de 64 elementos
        while i < len(X_aux):
            batch_x, batch_y = X_aux[i-64:i], y_aux[i-64:i]
            #calcula las predicciones para el batch
            output = net(batch_x, pesos)
            #modifica el vector de pesos usando gradiente descendente
            pesos = SGD(batch_x, batch_y, output, pesos, eta)
            
            i += 64
        iteraciones += 1
        
    return pesos

W = RegLogMl(trainx, trainy, W)

print(accuracy(trainx, trainy, W))
print(accuracy(testx, testy, W))
