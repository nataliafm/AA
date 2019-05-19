# -*- coding: utf-8 -*-

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statistics
import matplotlib.pyplot as plt

np.random.seed(2)

#Lectura del fichero de datos y separación de estos en características y etiquetas
datosx, datosy = [], []
with open('datos/airfoil_self_noise.dat', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        lista_caracteristicas = ''.join(row) #convierte la fila en un string
        caracteristicas = lista_caracteristicas.split("\t") #divide el string por el separador \t
        caracteristicas = [float(i) for i in caracteristicas] #convierte las características en floats
        
        minimo = min(caracteristicas)
        maximo = max(caracteristicas)
        media = statistics.mean(caracteristicas)
        
        datosx.append(caracteristicas[:-1])
        datosy.append(caracteristicas[-1])

datosx = preprocessing.scale(datosx)

#Concatenar una columna de 1s en primer lugar para poder hacer los cálculos de regresión lineal
unos = np.ones([len(datosx),1])
datosx = np.concatenate((unos,datosx),axis=1)

#Separar los datos en train y test
trainx, testx, trainy, testy = train_test_split(datosx, datosy, test_size=0.3, shuffle=True)

eta = 0.001
max_iters = 10000
W = np.random.uniform(size=len(datosx[0]))

def cost(X, y, w):
    y_hat = np.dot(X,w)
    suma = y_hat - y
    suma = [i**2 for i in suma]
    return sum(suma) / (2*len(X))

def gradient_descent(X, y, w, max_iters, eta):
    costs = []
    
    for i in range(max_iters):
        pred = np.dot(X, w)
        w -= (eta * (np.dot(np.transpose(X), pred-y)))/len(X)
        costs.append(cost(X,y,w))
        
    ''' #regularización
    for i in range(max_iters):
        Xt = np.transpose(X)
        l_ident = 0.3 * np.identity(len(Xt))
        inversa = np.linalg.inv(np.dot(Xt,X) + l_ident)
        w = np.dot(np.dot(inversa, Xt),y)
    '''
        
    return w, costs

def MSE(X, y, w):
    y_hat = np.dot(X,w)
    suma = y_hat - y
    suma = [i**2 for i in suma]
    suma = sum(suma)
    suma /= len(X)
    return suma

MSE_base_test = MSE(testx, testy, W)
MSE_base_train = MSE(trainx, trainy, W)

W, costs = gradient_descent(trainx, trainy, W, max_iters, eta)

MSE_model_test = MSE(testx, testy, W)
MSE_model_train = MSE(trainx, trainy, W)

print('Costes: ', costs[0], ", ", costs[-1])
print('MSE baseline: ', MSE_base_test, MSE_base_train)
print('MSE modelo: ', MSE_model_test, MSE_model_train)
print("R-squared: ", (1 - (MSE_model_train / MSE_base_train)), (1 - (MSE_model_test / MSE_base_test)))

plt.figure(1)
plt.plot(range(max_iters), costs)
plt.title('Evolución del valor de la función de coste')
plt.xlabel('Iteraciones')
plt.ylabel('Coste')
plt.show()