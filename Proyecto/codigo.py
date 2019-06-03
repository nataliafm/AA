# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hinge_loss
from sklearn.model_selection import cross_val_score

np.random.seed(2)
# Lectura de los datos
datos = []
datos = pd.read_csv('datos/agaricus-lepiota.data')
# Codificación de los datos para convertirlos de valores categóricos a numéricos

le = LabelEncoder()
for col in datos.columns:
    datos[col] = le.fit_transform(datos[col])
    
print('Gráficas con información de las variables')

#Gráfico de la distribución de las clases
plt.figure(1)
pd.Series(datos['p']).value_counts().sort_index().plot(kind = 'bar')
plt.xlabel("clase")
plt.title('Distribución de las clases (0=comestible, 1=venenosa)')
plt.show()

#Gráficos de la distribución de los datos
plt.figure(2)
pd.Series(datos['x']).value_counts().sort_index().plot(kind = 'bar')
plt.title('cap-shape')
plt.show()

plt.figure(3)
pd.Series(datos['s']).value_counts().sort_index().plot(kind = 'bar')
plt.title('cap-surface')
plt.show()

plt.figure(4)
pd.Series(datos['n']).value_counts().sort_index().plot(kind = 'bar')
plt.title('cap-color')
plt.show()

plt.figure(5)
pd.Series(datos['t']).value_counts().sort_index().plot(kind = 'bar')
plt.title('bruises?')
plt.show()

plt.figure(6)
pd.Series(datos['p.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('odor')
plt.show()

plt.figure(7)
pd.Series(datos['f']).value_counts().sort_index().plot(kind = 'bar')
plt.title('gill-attachment')
plt.show()

plt.figure(8)
pd.Series(datos['c']).value_counts().sort_index().plot(kind = 'bar')
plt.title('gill-spacing')
plt.show()

plt.figure(9)
pd.Series(datos['n.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('gill-size')
plt.show()

plt.figure(10)
pd.Series(datos['k']).value_counts().sort_index().plot(kind = 'bar')
plt.title('gill-color')
plt.show()

plt.figure(11)
pd.Series(datos['e']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-shape')
plt.show()

plt.figure(12)
pd.Series(datos['e.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-root')
plt.show()

plt.figure(13)
pd.Series(datos['s.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-surface-above-ring')
plt.show()

plt.figure(14)
pd.Series(datos['s.2']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-surface-below-ring')
plt.show()

plt.figure(15)
pd.Series(datos['w']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-color-above-ring')
plt.show()

plt.figure(16)
pd.Series(datos['w.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('stalk-color-below-ring')
plt.show()

plt.figure(17)
pd.Series(datos['p.2']).value_counts().sort_index().plot(kind = 'bar')
plt.title('veil-type')
plt.show()

plt.figure(18)
pd.Series(datos['w.2']).value_counts().sort_index().plot(kind = 'bar')
plt.title('veil-color')
plt.show()

plt.figure(19)
pd.Series(datos['o']).value_counts().sort_index().plot(kind = 'bar')
plt.title('ring-number')
plt.show()

plt.figure(20)
pd.Series(datos['p.3']).value_counts().sort_index().plot(kind = 'bar')
plt.title('ring-type')
plt.show()

plt.figure(21)
pd.Series(datos['k.1']).value_counts().sort_index().plot(kind = 'bar')
plt.title('spore-print-color')
plt.show()

plt.figure(22)
pd.Series(datos['s.3']).value_counts().sort_index().plot(kind = 'bar')
plt.title('population')
plt.show()

plt.figure(23)
pd.Series(datos['u']).value_counts().sort_index().plot(kind = 'bar')
plt.title('habitat')
plt.show()

# Separación de características y etiquetas
datos = np.array(datos)
datosy = np.transpose(datos)[0]
datosx = [i[1:] for i in datos]

# Eliminación de la característica 11
datosx = np.delete(datosx, 10, 1)

#Eliminación de la característica 16
datosx = np.delete(datosx, 14, 1)

#Selección de características
estimator = SVR(kernel="linear")
selector = RFE(estimator, 13)

datosx = selector.fit_transform(datosx, datosy)
print(selector.ranking_)
print(selector.support_)

#Separación de los datos en train y test
trainx, testx, trainy, testy = train_test_split(datosx, datosy, test_size=0.3, shuffle=True)

#Neural Networks
print('Resultados Redes Neuronales: ')
W = np.random.randint(low=0, high=2, size=len(testy))

auxW = []
for i in W:
    if i == 0:
        auxW.append(-1)
    else:
        auxW.append(i)
        
auxy = []
for i in testy:
    if i == 0:
        auxy.append(-1)
    else:
        auxy.append(i)

nn1 = MLPClassifier(hidden_layer_sizes=(10), alpha=1, activation='logistic', solver='lbfgs')
nn1.fit(trainx, trainy)

pred = nn1.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)

print('\nResultados para la función de activación sigmoide:')
        
scores = cross_val_score(nn1, trainx, trainy, cv=5)
print('Media de precisión con los valores de entrenamiento: ', statistics.mean(scores))

scores = cross_val_score(nn1, testx, testy, cv=5)
print('Media de precisión con los valores de prueba: ', statistics.mean(scores))

print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))

nn2 = MLPClassifier(hidden_layer_sizes=(11), alpha=1, activation='tanh', solver='lbfgs')
nn2.fit(trainx, trainy)


pred = nn2.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)
        
print('\nResultados para la función de activación tangente:')
        
scores = cross_val_score(nn2, trainx, trainy, cv=5)
print('Media de precisión con los valores de entrenamiento: ', statistics.mean(scores))

scores = cross_val_score(nn2, testx, testy, cv=5)
print('Media de precisión con los valores de prueba: ', statistics.mean(scores))

print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))

nn3 = MLPClassifier(hidden_layer_sizes=(11), alpha=1, activation='relu', solver='lbfgs')
nn3.fit(trainx, trainy)

pred = nn3.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)
        
print('\nResultados para la función de activación ReLu:')
        
scores = cross_val_score(nn3, trainx, trainy, cv=5)
print('Media de precisión con los valores de entrenamiento: ', statistics.mean(scores))

scores = cross_val_score(nn3, testx, testy, cv=5)
print('Media de precisión con los valores de prueba: ', statistics.mean(scores))

print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))

#Support Vector Machines (lineal)
input("Pulse una tecla para pasar al siguiente modelo")
svm = SVC(kernel='rbf', gamma='scale', max_iter=500)
svm.fit(trainx, trainy)

print('\nResultados SVM: ')
print('Precisión para los valores de entrenamiento: ', svm.score(trainx, trainy))
print('Precisión para los valores de prueba: ', svm.score(testx, testy))

pred = svm.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)

print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))

#Adaboost
input("Pulse una tecla para pasar al siguiente modelo")
clf = AdaBoostClassifier(n_estimators=25)
clf.fit(trainx, trainy)
print('\nResultados ADABOOST: ')
print('Precisión para los valores de entrenamiento: ', clf.score(trainx, trainy))
print('Precisión para los valores de prueba: ', clf.score(testx, testy))

pred = clf.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)
        
print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))

#Random Forest
input("Pulse una tecla para pasar al siguiente modelo")
rf = RandomForestClassifier(n_estimators=1, min_weight_fraction_leaf=0.0001, max_features=1, bootstrap=True)
rf.fit(trainx, trainy)

print('\nResultados RF: ')
print('Precisión para los valores de entrenamiento: ', rf.score(trainx, trainy))
print('Precisión para los valores de prueba: ', rf.score(testx, testy))

pred = rf.predict(testx)

auxp = []
for i in pred:
    if i == 0:
        auxp.append(-1)
    else:
        auxp.append(i)
        
print('Pérdida con un vector de etiquetas aleatorias: ', hinge_loss(auxy, auxW))
print('Pérdida con valores obtenidos: ', hinge_loss(auxy, auxp))
