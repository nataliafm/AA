# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

np.random.seed(2)

# Lectura de los datos
datos = []
datos = pd.read_csv('datos/agaricus-lepiota.data')

np.set_printoptions(threshold=sys.maxsize)

# Codificación de los datos para convertirlos de valores categóricos a numéricos
le = LabelEncoder()
for col in datos.columns:
    datos[col] = le.fit_transform(datos[col])
    
# Separación de características y etiquetas
datos = np.array(datos)
datosy = np.transpose(datos)[0]
datosx = [i[1:] for i in datos]

# Eliminación de la característica 11
datosx = np.delete(datosx, 10, 1)

#Selección de características
estimator = SVR(kernel="linear")
selector = RFE(estimator, None)

selector.fit_transform(datosx, datosy)
print(selector.ranking_)
print(selector.support_)

print(datosx[0])

#Separación de los datos en train y test
trainx, testx, trainy, testy = train_test_split(datosx, datosy, test_size=0.3, shuffle=True)


#print(np.transpose(datos)[10])