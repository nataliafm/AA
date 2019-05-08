# -*- coding: utf-8 -*-

import numpy as np
import csv
# Lectura de los datos, separación de estos en train y test, y separación de
    # características y etiquetas
trainx, trainy = [], []
testx, testy = [], []

with open('datos/optdigits.tra', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = list(map(int, row))
        trainx.append(row[:-1])
        trainy.append(row[-1])

with open('datos/optdigits.tes', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = list(map(int, row))
        testx.append(row[:-1])
        testy.append(row[-1])

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
