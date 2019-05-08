# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:16:16 2019

@author: natalia
"""
import numpy as np
import csv

trainx, trainy = [], []
testx, testy = [], []

with open('datos/optdigits.tra', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        trainx.append(row[:-1])
        trainy.append(int(row[-1]))

with open('datos/optdigits.tes', 'r') as csvFile:
    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        testx.append(row[:-1])
        testy.append(int(row[-1]))
        
