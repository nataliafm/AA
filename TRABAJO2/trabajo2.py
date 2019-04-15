# -*- coding: utf-8 -*-
"""
@author: Natalia Fernández Martínez
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)

print("EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n")

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

print("Apartado 1a\n")

#Generación de los puntos
puntos = simula_unif(50, 2, [-50, 50])

#Separar las x y las y de los puntos en 2 vectores distintos para hacer el gráfico
puntos_aux = np.transpose(puntos)

puntosx = puntos_aux[0]
puntosy = puntos_aux[1]

#Generación del gráfico
plt.figure(1)
plt.scatter(puntosx, puntosy)
plt.title("Nube de puntos generada con simula_unif, N=50, dim=2 y rango=[-50,50]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 1b\n")

#Generación de los puntos
puntos = simula_gaus(50, 2, [5, 7])

#Separar las x y las y de los puntos en 2 vectores distintos para hacer el gráfico
puntos_aux = np.transpose(puntos)

puntosx = puntos_aux[0]
puntosy = puntos_aux[1]

#Generación del gráfico
plt.figure(2)
plt.scatter(puntosx, puntosy)
plt.title("Nube de puntos generada con simula_gaus, N=50, dim=2 y rango=[5,7]")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2a\n")

#Obtener los valores de la recta
a,b = simula_recta([-50, 50])

#Función que clasifica puntos en función de la recta anterior
def f(x,y):
    return np.sign(y - a*x - b)

#Generación de los puntos
puntos = simula_unif(50, 2, [-50, 50])

#Se calcula el valor de f(x,y) para cada punto, y se añade al subvector de positivos
#   o de negativos en función de este valor
predicciones = []
positivos, negativos = [], []
for i in puntos:
    valor = f(i[0], i[1])
    predicciones.append(valor)
    
    if valor > 0.0: positivos.append(i)
    else: negativos.append(i)

#Separar las x y las y de los puntos en 2 vectores distintos para hacer el gráfico
negativost = np.transpose(negativos)
nx, ny = negativost[0], negativost[1]

positivost = np.transpose(positivos)
px, py = positivost[0], positivost[1]

#Obtener los puntos necesarios para poder pintar la recta
x = np.linspace(-50,50,100)
y = a*x + b

#Generación del gráfico
plt.figure(3)
plt.plot(x,y)
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2b\n")

#Desordenar aleatoriamente los puntos positivos y los negativos
np.random.shuffle(positivos)
np.random.shuffle(negativos)

pos_aux, neg_aux = [], []

#Se toma el 10% de puntos positivos y el 10% de puntos negativos, se guardan
#   en subvectores auxiliares y se eliminan de sus vectores originales
if len(positivos)*0.1 > 1:
    for i in range(int(len(positivos)*0.1)):
        index = np.random.choice(len(positivos))
        pos_aux.append(positivos[index])
        positivos = np.delete(positivos, index, 0)

if len(negativos)*0.1 > 1:
    for i in range(int(len(negativos)*0.1)):
        index = np.random.choice(len(negativos))
        neg_aux.append(negativos[index])
        negativos = np.delete(negativos, index, 0)

#Se unen los puntos positivos con el 10% de puntos negativos, y los puntos
#   negativos con el 10% de puntos positivos
if neg_aux != []: positivos = np.concatenate((positivos, np.asarray(neg_aux)))
if pos_aux != []: negativos = np.concatenate((negativos, np.asarray(pos_aux)))

#Separar las x y las y de los puntos en 2 vectores distintos para hacer el gráfico
negativost = np.transpose(negativos)
nx, ny = negativost[0], negativost[1]

positivost = np.transpose(positivos)
px, py = positivost[0], positivost[1]

#Obtener los puntos necesarios para poder pintar la recta
x = np.linspace(-50,50,100)
y = a*x + b

#Generación del gráfico
plt.figure(4)
plt.plot(x,y)
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b con ruido")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 3\n")

def f1(x,y):
    return (x-10)**2 + (y-20)**2 - 400

def f2(x,y):
    return 0.5 * (x+10)**2 + (y-20)**2 - 400

def f3(x,y):
    return 0.5 * (x-10)**2 - (y+20)**2 - 400

def f4(x,y):
    return y - 20*x**2 - 5*x + 3

recta1, recta2, recta3, recta4 = [], [], [], []

x1 = np.linspace(-50, 50, 100)
y1 = np.linspace(-50, 50, 100)
x1, y1 = np.meshgrid(x1, y1)
recta1 = f1(x1,y1)

x2 = np.linspace(-50, 50, 100)
y2 = np.linspace(-50, 50, 100)
x2, y2 = np.meshgrid(x2, y2)
recta2 = f2(x2,y2)

x3 = np.linspace(-50, 50, 100)
y3 = np.linspace(-50, 50, 100)
x3, y3 = np.meshgrid(x3, y3)
recta3 = f3(x3,y3)

x4 = np.linspace(-50, 50, 100)
y4 = np.linspace(-50, 50, 100)
x4, y4 = np.meshgrid(x4, y4)
recta4 = f4(x4,y4)

#Generación del gráfico
plt.figure(5)
plt.contour(x1,y1,recta1,[0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

#Generación del gráfico
plt.figure(6)
plt.contour(x2,y2,recta2,[0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

#Generación del gráfico
plt.figure(7)
plt.contour(x3,y3,recta3,[0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

#Generación del gráfico
plt.figure(8)
plt.contour(x4,y4,recta4,[0])
plt.scatter(nx, ny, label="positivos")
plt.scatter(px, py, label="negativos")
plt.title("Puntos separados por la recta y=ax+b ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()
