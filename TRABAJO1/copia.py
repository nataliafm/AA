# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Natalia Fernández Martínez
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, exp, symbols, sin, cos

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Apartado 2\n')

#Expresión de E(u,v) con símbolos
def E():
    x, y = symbols('x y')
    expresion = (x**2 * exp(y) - 2 * y**2 * exp(-x))**2
    return expresion

#Expresión de E(u,v) con valores numéricos
def Ee(u,v):
    return (u**2 * np.exp(v) - 2 * v**2 * np.exp(-u))**2

def Eee(u,v):
    expresion = E()
    return expresion.subs([('x',u), ('y',v)])

#Derivada parcial de E con respecto a u
def dEu(u,v):
    Eu = E()
    derivada = diff(Eu, 'x')
    return derivada.subs([('x',u), ('y',v)])

#Derivada parcial de E con respecto a v
def dEv(u,v):
    Ev = E()
    derivada = diff(Ev, 'y')
    return derivada.subs([('x',u), ('y',v)])

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

eta = 0.01 #learning rate
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0, 1.0])

def gradient_descent():
    iteraciones = 0
    err = 1.0
    punto = initial_point
    while iteraciones < maxIter and err > error2get:
        aux = punto
        punto = punto - eta * gradE(punto[0], punto[1])
        iteraciones += 1
        err = abs(punto[1] - aux[1])

    return punto, iteraciones

w, it = gradient_descent()

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = Ee(X,Y) #E_w([X, Y])
fig = plt.figure(1)
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', zorder=0)
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], Eee(min_point_[0], min_point_[1]), 'r*', markersize=10, zorder=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 3\n")

def funcion():
    x, y = symbols('x y')
    expresion = x**2 + 2 * y**2 + 2 * sin(2 * np.pi * x) * sin(2 * np.pi * y)
    return expresion

#def func(x,y):
#    return x**2 + 2 * y**2 + 2 * sin(2 * np.pi * x) * sin(2 * np.pi * y)

def f(x,y):
    expresion = funcion()
    return expresion.subs([('x',x), ('y',y)])

#Derivada parcial de E con respecto a u
def dFx(x,y):
    Fx = funcion()
    derivada = diff(Fx, 'x')
    return derivada.subs([('x',x), ('y',y)])

#Derivada parcial de E con respecto a v
def dFy(x,y):
    Fy = funcion()
    derivada = diff(Fy, 'y')
    return derivada.subs([('x',x), ('y',y)])

#Gradiente de E
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

eta = 0.01 #learning rate
maxIter = 50
error2get = 1e-14
initial_point = [0.1, 0.1]
puntos = [[]]
puntos[0] = initial_point

def gradient_descent():
    iteraciones = 0
    punto = puntos[-1]
    while iteraciones < 50:
        punto = punto - eta * gradF(punto[0], punto[1])
        iteraciones += 1
        puntos.append(punto)

    return punto, iteraciones

w, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

puntos.pop()
valores = []
for i in puntos:
    valores.append(f(i[0], i[1]))

plt.figure(2)
plt.plot(range(50), valores)
plt.title('Ejercicio 1.2. Descenso del valor de la función con las iteraciones, η=0.01')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de f(x,y)')
plt.show()

eta = 0.1
puntos = [[]]
puntos[0] = initial_point

x, it = gradient_descent()

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', x[0], ', ', x[1],')')

puntos.pop()
valores = []
for i in puntos:
    valores.append(f(i[0], i[1]))

plt.figure(3)
plt.plot(range(50), valores)
plt.title('Ejercicio 1.2. Descenso del valor de la función con las iteraciones, η=0.1')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de f(x,y)')
plt.show()

eta = 0.01
puntos = [[]]
initial_point = [1.0, 1.0]
puntos[0] = initial_point

a, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', a[0], ', ', a[1],')')

puntos = [[]]
initial_point = [-0.5, -0.5]
puntos[0] = initial_point

b, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', b[0], ', ', b[1],')')

puntos = [[]]
initial_point = [1.0, 1.0]
puntos[0] = initial_point

c, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', c[0], ', ', c[1],')')

print ('\n\t\t| \t    x \t\t\t y \t\t\t f(x,y)')
print ('---------------------------------------------------------------------------------')
print ('[0.1, 0.1] \t| ', w[0], ' \t', w[1], ' \t ', f(w[0], w[1]))
print ('[1.0, 1.0] \t| ', a[0], ' \t', a[1], ' \t ', f(a[0], a[1]))
print ('[-0.5, -0.5] \t| ', b[0], ' \t', b[1], ' \t ', f(b[0], b[1]))
print ('[-1.0, -1.0] \t| ', c[0], ' \t', c[1], ' \t ', f(c[0], c[1]))

input("Pulse una tecla para pasar al siguiente ejercicio")
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Apartado 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    Xw = np.dot(x,w)
    xwy = Xw - y
    t = np.transpose(xwy)
    m = np.dot(t,xwy)
    return  m/len(x)

# Pseudoinversa
def pseudoinverse(x,y):
    XT = np.transpose(x)
    XTX = np.dot(XT, x)
    inversa = np.linalg.inv(XTX)
    ps = np.dot(inversa, XT)

    w = np.dot(ps,y)
    return w

# Gradiente Descendente Estocastico
def sgd(x,y):
    w = np.zeros(3)
    N = 32
    error = 1.0
    eta = 0.01
    peso_mejor = np.zeros(3)
    error_minimo = 1.0
    it = 0
    while error > 0.05 and it <= 100:
        batch = []
        valores = []
        for i in range(N):
            aux = np.random.choice(len(x))
            batch.append(x[aux])
            valores.append(y[aux])

        for i in range(len(batch)):
            pred = np.dot(batch, w)
            w = w - (1/len(batch)) * eta * (np.dot(np.transpose(batch), pred-valores))

        error = Err(batch,valores,w)

        if error < error_minimo:
            peso_mejor = w
            error_minimo = error

        it += 1
    if it < 100:
        return w
    else:
        return peso_mejor

# Función para clasificar
def clasificar(x,w):
    y = []
    for i in x:
        aux = np.dot(i,w)
        if aux > 0:
            y.append(label5)
        else:
            y.append(label1)
    return y

# Lectura de los datos de entrenamiento
x, y = readData('C:/Users/natal/Desktop/TERCERO/AA/TRABAJO1/datos/X_train.npy', '/Users/natal/Desktop/TERCERO/AA/TRABAJO1/datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('/Users/natal/Desktop/TERCERO/AA/TRABAJO1/datos/X_test.npy', '/Users/natal/Desktop/TERCERO/AA/TRABAJO1/datos/y_test.npy')

x_todos = np.concatenate((x,x_test))
y_todos = np.concatenate((y,y_test))

unosx, unosy = [], []
cincosx, cincosy = [], []
for i in range(len(x_todos)):
    if y_todos[i] == label1:
        unosx.append(x_todos[i][1])
        unosy.append(x_todos[i][2])
    else:
        cincosx.append(x_todos[i][1])
        cincosy.append(x_todos[i][2])

plt.figure(3)
plt.scatter(unosx, unosy, label='1')
plt.scatter(cincosx, cincosy, label='5')
plt.title('Mapa de todos los puntos')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.legend(loc="lower right")
plt.show()

w_pseudoinversa = pseudoinverse(x,y)

print ('Bondad del resultado para la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w_pseudoinversa))
print ("Eout: ", Err(x_test, y_test, w_pseudoinversa))

valores = clasificar(x_test, w_pseudoinversa)

unosx, unosy = [], []
cincosx, cincosy = [], []
for i in range(len(x_test)):
    if valores[i] == label1:
        unosx.append(x_test[i][1])
        unosy.append(x_test[i][2])
    else:
        cincosx.append(x_test[i][1])
        cincosy.append(x_test[i][2])

plt.figure(4)
plt.scatter(unosx, unosy, label='1')
plt.scatter(cincosx, cincosy, label='5')
plt.title('Clasificación de x_test usando la pseudoinversa')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.legend(loc="lower right")
plt.show()

w_SGD = sgd(x,y)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w_SGD))
print ("Eout: ", Err(x_test, y_test, w_SGD))

valores = clasificar(x_test, w_SGD)

unosx, unosy = [], []
cincosx, cincosy = [], []
for i in range(len(x_test)):
    if valores[i] == label1:
        unosx.append(x_test[i][1])
        unosy.append(x_test[i][2])
    else:
        cincosx.append(x_test[i][1])
        cincosy.append(x_test[i][2])

plt.figure(5)
plt.scatter(unosx, unosy, label='1')
plt.scatter(cincosx, cincosy, label='5')
plt.title('Clasificación de x_test usando Gradiente Descendente Estocástico')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetría')
plt.legend(loc="lower right")
plt.show()

w_SGD = sgd(x,y)

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2a")

def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

entrenamiento = simula_unif(1000, 3, 1)
for i in range(len(entrenamiento)):
    entrenamiento[i][0] = 1.0

x = []
y = []

for i in entrenamiento:
    x.append(i[1])
    y.append(i[2])

plt.figure(6)
plt.scatter(x, y)
plt.title('Mapa de los puntos')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2b")

def f(x1, x2):
    return np.sign( (x1 - 0.2)**2 + x2**2 - 0.6 )

etiquetas = []

for i in entrenamiento:
    etiquetas.append(f(i[1], i[2]))

for i in range(int(0.1 * len(entrenamiento))):
    indice = np.random.choice(1000)
    if etiquetas[indice] == 1: etiquetas[indice] = -1
    else: etiquetas[indice] = 1

ax, ay = [], []
bx, by = [], []
for i in range(len(entrenamiento)):
    if etiquetas[i] == label1:
        ax.append(entrenamiento[i][1])
        ay.append(entrenamiento[i][2])
    else:
        bx.append(entrenamiento[i][1])
        by.append(entrenamiento[i][2])

plt.figure(7)
plt.scatter(ax, ay, label='a')
plt.scatter(bx, by, label='b')
plt.title('Clasificación de los puntos usando la función f(x1,x2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2c")

entr_trans = np.copy(entrenamiento)

for i in range(len(entr_trans)):
    entr_trans[i] = entr_trans[i]**2

w = sgd(entr_trans, etiquetas)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(entr_trans, etiquetas, w))

etiquetas = clasificar(entr_trans, w)

ax, ay = [], []
bx, by = [], []
for i in range(len(entrenamiento)):
    if etiquetas[i] == label1:
        ax.append(entrenamiento[i][1])
        ay.append(entrenamiento[i][2])
    else:
        bx.append(entrenamiento[i][1])
        by.append(entrenamiento[i][2])

plt.figure(8)
plt.scatter(ax, ay, label='a')
plt.scatter(bx, by, label='b')
plt.title('Clasificación de los puntos usando una aproximación con SGD')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2d")

def funcion_d():
    entrenamiento = simula_unif(1000, 3, 1)

    for i in range(len(entrenamiento)):
        entrenamiento[i][0] = 1.0

    etiquetas = []

    for i in entrenamiento:
        etiquetas.append(f(i[1], i[2]))

    for i in range(int(0.1 * len(entrenamiento))):
        indice = np.random.choice(1000)
        if etiquetas[indice] == 1: etiquetas[indice] = -1
        else: etiquetas[indice] = 1

    entr_trans = np.copy(entrenamiento)

    for i in range(len(entr_trans)):
        entr_trans[i] = entr_trans[i]**2

    w = sgd(entr_trans, etiquetas)

    Ein = Err(entr_trans, etiquetas, w)

    prueba = simula_unif(1000, 3, 1)

    for i in range(len(prueba)):
        prueba[i][0] = 1.0

    prueba_trans = np.copy(prueba)

    for i in range(len(prueba_trans)):
        prueba_trans[i] = prueba_trans[i]**2

    et_prueba = []
    for i in prueba:
        et_prueba.append(f(i[1], i[2]))

    Eout = Err(prueba_trans, et_prueba, w)

    sol = np.array([Ein, Eout])

    return sol

Ein = 0.0
Eout = 0.0
for i in range(1000):
    error = funcion_d()
    Ein += error[0]
    Eout += error[1]
    print(i)

print('Ein medio para 1000 iteraciones: ', Ein/1000.0)
print('Eout medio para 1000 iteraciones: ', Eout/1000.0)
