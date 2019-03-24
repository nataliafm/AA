# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Natalia Fernández Martínez
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, exp, symbols, sin

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

#Gradiente Descendente
def gradient_descent():
    iteraciones = 0
    err = 1.0
    punto = initial_point
    # itera mientras no se haya llegado al número máximo de iteraciones y
    #   la diferencia entre el punto actual y el anterior sea menor que 1e-14
    while iteraciones < maxIter and err > error2get:
        aux = punto
        punto = punto - eta * gradE(punto[0], punto[1]) #cálculo nuevo punto
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

# expresión de f(x,y) con símbolos
def funcion():
    x, y = symbols('x y')
    expresion = x**2 + 2 * y**2 + 2 * sin(2 * np.pi * x) * sin(2 * np.pi * y)
    return expresion

# expresión de f(x,y) con valores numéricos
def f(x,y):
    expresion = funcion()
    return expresion.subs([('x',x), ('y',y)])

#Derivada parcial de f(x,y) con respecto a x
def dFx(x,y):
    Fx = funcion()
    derivada = diff(Fx, 'x')
    return derivada.subs([('x',x), ('y',y)])
    
#Derivada parcial de f(x,y) con respecto a y
def dFy(x,y):
    Fy = funcion()
    derivada = diff(Fy, 'y')
    return derivada.subs([('x',x), ('y',y)])

#Gradiente de f(x,y)
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

eta = 0.01 #learning rate
initial_point = [0.1, 0.1]
puntos = [[]] # se van guardando los valores para hacer la gráfica
puntos[0] = initial_point

def gradient_descent():
    iteraciones = 0
    punto = puntos[-1]
    while iteraciones < 50:
        punto = punto - eta * gradF(punto[0], punto[1])
        iteraciones += 1
        puntos.append(punto)
        
    return punto, iteraciones

# gradiente descendente con η=0.01 y punto inicial [0.1, 0.1]
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

# gradiente descendente con η=0.1 y punto inicial [0.1, 0.1]
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

# gradiente descendente con η=0.01 y punto inicial [1.0, 1.0]
a, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', a[0], ', ', a[1],')')

puntos = [[]]
initial_point = [-0.5, -0.5]
puntos[0] = initial_point

# gradiente descendente con η=0.01 y punto inicial [-0.5, -0.5]
b, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', b[0], ', ', b[1],')')

puntos = [[]]
initial_point = [-1.0, -1.0]
puntos[0] = initial_point

# gradiente descendente con η=0.01 y punto inicial [-1.0, -1.0]
c, it = gradient_descent()
print ('Punto inicial: ', initial_point)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', c[0], ', ', c[1],')')

# tabla con los valores
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
    Xw = np.dot(x,w)      #Xw
    xwy = Xw - y          #Xw-y
    t = np.transpose(xwy) #(Xw-y)^T
    m = np.dot(t,xwy)     # (Xw-y)^T * (Xw-y)
    return  m/len(x)      # 1/N * (Xw-y)^T * (Xw-y) = 1/N * ||Xw-y||

# Pseudoinversa	
def pseudoinverse(x,y):
    XT = np.transpose(x)         # X^T
    XTX = np.dot(XT, x)          # X^T * X
    inversa = np.linalg.inv(XTX) # (X^T * X)^-1
    ps = np.dot(inversa, XT)     # (X^T * X)^-1 * X^T --> pseudoinversa de X
    w = np.dot(ps,y)             # w = pseudoinversa(x) * y
    return w                     # vector de pesos obtenido

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
        #generación de minibatch de N elementos
        batch = []
        valores = []
        for i in range(N):
            aux = np.random.choice(len(x))
            batch.append(x[aux])
            valores.append(y[aux])
        
        #cálculo de w para el minibatch
        for i in range(len(batch)):
            pred = np.dot(batch, w)
            w = w - (1/len(batch)) * eta * (np.dot(np.transpose(batch), pred-valores))
        
        #cálculo del error para el minibatch
        error = Err(batch,valores,w)

        if error < error_minimo: #se va conservando el mejor peso
            peso_mejor = w
            error_minimo = error
            
        it += 1
    if it < 100:
        return w
    else:                 #si no se ha llegado a una solución en todas las
        return peso_mejor # iteraciones, se devuelve el mejor valor obtenido

# Función para clasificar
def clasificar(x,w):
    y = []
    for i in x:
        aux = np.dot(i,w) # aux = w[0] + i[1]*w[1] + i[2]*w[2]
        if aux > 0:       # aux > 0 --> aux = 1
            y.append(label5)
        else:             # aux < 0 --> aux = -1
            y.append(label1)
    return y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

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

#obtención del vector de pesos usando el algoritmo de la pseudoinversa
w_pseudoinversa = pseudoinverse(x,y)

print ('Bondad del resultado para la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w_pseudoinversa))
print ("Eout: ", Err(x_test, y_test, w_pseudoinversa))

valores = clasificar(x_test, w_pseudoinversa) #asignación de etiquetas

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

#obtención del vector de pesos usando gradiente descendente estocástico
w_SGD = sgd(x,y)

print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x,y,w_SGD))
print ("Eout: ", Err(x_test, y_test, w_SGD))

valores = clasificar(x_test, w_SGD) # asignación de etiquetas

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

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2a")

def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

entrenamiento = simula_unif(1000, 3, 1) # crea 1000 puntos 2D entre [-1,1]x[-1,1]

#los puntos se crean con 3 dimensiones, pero el primer valor se cambia a 1.0
#   para tener el vector en la forma (1, x1, x2)
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
    
for i in range(int(0.1 * len(entrenamiento))): #introducción de 10% de ruido
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

#transformación no-lineal de los puntos de entrenamiento
for i in range(len(entr_trans)):
    entr_trans[i] = entr_trans[i]**2

w = sgd(entr_trans, etiquetas)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(entr_trans, etiquetas, w))

etiquetas = clasificar(entr_trans, w) #obtención de las etiquetas

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
    #crea los puntos de entrenamiento
    entrenamiento = simula_unif(1000, 3, 1)
    
    for i in range(len(entrenamiento)):
        entrenamiento[i][0] = 1.0
    
    #les asigna las etiquetas correspondientes con f(x1,x2)
    etiquetas = []

    for i in entrenamiento:
        etiquetas.append(f(i[1], i[2]))
        
    for i in range(int(0.1 * len(entrenamiento))):
        indice = np.random.choice(1000)
        if etiquetas[indice] == 1: etiquetas[indice] = -1
        else: etiquetas[indice] = 1
    
    #transformación no-lineal de los puntos
    entr_trans = np.copy(entrenamiento)

    for i in range(len(entr_trans)):
        entr_trans[i] = entr_trans[i]**2
    
    #cálculo de vector de pesos para los inputs transformados
    w = sgd(entr_trans, etiquetas)
    
    #obtención de Ein
    Ein = Err(entr_trans, etiquetas, w)
    
    #crea 1000 nuevos puntos de prueba
    prueba = simula_unif(1000, 3, 1)
    
    for i in range(len(prueba)):
        prueba[i][0] = 1.0
    
    #transforma los puntos de la misma manera que los de entrenamiento
    prueba_trans = np.copy(prueba)

    for i in range(len(prueba_trans)):
        prueba_trans[i] = prueba_trans[i]**2 
    
    #asigna a cada punto (sin transformar) su etiqueta con f(x1,x2)
    et_prueba = []
    for i in prueba:
        et_prueba.append(f(i[1], i[2]))
    
    #obtención de Eout con los valores transformados
    Eout = Err(prueba_trans, et_prueba, w)
    
    sol = np.array([Ein, Eout])
    
    return sol

Ein = 0.0
Eout = 0.0

#repetición del experimento 1000 veces
for i in range(1000):
    error = funcion_d()
    Ein += error[0]
    Eout += error[1]
    
print('Ein medio para 1000 iteraciones: ', Ein/1000.0)
print('Eout medio para 1000 iteraciones: ', Eout/1000.0)