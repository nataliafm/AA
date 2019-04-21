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

input("Pulse una tecla para pasar al siguiente ejercicio")
print("MODELOS LINEALES\n")
print("Apartado 1. Algoritmo Perceptron\n")

def ajusta_PLA(datos, label, max_iter, vini):
    #inicializar el vector de pesos y los contadores
    w = vini
    contador = 0
    num_cambios = 1
    
    #itera mientras que no se haya llegado al número máximo de iteraciones, 
    #   o hasta que todos los datos estén bien clasificados
    while contador < max_iter and num_cambios > 0:
        num_cambios = 0
        
        #en una iteración se recorren todos los datos
        for i in range(len(datos)):
            #si un dato está mal clasificado, se modifica el vector de pesos
            if np.sign(np.dot(np.transpose(w),datos[i])) != label[i]:
                w = w + label[i] * datos[i]
                num_cambios += 1
                
        contador += 1

    return w, contador

# modificar el vector de puntos del apartado 2a de la sección 1 para añadir la 
#   columna de unos en la posición 0 
puntos = np.transpose(puntos)
unos = np.ones(len(puntos[0]))

puntos = np.insert(puntos, 0, unos, axis=0)
puntos = np.transpose(puntos)
    
print("Apartado 1a\n")
print("(a)")

#inicializar el vector de pesos con ceros
w = np.zeros(len(puntos[0]))

#obtener el vector resultante, usando 10000 iteraciones como tope
pesos, iteraciones = ajusta_PLA(puntos, predicciones, 10000, w)

print("Número de iteraciones necesarias: ", iteraciones, "\n")

print("(b)")

suma_iteraciones = 0
for j in range(10):
    #inicializar el vector de pesos con valores aleatorios en el intervalo [0,1]
    w = np.random.rand(len(puntos[0]))
    
    #obtener el vector resultante, usando 10000 iteraciones como tope
    pesos, iteraciones = ajusta_PLA(puntos, predicciones, 10000, w)
    
    suma_iteraciones += iteraciones
    
suma_iteraciones = suma_iteraciones / 10

print("Número de iteraciones necesarias: ", suma_iteraciones, "\n")

pos, neg = [], []
for i in range(len(predicciones)):
    if predicciones[i] > 0.0: pos.append(puntos[i])
    else: neg.append(puntos[i]) 
    
pos = np.transpose(pos)
posx, posy = pos[1], pos[2]

neg = np.transpose(neg)
negx, negy = neg[1], neg[2]

a = -(pesos[0]/pesos[2])/(pesos[0]/pesos[1])
b = -pesos[0]/pesos[2]

x = np.linspace(-50,50,100)
y = a*x + b

#Generación del gráfico
plt.figure(9)
plt.plot(x,y)
plt.scatter(posx, posy, label="positivos")
plt.scatter(negx, negy, label="negativos")
plt.title("Frontera obtenida mediante perceptron ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 1b\n")

#unir en el vector puntos los datos del apartado 2b de la sección 1
puntos = np.concatenate((positivos, negativos))

predicciones = []
for i in range(len(positivos)):
    predicciones.append(1.0)
    
for i in range(len(negativos)):
    predicciones.append(-1.0)

#modificar el vector de puntos para añadir la primera columna de unos
puntos = np.transpose(puntos)
unos = np.ones(len(puntos[0]))

puntos = np.insert(puntos, 0, unos, axis=0)
puntos = np.transpose(puntos)

print("(a)")

#inicializar el vector de pesos con ceros
w = np.zeros(len(puntos[0]))

#obtener el vector resultante, usando 10000 iteraciones como tope
pesos, iteraciones = ajusta_PLA(puntos, predicciones, 10000, w)

print("Número de iteraciones necesarias: ", iteraciones, "\n")

print("(b)")

suma_iteraciones = 0
for j in range(10):
    #inicializar el vector de pesos con valores aleatorios en el intervalo [0,1]
    w = np.random.rand(len(puntos[0]))
    
    #obtener el vector resultante, usando 10000 iteraciones como tope
    pesos, iteraciones = ajusta_PLA(puntos, predicciones, 10000, w)
    
    suma_iteraciones += iteraciones
    
suma_iteraciones = suma_iteraciones / 10

print("Número de iteraciones necesarias: ", suma_iteraciones, "\n")

positivos = np.transpose(positivos)
unos = np.ones(len(positivos[0]))
positivos = np.insert(positivos, 0, unos, axis=0)
posx, posy = positivos[1], positivos[2]

negativos = np.transpose(negativos)
unos = np.ones(len(negativos[0]))
negativos = np.insert(negativos, 0, unos, axis=0)
negx, negy = negativos[1], negativos[2]

a = -(pesos[0]/pesos[2])/(pesos[0]/pesos[1])
b = -pesos[0]/pesos[2]

x = np.linspace(-50,50,100)
y = a*x + b

#Generación del gráfico
plt.figure(10)
plt.plot(x,y)
plt.scatter(posx, posy, label="positivos")
plt.scatter(negx, negy, label="negativos")
plt.title("Frontera obtenida mediante perceptron ")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2a\n")

#función que calcula la función logística
def sigmoide(X):
    return 1/(np.exp(-X)+1)

#función que calcula la probabilidad de que X sea positivo
def probabilidad(X, w):
    return sigmoide(np.dot(X, w))

#función que modifica el vector de pesos
def actualizar_pesos(X, y, w, eta):
    #se calcula la probabilidad de que X sea positivo
    predicciones = probabilidad(X, w)
    #se calcula el gradiente 
    gradiente = np.dot(np.transpose(X), predicciones - y)
    #se resta al vector de pesos la media para cada elemento de X de multiplicar
    #   el gradiente por el learning rate
    w -= gradiente * eta / len(X)
    return w

eta = 0.1

#obtener los puntos
X = simula_unif(100, 2, (0,2))
#insertar una columna de unos en primer lugar
X = np.transpose(X)
unos = np.ones(len(X[0]))
X = np.insert(X, 0, unos, axis=0)
X = np.transpose(X)

#generar dos puntos aleatorios del dominio y la recta que pasa por ellos
punto1 = simula_unif(1, 2, (0,2))
punto2 = simula_unif(1, 2, (0,2))

a = (punto2[0][1] - punto1[0][1]) / (punto2[0][0] - punto1[0][0])
b = X[0][2] - a * X[0][1]

#obtener las etiquetas para cada punto en función de la recta generada
y = []
for i in range(len(X)):
    valor = f(X[i][1], X[i][2])
    if valor == 1.0: y.append(valor)
    else: y.append(0.0) 

#inicializar el vector de pesos a 0
w = np.zeros(3)

#función que calcula el vector de pesos para una muestra X con unas etiquetas y
# usando regresiçon logística implementada con SGD
def SGD_LG(X, y, w):
    num_iteraciones = 0
    pesos = w
    pesos_anterior = np.ones(3)
    resta = pesos_anterior - pesos
    
    X_aux, y_aux = np.copy(X), np.copy(y)
    
    #itera mientras ||w(t−1)−w(t)|| sea mayor que 0.01
    while np.linalg.norm(resta) > 0.01:
        #guarda el vector de pesos para hacer la comprobación del bucle
        pesos_anterior = np.copy(pesos)
        
        #hace una permutación aleatoria en el orden de los datos
        #usando la variable estado para hacer la misma permutación en los dos vectores
        estado = np.random.get_state()
        np.random.shuffle(X_aux)
        np.random.set_state(estado)
        np.random.shuffle(y_aux)
        
        i = 16
        #recorre los datos de X en mini-batches de 16 elementos
        while (i < len(X)):
            #obtiene los elementos del mini-batch
            batch_x, batch_y = X_aux[i-16:i], y_aux[i-16:i]
            #los usa para actualizar el vector de pesos
            pesos = actualizar_pesos(batch_x, batch_y, pesos, eta)
            i += 16
        
        resta = pesos_anterior - pesos
        num_iteraciones += 1
    
    return pesos, num_iteraciones

#calcula el vector de pesos para los datos generados
pesos, iteraciones = SGD_LG(X, y, w)     
  
print("Número de iteraciones necesarias: ", iteraciones, "\n")

pos, neg = [], []
for i in range(len(y)):
    if y[i] == 1.0: pos.append(X[i])
    else: neg.append(X[i]) 
    
pos = np.transpose(pos)
posx, posy = pos[1], pos[2]

neg = np.transpose(neg)
negx, negy = neg[1], neg[2]

x = np.linspace(0,2,100)
y = (-pesos[0]-pesos[1]*x)/pesos[2]

#Generación del gráfico
plt.figure(11)
plt.plot(x,y)
plt.scatter(posx, posy, label="positivos")
plt.scatter(negx, negy, label="negativos")
plt.title("Frontera obtenida mediante SGD")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()

input("Pulse una tecla para pasar al siguiente apartado")
print("Apartado 2b\n")

# Funcion para calcular el error
def Err(h_x, y):
    dif = np.subtract(h_x, y)
    return float(np.count_nonzero(dif)) / len(dif)

#obtener los puntos de prueba
prueba_x = simula_unif(1000, 2, (0,2))
#insertar una columna de unos en primer lugar
prueba_x = np.transpose(prueba_x)
unos = np.ones(len(prueba_x[0]))
prueba_x = np.insert(prueba_x, 0, unos, axis=0)
prueba_x = np.transpose(prueba_x)

#obtener las etiquetas correspondientes a los puntos
prueba_y = []
for i in range(len(prueba_x)):
    valor = f(prueba_x[i][1], prueba_x[i][2])
    if valor == 1.0: prueba_y.append(valor)
    else: prueba_y.append(0.0)

#calcular las predicciones para los puntos de prueba usando el vector de pesos
#   generado antes
predicciones = []
for i in range(len(prueba_x)):
    pred = np.dot(np.transpose(pesos), prueba_x[i])
    if pred > 0.5: predicciones.append(1.0)
    else: predicciones.append(0.0)
    
#cálculo del error fuera de la muestra
Eout = Err(predicciones, prueba_y)
print("Eout: ", Eout)

pos, neg = [], []
for i in range(len(predicciones)):
    if predicciones[i] == 1.0: pos.append(prueba_x[i])
    else: neg.append(prueba_x[i]) 
    
pos = np.transpose(pos)
posx, posy = pos[1], pos[2]

neg = np.transpose(neg)
negx, negy = neg[1], neg[2]

#Generación del gráfico
plt.figure(12)
plt.plot(x,y)
plt.scatter(posx, posy, label="positivos")
plt.scatter(negx, negy, label="negativos")
plt.title("Frontera obtenida mediante SGD")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="lower right")
plt.show()
