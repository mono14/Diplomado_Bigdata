# -*- coding: utf-8 -*-
"""

@author: MILITUS
"""
# 
# Manipulación de datos con numpy y pandas que permiten manejar objetos de tipo array (matriz) en python
# En python el tipado de datos es dinámico - Todos los tipos en realidad son clases
# Visualización de datos con matplolib
# Análisis Exploratorio básico - Qué es el AE -Cual es el mínimo necesario para hacer exploración de datos
# Análisis en componentes principales ACP 


'''Paquetes de GRAFICACIÓN matplotlib y seaborn
Generalmente se suele importar matplotlib de la siguiente forma.'''
import matplotlib.pyplot as plt
import seaborn
# NumPy es un paquete de Python que significa “Numerical Python”, es la librería principal para la informática científica, proporciona potentes estructuras de datos, implementando matrices y matrices multidimensionales. Estas estructuras de datos garantizan cálculos eficientes con matrices.
import numpy as np # carga el paquete

x=np.array([0,1,2,3,4,5])
y=np.array([0,3,7,13,21,32])
# Se genera gráfica
plt.plot(x,y)
# Visualizamos la gráfica
plt.show()

# Creación de múltiples grágicas (Subplot)

def f1(x):
    return np.exp(-x) * np.sin(x)+3 * np.cos(x)
def f2(x):
    return 4 * np.exp(-1/(5*x)) * np.cos(x)+0.5 * np.cos(5/x)
# Creamos dos arreglos que serán los ejes de X1, X2
x1= np.arange(0.1,5.0,0.1)
x2= np.arange(0.1,5.0,0.02)

#Creamos una figura en una ventana
plt.figure(1)
''' Configuramos el (Subplot1)
en el Subplot(211): eL 2 quiere decir que hay dos filas cada una con una figura
El número 1 quiere decir que hay una columna en la ventana y el último 1 corres
ponde al número del Subplot (Subplot1, Subplot2 etc.)
'''
plt.subplot(211)
# Configuramos la gráfica con los ejes x, y 
plt.plot(x1,f1(x1))
# Lo mismo para el (subplot2)
plt.subplot(212)
plt.plot(x2,f2(x2))
 

# Ahora suponemos que necesitamos crear varias gráficas en varias ventanas
def f1(x):
    return np.exp(-x) * np.sin(x)+3 * np.cos(x)
def f2(x):
    return 4 * np.exp(-1/(5*x)) * np.cos(x)+0.5 * np.cos(5/x)

x1= np.arange(0.1,5.0,0.1)
x2= np.arange(0.1,5.0,0.02)
# Se crea la otra figura
plt.figure(1)
plt.subplot(211)
# Configuramos la gráfica con los ejes x, y 
plt.plot(x1,f1(x1))
# Lo mismo para el (subplot2)
plt.subplot(212)
plt.plot(x2,f2(x2))

# Creamos una segunda ventana
plt.figure(2)
# Configuramos Subplot1
plt.subplot(321)
# Configuramos la gráfica con los ejes  [X,Y]
plt.plot(x1,1/f1(x1))
# Lo mismo par el subolot2
plt.subplot(322) 
plt.plot(x2,2*f2(x2))

# Lo mismo para el Subplot3
plt.subplot(323)
plt.plot(x2,f2(x2))
plt.show()

# Editando las profpiedades de las gráficas
x=np.array([0,1,2,3,4,5])
y1=np.array([0,3,7,13,21,32])
y2=np.array([32,-1,9,-4,12,0])
y3=np.array([-32,0,32,16,8,21])
# Generamos la gráfica
plt.plot(x,y1,x,y2,x,y3)
# Mostramos en pantalla
plt.show()
# Ahora si cambiamos
plt.plot(x,y1,'bo',x,y2,'r--',x,y3,'k')

# Damos nombres a los ejes (x,y)
plt.xlabel('Eje X',fontsize=28)
plt.ylabel('Eje Y',fontsize=28)
# Título de la grágica en formato laTEX
plt.title(r'$\alpha_i > \beta_i$', fontsize=28)
# Añadimos texto y notificacciones en formato laTEX
# Notese que los primeros parámetros corresponden a la posiciòn (x,y) dentro de los vectores
# para crear la gráfica que fueron declardos previamente

plt.text(3,7,r'$\sum {i=0}^\infty x i$',fontsize=20)

# Añadimos una segunda ecuación
plt.text(0,-32,r'$\mathcal{A}\mathrm{sin} (2 \omega t)$',fontsize=20)
# Agregamos una anotación, el parámetro 'xy=(1,3)' denotan la posición del punto sobre el cual
#  quermeos hacer la anotación 'xytext' es la posición donde estará el texto
# 'arrowprops' permitirá editar el estilo de la flecha
plt.annotate('Anotación', xy=(1,3), xytext=(2,-4),arrowprops=dict(facecolor='black',shrink=0.05))
# Ahora creamos y parametrizamos el 'legend' para el primer eje 
x=np.array([0,1,2,3,4,5])
y1=np.array([0,3,7,13,21,32])
y2=np.array([32,-1,9,-4,12,0])

p1,= plt.plot(x,y1,'bo',label="texto1")
# Lo mismo para el segundo eje, es probable que la siguente instgrucción remueva l1
p2,= plt.plot(x,y2,'r--',label="texto2")
l1 = plt.legend([p1], ["Label 1"],loc=1)
l2 = plt.legend([p2],["Label 2"],loc=2)
# Para volver a visualizar el primer eje 
plt.gca().add_artist(l1)
plt.show()

# Otro ejemplo utilizando scatter() 
def f(x):
    return np.exp(-x ** 2)

#Creamos un vector con los puntos que le pasaremos a la funcion previamente creada.
x = np.linspace(-1, 5, num=30)
#Representamos la función utilizando el objeto plt de matplotlib
plt.xlabel("Eje $x$")
plt.ylabel("$f(x)$")
plt.legend()
plt.title("Funcion $f(x)$")
plt.grid(True)
fig = plt.plot(x, f(x), label="Función f(x)")

# Gráfico de puntos con matplotlib y la función scatter()
# Un diagrama de dispersión (o gráfico de dispersión ) es un gráfico de dos 
# dimensiones donde cada dato se representa como un punto que representa los 
# valores de dos conjuntos de variables cuantitativas, uno a lo largo del eje x
#  y el otro a lo largo del eje y .
N = 100
x1 = np.random.randn(N) #creando vector x
y1 = np.random.randn(N) #creando vector x
print(x1)
print(y1)
s = 50 + 50 * np.random.randn(N) # variable para modificar el tamaño(size)
c = np.random.randn(N) # variable para modificar el color(color)

plt.scatter(x1, y1, s=s, c=c, cmap=plt.cm.Blues) 
plt.grid(True)
plt.colorbar()

fig = plt.scatter(x1, y1)
# ===================
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
# =================
x = np.random.randn(50)
y = np.random.randn(50)
plt.scatter(x, y, color='r', s=30)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Graficación de puntos utilizando Scatter Plot')
plt.show()
# ==================
n=50
x = np.random.rand(n)
y = np.random.rand(n)
colors = np.random.rand(n)
size =pow(20 * np.random.rand(n),2)
plt.scatter(x, y, s=size, c=colors, alpha=0.5)
plt.show()

# ==========================
# importar todas las funciones de pylab y pylot
from pylab import *
# importar el módulo pyplot
import matplotlib.pyplot as plt
x = arange(15)          # array de floats, de 0.0 a 9.0
plt.plot(x)                  # generar el gráfico de la función y=x
plt.show()                   # mostrar el gráfico en pantalla
plt.ion()             # Activo el modo interactivo para que cada cambio que se haga en la gráfica se muestre en el momento
plt.plot(x)           # Hago un plot que se muestra sin hacer show()

import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *

mi_dibujo, = plot(x)   # Nótese que después de mi_dibujo hay una coma; esto es para indicar 
# que mi_dibujo debe tomar el valor del primer (y en este caso el único) elemento de la lista 
# y no la lista en sí
plt.plot(x,'o')  # pinta 10 puntos como o

plt.plot(x,'o-') # igual que antes pero ahora los une con una linea continua
mi_dibujo, = plot(x*2,'c-', hold=True)
'''
Colores
Símbolo	Color
“b”	Azul
“g”	Verde
“r”	Rojo
“c”	Cian
“m”	Magenta
“y”	Amarillo
“k”	Negro
“w”	Blanco
Marcas y líneas

Símbolo	Descripción
“-“	Línea continua
“–”	Línea a trazos
“-.”	Línea a puntos y rayas
“:”	Línea punteada
“.”	Símbolo punto
“,”	Símbolo pixel
“o”	Símbolo círculo relleno
“v”	Símbolo triángulo hacia abajo
“^”	Símbolo triángulo hacia arriba
“<”	Símbolo triángulo hacia la izquierda
“>”	Símbolo triángulo hacia la derecha
“s”	Símbolo cuadrado
“p”	Símbolo pentágono
“*”	Símbolo estrella
“+”	Símbolo cruz
“x”	Símbolo X
“D”	Símbolo diamante
“d”	Símbolo diamante delgado
'''
clf()   # Limpiamos toda la figura
x2=x**2   # definimos el array x2
x3=x**3    # definimos el array x3
# dibujamos tres curvas en el mismo gráfico y figura
plot(x, x,'b.', x, x2,'rd', x, x3,'g^')
'''
Además del marcador y el color indicado de la manera anterior, se pueden cambiar muchas otras propiedades de la gráfica como parámetros de plot() independientes como los de la tabla adjunta:

Parámetro	          Significado y valores
alpha	                grado de transparencia, float (0.0=transparente a 1.0=opaco)
color o c	            Color de matplotlib
label	                Etiqueta con cadena de texto, string
markeredgecolor o mec	Color del borde del símbolo
markeredgewidth o mew	Ancho del borde del símbolo, float (en número de puntos)
markerfacecolor o mfc	Color del símbolo
markersize o ms	        Tamaño del símbolo, float (en número de puntos)
linestyle o ls	        Tipo de línea, “-“ “–” “-.” “:” “None”
linewidth o lw	        Ancho de la línea, float (en número de puntos)
marker	                Tipo de símbolo,”+” “*” “,” “.” “1” “2” “3” “4” “<” “>” “D” “H” “^” “_” “d” “h” “o” “p” “s” “v” “x” “|” TICKUP TICKDOWN TICKLEFT TICKRIGHT
'''
plt.plot(x, lw=5, c='y', marker='o', ms=10, mfc='red')
# ===============================
# Trabajando con texto dentro del gráfico con log()

x = arange(0, 5, 0.05)
p, = plot(x,log10(x)*sin(x**2))
xlabel('Eje X')        # Etiqueta del eje OX

ylabel('Eje Y')        # Etiqueta del eje OY

title('Mi grafica')    # Título del gráfico
text(1, -0.4, 'Nota')  # Texto en coodenadas (1, -0.4)
# ===========================================
from math import *
from numpy import *

t = arange(0.1, 20, 0.1)

y1 = sin(t)/t
y2 = sin(t)*exp(-t)
p1, p2 = plot(t, y1, t, y2)

# Texto en la gráfica en coordenadas (x,y)
texto1 = text(2, 0.6, r'$\frac{\sin(t)}{t}$', fontsize=20)
texto2 = text(13, 0.2, r'$\sin(t) \cdot e^{-t}$', fontsize=16)

# Añado una malla al gráfico
grid()

title('Representación de dos funciones')
xlabel('Tiempo / s')
ylabel('Amplitud / cm')
# Punto a señalar en la primera gráfica
px = 7.5
py = sin(px)/px

# Pinto las coordenadas con un punto negro
punto = plot([px], [py], 'bo')

# Hago un señalización con flecha
nota = plt.annotate(r'$\frac{\sin(7.5)}{\exp(-7.5)} = 0.12$',
         xy=(px, py), xycoords='data',
         xytext=(3, 0.4), fontsize=9,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
show()

# ==================================
#  Representación gráfica de funciones
def f1(x):
    y = sin(x)
    return y

def f2(x):
    y = sin(x)+sin(5.0*x)
    return y


def f3(x):
    y = sin(x)*exp(-x/10.)
    return y

# array de valores que quiero representar
x = linspace(0, 10*pi, 800)

p1, p2, p3 = plot(x, f1(x), x, f2(x), x, f3(x))

# Añado leyenda, tamaño de letra 10, en esquina superior derecha
# legend(('Funcion 1', 'Funcion 2', 'Funcion 3'), prop = {'size':10}, loc = 'upper right')

# Plots con label
p1 = plot(x, f1(x), label='Funcion 1')
p2 = plot(x, f2(x), label='Funcion 2')
p3 = plot(x, f3(x), label='Funcion 3')
legend(loc='lower right')
xlabel('Tiempo / s')
ylabel('Amplitud / cm')
title('Representacion de tres funciones')

# Creo una figura (ventana), pero indico el tamaño (x,y) en pulgadas
figure(figsize=(12, 5))

show()
# ====================================
# Histogramas
# En Python podemos hacer histogramas muy fácilmente con la función hist() indicando como parámetro un array con los números del conjunto.
# Importamos el módulo de numeros aleatorios de numpy
from numpy import random

# utilizo la función randn() del modulo random para generar
# un array de números aleatorios con distribución normal
nums = random.randn(200)  # array con 200 números aleatorios
xlabel('Valor ')
ylabel('Número ')

# Genero el histograma
h = hist(nums)

# los números del array se dividieron automáticamente en 10 intervalos (o bins) 
(array([ 2, 10, 11, 28, 40, 49, 37, 12,  6,  5]), array([-2.98768497, -2.41750815, -1.84733134, -1.27715452, -0.70697771,
    -0.13680089,  0.43337593,  1.00355274,  1.57372956,  2.14390637,
    2.71408319]))

hist(nums, bins=20)
legend(('Histograma 1', 'Histograma 2'), prop = {'size':10}, loc = 'upper right')


# =========================
# Gráficos en tercera dimensión (3D) con Python y Matplotlib - Ejemplos Prácticoss
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

# Creamos la figura
fig = plt.figure()

# Agregamos un plano 3D
ax1 = fig.add_subplot(111,projection='3d')

# Mostramos el gráfico
plt.show()
# _______________________________

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

# Datos en array bi-dimensional
x = np.array([[1,2,3,4,5,6,7,8,9,10]])
y = np.array([[5,6,7,8,2,5,6,3,7,2]])
z = np.array([[1,2,6,3,2,7,3,3,7,2]])

# plot_wireframe nos permite agregar los datos x, y, z. Por ello 3D
# Es necesario que los datos esten contenidos en un array bi-dimensional
ax1.plot_wireframe(x, y, z)

# Mostramos el gráfico
plt.show()
# =================================
# Scatter en Tercera Dimensión
# Podemos graficar scatter utilizando la misma tecnica anterior. En este caso, no sera necesario utilizar el método plot_wireframe.
# Scatter

# importamos las librerias necesarias
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

# Definimos los datos de prueba
x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

# Datos adicionales
x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
z2 = [1,2,6,3,2,7,3,3,7,2]

# Agregamos los puntos en el plano 3D
ax1.scatter(x, y, z, c='g', marker='o')
ax1.scatter(x2, y2, z2, c ='r', marker='o')

# Mostramos el gráfico
plt.show()
# ================================
# Barras en Tercera Dimensión
# Con el método bar3d podemos generar barras 3D de manera muy sencilla
# Importamos los modulos necesarios
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
 
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# Definimos los datos
x3 = [1,2,3,4,5,6,7,8,9,10]
y3 = [5,6,7,8,2,5,6,3,7,2]
z3 = np.zeros(10)

dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

# utilizamos el método bar3d para graficar las barras
ax1.bar3d(x3, y3, z3, dx, dy, dz)

# Mostramos el gráfico
plt.show()
# ==========================
# GRÁFICOS EN 3D en espirales con Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Ejemplo 1
fig = plt.figure()
ax = Axes3D(fig)
# Datos para lalinea en 3D
zline=np.linspace(0,15,1000)
xline=np.sin(zline)
yline=np.cos(zline)
ax.plot3D(xline, yline, zline,'gray')
# ===============================
'''Mapas de calor con Seaborn
Un mapa de calor es una representación gráfica de los valores contenidos en una matriz mediante el uso de colore
Siendo una herramienta excelente para para mostrar las relaciones existentes entre las variables de diferentes características,
ya que al mostrar la relación mediante un color se obtiene una interpretación fácil e intuitiva
 Importación de las librerías
'''
import seaborn as sns
import pandas as pd
import numpy as np
# Creación de un conjunto de datos aleatorio
data = pd.DataFrame(np.random.random((5, 5)))
# Representación del mapa de calor
sns.heatmap(data, center=0, cmap='Blues_r', annot=True, fmt='.3f')
'''
Esta es la matriz que se representa, para lo que se utiliza la función heatmap de seaborn. La función únicamente 
necesita la matriz que contiene los valores a representar, aunque se puede indicar otros parámetros para personalizar el resultado. 
center: el valor en el cual centrar el mapa de color al representar los datos.
cmap: indica el mapa que se utilizará para la representación de los valores,
annot: indica si se representa o no la magnitud de cada celda en el mapa además del color, por defecto no se representará
fmt: es el formato con el que se representará la magnitud.
'''


# =========================================
# Diagrama de araña en Python
'''
Los diagramas de araña son una de las mejores maneras para mostrar los valores que toman varias variables
con una magnitud En el caso de que se desee ver cómo se agrupan varias categorías respecto a las mismas 
características se pueden mostrar unas respecto a otras.
'''

# Importación de las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Creación de un conjunto de datos aleatorio
angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
values = np.random.random(5)
# Se repite el primer valor para cerrar el gráfico
angles=np.concatenate((angles, [angles[0]]))
values=np.concatenate((values, [values[0]]))
labels=['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']
# Representación del mapa de calor
plt.polar(angles, values, 'o-', linewidth=2)
plt.fill(angles, values, alpha=0.25)
plt.thetagrids(angles * 180 / np.pi, labels)

# ==============================



# =============================

# Otro ejemplo 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm

# Figura
fig = plt.figure()

# Tomo el eje actual y defino una proyección 3D
ax = gca(projection='3d')

# Dibujo 3D
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)

# el metodo meshgrid devuelve una matriz de coordenadas
# a partir de vectores de coordendas, que usamos para
# los datos del eje Z
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Grafico surface en 3D
surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Límites del eje Z
ax.set_zlim(-1.01, 1.01)

# Barra de nivel, un poco más pequeña
fig.colorbar(surf, shrink=0.5, aspect=10)
xx, yy = np.mgrid[0:seleccion.shape[0], 0:seleccion.shape[1]]

ax.plot_surface(xx, yy, seleccion ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

# ================================
# Representación de datos bidimensionales con imagenes
import matplotlib.pyplot as plt
# Leo la imagen png a array de numpy

img = plt.imread("img.jpg")

# imagen en color, tres bandas
imshow(img)

# Tomo la banda R
R = img[:, :, 0]
imshow(R, cmap=gray())


# Hacemos un zoom y escojemos la zona de interes
# xlim e ylim necesitan enteros
xlim0, xlim1 = array(xlim(), dtype=int)
ylim0, ylim1 = array(ylim(), dtype=int)

print(ylim0, ylim1)
# (328, 192)

# Se realiza una selección basada en los límites actuales de la gráfica
# Atencion!: Hay que fijarse que Y es la primera dimension (primer eje) y por
# eso va de primero y ademas el origen está arriba a la izquierda, por lo
# hay que poner primero el segundo limite ylim1
seleccion = R[ylim1:ylim0, xlim0:xlim1]

# Quito los ejes
axis('off')

# Muestro la grafica con barra de color
imshow(seleccion, cmap=jet())
cb = colorbar()

# Limpio la figura
clf()

# Muestro la imagen en gris y se creo contornos
# con mapa de color jet()
imshow(seleccion, cmap=gray())
contour(seleccion, levels=arange(0,1,0.2), color=jet())

# ===============================
'''Manipulación y procesamiento de imágenes usando Numpy y Scipy
https://claudiovz.github.io/scipy-lecture-notes-ES/advanced/image_processing/index.html
  Creando un array numpy desde un archivo de imagen'''
from scipy import misc
l = misc.ascent()
misc.imsave('lena2.png', l) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(l)
plt.show()
# Creando un array numpy desde un archivo de imagen
lena = misc.imread('lena2.png')
type(lena)
lena.shape, lena.dtype
l.tofile('lena.raw') # Creación de un fichero raw
lena_from_raw = np.fromfile('lena.raw', dtype=np.int64)
lena_from_raw.shape

lena_from_raw.shape = (512, 512)
import os
os.remove('lena.raw')
# Trabajando en una lista de archivos de imágenes
for i in range(10):
    im = np.random.random_integers(0, 255, 10000).reshape((100, 100))
    misc.imsave('random_%02d.png' % i, im)

from glob import glob
filelist = glob('random*.png')
filelist.sort()
l = misc.ascent()
import matplotlib.pyplot as plt
plt.imshow(l, cmap=plt.cm.gray)
plt.imshow(l, cmap=plt.cm.gray, vmin=30, vmax=200)


# ================================
# Arreglos con Indices " Fancy Indexing"
rand = np.random.RandomState(42) # Fija la semilla aleatoria
x = rand.randint(100,size=10)
print(x)
print(x[3],x[7], x[2])
# Es equivalente pero pasando los indices en un array
ind = [3,7,4]
print(x[ind])  # Es lo que hace que se parezca a R, son cacracterísticas similares

ind= np.array([[3,7],
               [4,5]])
print(x[ind])
# Ejemplos de seleción de puntos
# Seleccionando puntos al azar
mean=[0,0]
cov = [[1,2],[2,5]]
X = rand.multivariate_normal(mean, cov, 100)
print(X)

# DIMENSIÓN DE LA MATRIZ - 100 filas 2 columnas
print(X.shape)  # Enfoque POO porque se está llamando un método shape

# Otros ejemlos de GRAFICACIÓN
import matplotlib.pyplot as plt
import seaborn  # Paquete de graficación muy usado en python

seaborn.set() # Para el estilo
plt.scatter(X[:,0],X[:,1]) # Ploteamos todos los puntos
open_close_plot()

# Ahora seleccionamos 20 filas al azar para plotear NOTA: Se dee evaluar todo el codigo al mismo tiempo
indices=np.random.choice(X.shape[0],20, replace=False)
seleccion = X[indices] # fancy Index aquí
plt.scatter(X[:,0],X[:,1], alpha=0.3)
# Los puntos azules pequeños son los de X y los más grandes son los de la seleccion
open_close_plot()

# Visualización de Datos
import matplotlib.pyplot as plt
import numpy as np
# Estilo Clásico
plt.style.use('classic')

# Ejemplos de gráficos de líneas  (graficando funciones)
# Datos del eje X para los siguietnes gráficos , bueno para ilustrar se generan los siguientes datos
x = np.linspace(0,10,100) # Se generan los datos del 0 al 10.
print(x)
# Nota: lo siguiente se ejecuta todo junto
plt.plot(x,np.sin(x)) # Se plotea x con sin(x)
plt.plot(x,np.cos(x)) # Se plotea x con cos(x)
open_close_plot()

# Otro ejemplo
fig=plt.figure()
plt.plot(x,np.sin(x),'-')
plt.plot(x,np.cos(x),'--');
open_close_plot()

# Cambiar la carpeta de trabajo para guardar el gráfico

import os
print(os.getcwd())
os.chdir("C:/Users/MILITUS/.spyder-py3/")
print(os.getcwd())
fig.savefig("prueba.png")
plt.figure() # Crea la figura con el objeto fig
# Creamos el primer panel
plt.subplot(2,1,1) # filas, columnas, nro. de paneles
plt.plot(x,np.sin(x))
# Creamos el segundo panel
plt.subplot(2,1,2)
plt.plot(x,np.cos(x))


# Ahora un estilo Orientado a Objetos para situaciones más complejas
fig, ax = plt.subplots(2)
# Llama el método plot
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
open_close_plot()

# Otro ejemplo Orientado a Objetos 
fig=plt.figure()
ax=plt.axes() # ax es un objeto tipo plot
x=np.linspace(0,10,100)
ax.plot(x,np.sin(x)) # Se invoca el método a través del punto, x es una instancia de la clase matplotlib
open_close_plot()

# Estilo funcional
x=np.linspace(0,10,100)
plt.plot(x,np.sin(x))

# Usando colores
plt.plot(x,np.sin(x - 0), color='blue')  # Nombre del color
plt.plot(x,np.sin(x - 1), color='g')  # Código del color  (rg bcmyk)
plt.plot(x,np.sin(x - 2), color='0.75')  # Escala de grise entre 0 y 1
plt.plot(x,np.sin(x - 3), color='#FFDD44')  # Código hexadecimal RRGGBB from  00 to FF
plt.plot(x,np.sin(x - 4), color=(1.0,0.2,0.3))  # Tupla RGB entre 0 y 1
plt.plot(x,np.sin(x - 5), color='chartreuse')  # NOmbres en color HTML


# Tipos de punto en  líneas que pueda haber en python
plt.plot(x,x + 0 , linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 3, linestyle='dashdot')
plt.plot(x, x+ 4, linestyle='dotted')
open_close_plot()

# Estilos de líneas
plt.plot(x, x + 4, linestyle='-')
plt.plot(x, x + 5, linestyle='--')
plt.plot(x, x + 6, linestyle='-.')
plt.plot(x, x + 7, linestyle=':')
open_close_plot()

# Se pueden cambiar los límites
plt.plot(x,np.sin(x))
plt.xlim(-1,11)
plt.ylim(-1.5,1.5)


# Se puede colocar Títulos
plt.plot(x,np.sin(x))
plt.title("Función seno(x)")
plt.xlabel("x")
plt.ylabel("Seno(x)")
open_close_plot()

# Colocar leyendas
plt.plot(x, np.sin(x), '-g', label='Seno(x)')
plt.plot(x, np.cos(x), ':b', label='Coseno(x)')
plt.axis('equal')
plt.legend()  # Con legend() plotea los label
open_close_plot()

# Otro estilo Orientado a Objetos
ax = plt.axes()  # Todo se hace a través del Objeto ax
ax.plot(x,np.sin(x)) # Se invoca el método plot()
ax.set(xlim=(0,10), ylim=(-2,2), xlabel='x', ylabel='Seno(x)',title='Un ploteo de Seno(x)') # el set, modifica todos los atributos de la class
open_close_plot()

# GRÁFICOS Scatter plot = Ejes XY

# Ejemplo 1
x = np.linspace(0,10,30)
y = np.sin(x)
plt.plot(x,y,'o',color='black')
open_close_plot()

# Ejemplo 2
rng = np.random.RandomState(0)
for marca in ['o','.',',','x','+','v','<','>','s','d']:
    plt.plot(rng.rand(5),rng.rand(5),marca,label="marca='{0}'".format(marca))
plt.legend(numpoints=1)
plt.xlim(0,1.8)
open_close_plot()

# Ejemplo 3
plt.plot(x,y,'-ok')


# Ejemplo 4
plt.plot(x,y,'-p',color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2,1.2)


# Comando scatter, más potente 
# Ejemplo 1
plt.scatter(x,y,marker='o')
open_close_plot()

# Ejemplo 2
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colores=rng.rand(100)
tamanos = 1000 * rng.rand(100)
plt.scatter(x,y,c=colores, s=tamanos,alpha=0.3,cmap='viridis')
plt.colorbar()
open_close_plot()

# Ejemplo 3
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)

# Ploteando Histogramas  y Densidad
plt.style.use('seaborn-white')
datos= np.random.randn(1000)
plt.hist(datos)
open_close_plot()


# El paquete seaborn para crear un dataset de imagenes
import seaborn as sns
iris=sns.load_dataset('iris')
print(iris.head)
corr=sns.pairplot(iris,hue='species',size=2.5)  # Cruza las variables todas contra todas, y el color lo va a dar según las species


# Ahora con seaborn
seaborn.set() # Establece el estilo de ploteo
plt.hist(datos)
plt.title('Distribución de los datos')
plt.xlabel('Alto')
plt.ylabel('Número')
open.close_plot()

# =========================================================================
"""OTRO ANÁLISIS EXPLORATORIO
# Utilizando una archivo .csv de datos separados por comas, se requierea realizar un análisis
 para saber que estrategias serán las mejores y obtener de este modo buenas conclusiones.

# Paso 1: Cargar la tabla de datos """
import pandas as pd
#import prince
import os
import numpy as np


#datos=pd.read_csv("D:/Big-data/Base Datos Prueba/DENUE_INEGI_01_2016.csv",delimiter=';',decimal=',')
datos=pd.read_csv("D:/Big-data/Base Datos Prueba/titanic_DataSET.csv",delimiter=';',decimal=',')
print(datos)
print(datos.head) # Previsualización de la data para tener una mejor idea de los datos que se tienen para trabajar
""" 
Podemos ver que cuando se previsualizan los datos, la columna Cabin tiene valores en NaN, 
éste valor se traduce en python como un None y en humano como un valor nulo. 
Sería súper útil saber que registros por columna tienen datos valores nulos y en un sólo 
método puedes obtener esa cuenta.
""" 
print(datos.count()) # Se obtiene la cuenta de los datos no nulos
print(datos.shape) # Visualiza Nro. de registro o filas y variables es la dimensión de la matriz
col_names=datos.columns.tolist() # Obtenemos los nombres de la columnas como una lista
# Iteramos sobre la lista
for column in col_names:
    print("Valores nulos en <{0}>: {1}".format(column,datos[column].isnull().sum()))

# No todo es pandas, podemos hacer un poquito de programaicón
""" Usualmente cuando se trabaja con dataset o colecciones de datos, se requiere hacer algún
 tipo de estandarización a los valores o darles algún formato en particular. 
 Si tomamos, como ejemplo la columna Sex, que tiene como únicos valores male y female y lo  
 queremos reemplazar por M y F respectivamente. Una vez, invocando habilidades de programación 
 podemos hacerlo con diccionarios y funciones — las cuales no siempre son tan fáciles 
 de entender, pero espero que en el ejemplo no sea tan complicado 
# Creamos un diccionario con los valores originales y los que se van a reemplazar
# Utilizamos un función de reeplazo en una sola línea n * n
# Verficamos el cambio 
"""
dicc={'male':'M','female':'F'}
print(dicc)

datos['sex'].head()

# Una forma más sencilla de acceder a las columnas
datos.age
""" 
También obtener información de los principales indicadores estadísticos sobre nuestro dataset
 en una sola línea
"""
datos.describe()
"""
Podemos ver que para Fare, el mínimo valor es 0…¡¿Eso quiere decir que hubo personas que 
viajaron gratis?!
"""
datos[datos.fare==0]
# Podemos hacer una agrupación de tablas por referencia cruzada 
# Para determinar en números quién de las mujeres y hombres sobrevieron más
pd.crosstab(datos.survived,datos.sex) 

""" 
Otro ejemplo  de Agrupaciones por varias columnas. 
¿ Cuántos hombres y mujeres sobrevivieron por clase?
"""
pclass_survived_count_df= datos.groupby(['pclass','sex']).sum()
pclass_survived_count_df

pclass_survived_count_df= datos.groupby(['pclass','sex'])['survived'].sum()
pclass_survived_count_df


"""PASO 2 : PRESENTACIÓN DE ESTADISTICAS BÁSICAS"""
print(datos.dropna().describe()) # Funnciona para las variables numéricas
# El describe(), permite hallar el count, mean, sts, min,25%, 50%, 75, max ect CON UN ENFOQUE FUNCIONAL
# Se pueden encontra variables cualitativas o categoricas
# CON UN ENFOQUE OO SERÍA
print(datos.describe())
print(datos.mean(numeric_only=True))  # Para que no de error con las variables que no son numéricas
print(datos.median(numeric_only=True))
print(datos.std(numeric_only=True))
print(datos.max(numeric_only=True))  # No basta con calcularlos bastaría con interpretar un poco esto
print(datos.quantile(np.array([0,0.25,0.50,0.75,1]))) # Se obtiene los percentiles o porcentajes de 0 a 1
"""Contando datos en las variables Categóricas  """
print(pd.crosstab(index=datos['survived'],columns='count')) # Si o NO. Cruza la variable survived y crea una tabla cruzada contando cuantos sobrevivieron y cuantos no
print(pd.crosstab(index=datos['sex'],columns='count'))
print(pd.crosstab(index=datos['embarked'],columns='count'))
print(datos['embarked'].value_counts()) # También se puede hacer con value_counts() como vimos antes
sex_embarked=pd.crosstab(index=datos['sex'],columns=datos['embarked']) # se puede cruzar una tabla entre sex y embarked entonces se descubre algo interesante
print(sex_embarked)

""" PASO 3 GRÁFICOS IMPORTANTES
Visualizamos cuántas personas vivieron al desatre?
La libreria más utilizada para hacer análisis gráfico o ploteo
Determinamos primero el área del canvas (área del dibujo) para las gráficas
Utilizamos subplot2grid para poder tener los gráficos uno al lado del otro 
"""
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30,10))  # Creamos un canvas o figura  de 30 x 10 
# Veremos un plot al costado del otro, para esto, realizamos una grilla (celdas)
plt.subplot2grid((2,3),(0,0))
datos.survived.value_counts().plot(kind='bar',alpha=0.5)
plt.title("Sobrevivieron - Cuenta total -")

plt.subplot2grid((2,3),(0,1))
datos.survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.title('Sobrevivieron - procentaje total -')
plt.show()
"""
¿Fueron los hombres los que sobrevivieron más?, o fueron las mujeres?
 Muy probablemente lo que vimos en la película ¿fue cierto?
 La figura de la izquierda señala los sobrevivientes en número
 en cambio la de la derecha, los visualiza en porcentajes.
 Menos del 40% de nuestro dataset sobrevivieron.
"""
#Podemos filtrar por sobrevivencia
datos.sex[datos.survived==1]
fig = plt.figure(figsize=(30,10))
datos.sex[datos.survived==1].value_counts(normalize=True).plot(kind='barh', alpha=0.5, color='br')
plt.title(' Sobrevivieron - Masculino Vs Femenino')
plt.show()

"""
¿El tipo de tiquete (clase de cabina) influyó en la supervivencia de los pasajeros?
Los colores son: bgrcmykw
"""
print(datos.pclass)
fig = plt.figure(figsize=(10,5))
datos.pclass[datos.survived==1].value_counts(normalize=True).plot(kind='bar', alpha=0.5, color='rby')
plt.title(' Sobrevivientes por clase de Ticket')
plt.show()
"""
Resulta interesante que quienes tenían un tiquete de clase intermedia tenian poco 
chance de sobrevivir, ¡sería un interesante análisis!
"""
"""
Algo que se podría deducir en relación a la edad y economía es que probablemente las personas
más jóvenes tenían menos dinero y por ende compraron los tiquetes más baratos.
Ésto lo podemos ver con una gráfica de densidad, otro tipo (kind) de gráfica disponible en matplotlib
¿Habrá alguna relaciçon entre tipo de ticket y edad? (Poder adquisitivo)
"""
#Funciones de densidad
densidad = datos[datos.columns[:2]].plot(kind='density')
open_clos_plot()

densidad= datos['age'].plot(kind='density')
open_close_plot()

densidad= datos[datos.columns[:1]].plot(kind='hist')
open_close_plot()

densidad= datos['age'].plot(kind='hist')
open_close_plot()

densidad= datos[datos.columns[:8]].plot(kind='hist')
open_close_plot()


fig=plt.figure(figsize=(20,10))
for t_class in [1,2,3]:
    datos.age[datos.pclass == t_class].plot(kind='kde')

plt.legend(["1ra. Clase", "2da. Clase", "3ra. Clase."])
plt.show()

"""
La línea de la primera clase, nos muestra que el promedio de edad del comprador es de 40 años
La línea de la tercera clase, tiene un promedio mucho más joven
Podriamos hacer una inferencia temprana y decir que los hombres que se salvaron fueron
en su mayoría ricos y mayores de 30 años.
Si revisamos la línea verde de la (3ra. clase) vemos que el promedio de edad es cerca de los 20 años
y en 1ra. clase el promedio de edad es 40, lo que muestra una relación entre edad-economía
Miremos algo muy interesante de la gráfica, la cual denota que las líneas de edad no empiezan en 0….
Podría conisderarse que ¡Habían infantes!
De manera afortunada todos los bebés de nuestro dataset o titanic sobrevivieron, un posterior análisis
sería averiguar si sus padres o al menos sus madres sobrevivieron.
"""
datos[datos.age < 1]

boxplots=datos.boxplot(return_type='axes')
open_clos_plot()
