# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:47:21 2019

@author: front
"""
from consultas import db,numero_Departamentos,numero_15,departamentos
from Graficas_Bokeh import Funcion_Grafica2016,Funcion_Grafica2016_15Departamentos

collection = db.Data_2016

Amazonas=collection.find({"DEPARTAMENTO":"AMAZONAS"}).count()
Antioquia=collection.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Arauca=collection.find({"DEPARTAMENTO":'ARAUCA'}).count()
Atlántico=collection.find({"DEPARTAMENTO":'ATLÁNTICO'}).count()
Bolívar=collection.find({"DEPARTAMENTO":'BOLÍVAR'}).count()
Boyacá=collection.find({"DEPARTAMENTO":'BOYACÁ'}).count()
Caldas=collection.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Caquetá=collection.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Casanare=collection.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Cauca=collection.find({"DEPARTAMENTO":'CAUCA'}).count()
Cesar=collection.find({"DEPARTAMENTO":'CESAR'}).count()
Chocó=collection.find({"DEPARTAMENTO":'CHOCÓ'}).count()
Córdoba=collection.find({"DEPARTAMENTO":'CÓRDOBA'}).count()
Cundinamarca=collection.find({"DEPARTAMENTO":'CUNDINAMARCA'}).count()
Guainía=collection.find({"DEPARTAMENTO":'GUAINÍA'}).count()
Guaviare=collection.find({"DEPARTAMENTO":'GUAVIARE'}).count()
Huila=collection.find({"DEPARTAMENTO":'HUILA'}).count()
La_Guajira=collection.find({"DEPARTAMENTO":'GUAJIRA'}).count()
Magdalena=collection.find({"DEPARTAMENTO":'MAGDALENA'}).count()
Meta=collection.find({"DEPARTAMENTO":'META'}).count()
Nariño=collection.find({"DEPARTAMENTO":'NARIÑO'}).count()
Norte_de_Santander=collection.find({"DEPARTAMENTO":'NORTE DE SANTANDER'}).count()
Putumayo=collection.find({"DEPARTAMENTO":'PUTUMAYO'}).count()
Quindío=collection.find({"DEPARTAMENTO":'QUINDÍO'}).count()
Risaralda=collection.find({"DEPARTAMENTO":'RISARALDA'}).count()
San_Andrés=collection.find({"DEPARTAMENTO":'SAN ANDRÉS'}).count()
Santander=collection.find({"DEPARTAMENTO":'SANTANDER'}).count()
Sucre=collection.find({"DEPARTAMENTO":'SUCRE'}).count()
Tolima=collection.find({"DEPARTAMENTO":'TOLIMA'}).count()
Valle_del_Cauca=collection.find({"DEPARTAMENTO":'VALLE'}).count()
Vaupés=collection.find({"DEPARTAMENTO":'VAUPÉS'}).count()
Vichada=collection.find({"DEPARTAMENTO":'VICHADA'}).count()

lista_2016=[
    Amazonas,Antioquia,Arauca,Atlántico,Bolívar,Boyacá,Caldas,Caquetá,Casanare,
    Cauca,Cesar,Chocó,Cundinamarca,Córdoba,Guainía,Guaviare,La_Guajira,Huila,
    Magdalena,Meta,Nariño,Norte_de_Santander,Putumayo,Quindío,Risaralda,
    San_Andrés,Santander,Sucre,Tolima,Valle_del_Cauca,Vaupés,Vichada
    ]
Funcion_Grafica2016(lista_2016,numero_Departamentos)

matriz=[]
for i in range(32):
    matriz.append([departamentos[i]])
    for j in range(1):
        matriz[i].append(lista_2016[i])
a=sorted(matriz,key=lambda x: x[1],reverse=True)
print("Orden 2016",a)
#####Graficas generos por violencia

listamayoramenor=sorted(lista_2016,reverse=True)
print(listamayoramenor)
Lista=[]
i=0
while i<15:
    Lista.append(listamayoramenor[i])
    i+=1
Funcion_Grafica2016_15Departamentos(Lista,numero_15)