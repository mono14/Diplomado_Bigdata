# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:26:02 2019

@author: front
"""

from consultas import db,numero_Departamentos,numero_15,departamentos
from Graficas_Bokeh import Funcion_Grafica2018,Funcion_Grafica2018_15Departamentos
collection2018=db.Data_2018

Amazonas=collection2018.find({"DEPARTAMENTO":"AMAZONAS"}).count()
Antioquia=collection2018.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Arauca=collection2018.find({"DEPARTAMENTO":'ARAUCA'}).count()
Atlántico=collection2018.find({"DEPARTAMENTO":'ATLÁNTICO'}).count()
Bolívar=collection2018.find({"DEPARTAMENTO":'BOLÍVAR'}).count()
Boyacá=collection2018.find({"DEPARTAMENTO":'BOYACÁ'}).count()
Caldas=collection2018.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Caquetá=collection2018.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Casanare=collection2018.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Cauca=collection2018.find({"DEPARTAMENTO":'CAUCA'}).count()
Cesar=collection2018.find({"DEPARTAMENTO":'CESAR'}).count()
Chocó=collection2018.find({"DEPARTAMENTO":'CHOCÓ'}).count()
Córdoba=collection2018.find({"DEPARTAMENTO":'CÓRDOBA'}).count()
Cundinamarca=collection2018.find({"DEPARTAMENTO":'CUNDINAMARCA'}).count()
Guainía=collection2018.find({"DEPARTAMENTO":'GUAINÍA'}).count()
Guaviare=collection2018.find({"DEPARTAMENTO":'GUAVIARE'}).count()
Huila=collection2018.find({"DEPARTAMENTO":'HUILA'}).count()
La_Guajira=collection2018.find({"DEPARTAMENTO":'GUAJIRA'}).count()
Magdalena=collection2018.find({"DEPARTAMENTO":'MAGDALENA'}).count()
Meta=collection2018.find({"DEPARTAMENTO":'META'}).count()
Nariño=collection2018.find({"DEPARTAMENTO":'NARIÑO'}).count()
Norte_de_Santander=collection2018.find({"DEPARTAMENTO":'NORTE DE SANTANDER'}).count()
Putumayo=collection2018.find({"DEPARTAMENTO":'PUTUMAYO'}).count()
Quindío=collection2018.find({"DEPARTAMENTO":'QUINDÍO'}).count()
Risaralda=collection2018.find({"DEPARTAMENTO":'RISARALDA'}).count()
San_Andrés=collection2018.find({"DEPARTAMENTO":'SAN ANDRÉS'}).count()
Santander=collection2018.find({"DEPARTAMENTO":'SANTANDER'}).count()
Sucre=collection2018.find({"DEPARTAMENTO":'SUCRE'}).count()
Tolima=collection2018.find({"DEPARTAMENTO":'TOLIMA'}).count()
Valle_del_Cauca=collection2018.find({"DEPARTAMENTO":'VALLE'}).count()
Vaupés=collection2018.find({"DEPARTAMENTO":'VAUPÉS'}).count()
Vichada=collection2018.find({"DEPARTAMENTO":'VICHADA'}).count()
lista_2018=[
    Amazonas,Antioquia,Arauca,Atlántico,Bolívar,Boyacá,Caldas,Caquetá,Casanare,
    Cauca,Cesar,Chocó,Cundinamarca,Córdoba,Guainía,Guaviare,La_Guajira,Huila,
    Magdalena,Meta,Nariño,Norte_de_Santander,Putumayo,Quindío,Risaralda,
    San_Andrés,Santander,Sucre,Tolima,Valle_del_Cauca,Vaupés,Vichada
    ]
Funcion_Grafica2018(lista_2018,numero_Departamentos)
listamayoramenor2018=sorted(lista_2018,reverse=True)
matriz=[]
for i in range(32):
    matriz.append([departamentos[i]])
    for j in range(1):
        matriz[i].append(lista_2018[i])
a=sorted(matriz,key=lambda x: x[1],reverse=True)
print("Orden 2018",a)
Lista2018=[]
i2018=0
while i2018<15:
    Lista2018.append(listamayoramenor2018[i2018])
    i2018+=1
Funcion_Grafica2018_15Departamentos(Lista2018,numero_15)