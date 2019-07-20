# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:18:50 2019

@author: front
"""
from consultas import db,numero_Departamentos,numero_15,departamentos
from Graficas_Bokeh import Funcion_Grafica2017,Funcion_Grafica2017_15Departamentos
collection2017=db.Data_2017

Amazonas=collection2017.find({"DEPARTAMENTO":"AMAZONAS"}).count()
Antioquia=collection2017.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Arauca=collection2017.find({"DEPARTAMENTO":'ARAUCA'}).count()
Atlántico=collection2017.find({"DEPARTAMENTO":'ATLÁNTICO'}).count()
Bolívar=collection2017.find({"DEPARTAMENTO":'BOLÍVAR'}).count()
Boyacá=collection2017.find({"DEPARTAMENTO":'BOYACÁ'}).count()
Caldas=collection2017.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Caquetá=collection2017.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Casanare=collection2017.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Cauca=collection2017.find({"DEPARTAMENTO":'CAUCA'}).count()
Cesar=collection2017.find({"DEPARTAMENTO":'CESAR'}).count()
Chocó=collection2017.find({"DEPARTAMENTO":'CHOCÓ'}).count()
Córdoba=collection2017.find({"DEPARTAMENTO":'CÓRDOBA'}).count()
Cundinamarca=collection2017.find({"DEPARTAMENTO":'CUNDINAMARCA'}).count()
Guainía=collection2017.find({"DEPARTAMENTO":'GUAINÍA'}).count()
Guaviare=collection2017.find({"DEPARTAMENTO":'GUAVIARE'}).count()
Huila=collection2017.find({"DEPARTAMENTO":'HUILA'}).count()
La_Guajira=collection2017.find({"DEPARTAMENTO":'GUAJIRA'}).count()
Magdalena=collection2017.find({"DEPARTAMENTO":'MAGDALENA'}).count()
Meta=collection2017.find({"DEPARTAMENTO":'META'}).count()
Nariño=collection2017.find({"DEPARTAMENTO":'NARIÑO'}).count()
Norte_de_Santander=collection2017.find({"DEPARTAMENTO":'NORTE DE SANTANDER'}).count()
Putumayo=collection2017.find({"DEPARTAMENTO":'PUTUMAYO'}).count()
Quindío=collection2017.find({"DEPARTAMENTO":'QUINDÍO'}).count()
Risaralda=collection2017.find({"DEPARTAMENTO":'RISARALDA'}).count()
San_Andrés=collection2017.find({"DEPARTAMENTO":'SAN ANDRÉS'}).count()
Santander=collection2017.find({"DEPARTAMENTO":'SANTANDER'}).count()
Sucre=collection2017.find({"DEPARTAMENTO":'SUCRE'}).count()
Tolima=collection2017.find({"DEPARTAMENTO":'TOLIMA'}).count()
Valle_del_Cauca=collection2017.find({"DEPARTAMENTO":'VALLE'}).count()
Vaupés=collection2017.find({"DEPARTAMENTO":'VAUPÉS'}).count()
Vichada=collection2017.find({"DEPARTAMENTO":'VICHADA'}).count()
lista_2017=(
    Amazonas,Antioquia,Arauca,Atlántico,Bolívar,Boyacá,Caldas,Caquetá,Casanare,
    Cauca,Cesar,Chocó,Cundinamarca,Córdoba,Guainía,Guaviare,La_Guajira,Huila,
    Magdalena,Meta,Nariño,Norte_de_Santander,Putumayo,Quindío,Risaralda,
    San_Andrés,Santander,Sucre,Tolima,Valle_del_Cauca,Vaupés,Vichada
    )
Funcion_Grafica2017(lista_2017,numero_Departamentos)
listamayoramenor2017=sorted(lista_2017,reverse=True)
matriz=[]
for i in range(32):
    matriz.append([departamentos[i]])
    for j in range(1):
        matriz[i].append(lista_2017[i])
a=sorted(matriz,key=lambda x: x[1],reverse=True)
print("Orden 2017",a)
Lista2017=[]
i2017=0
while i2017<15:
    Lista2017.append(listamayoramenor2017[i2017])
    i2017+=1
Funcion_Grafica2017_15Departamentos(Lista2017,numero_15)