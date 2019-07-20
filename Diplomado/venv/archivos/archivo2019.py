# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:43:57 2019

@author: front
"""


from consultas import db,numero_Departamentos,numero_15,departamentos
from Graficas_Bokeh import Funcion_Grafica2019,Funcion_Grafica2019_15Departamentos

collection_2019 = db.Data_2019

Amazonas=collection_2019.find({"DEPARTAMENTO":"AMAZONAS"}).count()
Antioquia=collection_2019.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Arauca=collection_2019.find({"DEPARTAMENTO":'ARAUCA'}).count()
Atlántico=collection_2019.find({"DEPARTAMENTO":'ATLÁNTICO'}).count()
Bolívar=collection_2019.find({"DEPARTAMENTO":'BOLÍVAR'}).count()
Boyacá=collection_2019.find({"DEPARTAMENTO":'BOYACÁ'}).count()
Caldas=collection_2019.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Caquetá=collection_2019.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Casanare=collection_2019.find({"DEPARTAMENTO":'ANTIOQUIA'}).count()
Cauca=collection_2019.find({"DEPARTAMENTO":'CAUCA'}).count()
Cesar=collection_2019.find({"DEPARTAMENTO":'CESAR'}).count()
Chocó=collection_2019.find({"DEPARTAMENTO":'CHOCÓ'}).count()
Córdoba=collection_2019.find({"DEPARTAMENTO":'CÓRDOBA'}).count()
Cundinamarca=collection_2019.find({"DEPARTAMENTO":'CUNDINAMARCA'}).count()
Guainía=collection_2019.find({"DEPARTAMENTO":'GUAINÍA'}).count()
Guaviare=collection_2019.find({"DEPARTAMENTO":'GUAVIARE'}).count()
Huila=collection_2019.find({"DEPARTAMENTO":'HUILA'}).count()
La_Guajira=collection_2019.find({"DEPARTAMENTO":'GUAJIRA'}).count()
Magdalena=collection_2019.find({"DEPARTAMENTO":'MAGDALENA'}).count()
Meta=collection_2019.find({"DEPARTAMENTO":'META'}).count()
Nariño=collection_2019.find({"DEPARTAMENTO":'NARIÑO'}).count()
Norte_de_Santander=collection_2019.find({"DEPARTAMENTO":'NORTE DE SANTANDER'}).count()
Putumayo=collection_2019.find({"DEPARTAMENTO":'PUTUMAYO'}).count()
Quindío=collection_2019.find({"DEPARTAMENTO":'QUINDÍO'}).count()
Risaralda=collection_2019.find({"DEPARTAMENTO":'RISARALDA'}).count()
San_Andrés=collection_2019.find({"DEPARTAMENTO":'SAN ANDRÉS'}).count()
Santander=collection_2019.find({"DEPARTAMENTO":'SANTANDER'}).count()
Sucre=collection_2019.find({"DEPARTAMENTO":'SUCRE'}).count()
Tolima=collection_2019.find({"DEPARTAMENTO":'TOLIMA'}).count()
Valle_del_Cauca=collection_2019.find({"DEPARTAMENTO":'VALLE'}).count()
Vaupés=collection_2019.find({"DEPARTAMENTO":'VAUPÉS'}).count()
Vichada=collection_2019.find({"DEPARTAMENTO":'VICHADA'}).count()

lista_2019=[
    Amazonas,Antioquia,Arauca,Atlántico,Bolívar,Boyacá,Caldas,Caquetá,Casanare,
    Cauca,Cesar,Chocó,Cundinamarca,Córdoba,Guainía,Guaviare,La_Guajira,Huila,
    Magdalena,Meta,Nariño,Norte_de_Santander,Putumayo,Quindío,Risaralda,
    San_Andrés,Santander,Sucre,Tolima,Valle_del_Cauca,Vaupés,Vichada
    ]
Funcion_Grafica2019(lista_2019,numero_Departamentos)

matriz=[]
for i in range(32):
    matriz.append([departamentos[i]])
    for j in range(1):
        matriz[i].append(lista_2019[i])
a=sorted(matriz,key=lambda x: x[1],reverse=True)
print("orden 2019",a)
#####Graficas generos por violencia
listamayoramenor2019=sorted(lista_2019,reverse=True)
Lista2019=[]
i2019=0
while i2019<15:
    Lista2019.append(listamayoramenor2019[i2019])
    i2019+=1
Funcion_Grafica2019_15Departamentos(Lista2019,numero_15)