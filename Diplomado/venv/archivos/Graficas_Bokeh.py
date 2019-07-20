# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:04:51 2019

@author: Daniel Contreras
"""

## para guardar en una locacion hay que poner la ruta adelante con slash
#from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import output_notebook,show,output_file
from bokeh.resources import INLINE
output_notebook(resources = INLINE)


def Funcion_Grafica2016(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2016.html', title="histogram.py example")
    show(p)
    
def Funcion_Grafica2017(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2017.html', title="histogram.py example")
    show(p)
    
def Funcion_Grafica2018(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2018.html', title="histogram.py example")
    show(p)

def Funcion_Grafica2019(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2019.html', title="histogram.py example")
    show(p)
    
    
#####Graficas de los 15 con mas problemas

def Funcion_Grafica2016_15Departamentos(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x 15 Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 30000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2016_15Departamaentos.html', title="histogram.py example")
    show(p)

def Funcion_Grafica2017_15Departamentos(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x 15 Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2017_15Departamaentos.html', title="histogram.py example")
    show(p)

def Funcion_Grafica2018_15Departamentos(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x 15 Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 40000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2018_15Departamaentos.html', title="histogram.py example")
    show(p)
    
def Funcion_Grafica2019_15Departamentos(Lista,Numero_Departamento):
    fruits = Numero_Departamento
    counts = Lista
    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))
    p = figure(x_range=fruits,plot_width=900,plot_height=400,title="Casos de violencia x 15 Departamento")
    p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', fill_color=factor_cmap('fruits', palette=viridis(32), factors=fruits))
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 30000
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    output_file('2019_15Departamaentos.html', title="histogram.py example")
    show(p)
    
###graficas hombre y mujer
'''from math import pi

import pandas as pd

from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum

output_file("pie.html")

x = {
    'United States': 157,
    'United Kingdom': 93,
    'Japan': 89,
    'China': 63,
    'Germany': 44,
    'India': 42,
    'Italy': 40,
    'Australia': 35,
    'Brazil': 32,
    'France': 31,
    'Taiwan': 31,
    'Spain': 29
}

data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = Category20c[len(x)]

p = figure(plot_height=350, title="Pie Chart", toolbar_location=None,   
           tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='country', source=data)

p.axis.axis_label=None
p.axis.visible=False
p.grid.grid_line_color = None

show(p)'''