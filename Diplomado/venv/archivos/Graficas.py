
output_file("test.html")
p=figure(title="no me joda")
p.line(numero_Departamentos,lista,line_width=2)
show(p)


hist, bin_edges = np.histogram(lista, density=True)

x = np.linspace(-2, 2, 1000)

p = figure()
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")

output_file("hist.html")
show(p)