import networkx as nx

import matplotlib.pyplot as plt

def busca(g, inicio):
    fila = []
    visitados = set()
    anterior = {}

    fila.append(inicio)

    while len(fila):
        v = fila.pop(0)
        print(v)
        for vizinho in g.edges(v):
            if vizinho not in visitados:
                vizinho = vizinho[1]
                visitados.add(vizinho)
                print("vizinho:", vizinho)
                fila.append(vizinho)
                anterior[vizinho] = v

    return anterior


g = nx.DiGraph()
arestas = [("a", "b"), ("b", "c"), ("a", "d")]

g.add_edges_from(arestas)

print(g.edges("a"))
anterior = busca(g, "a")

print(anterior)

j = nx.DiGraph()
for vertice, ant in anterior.items():   #recriar grafo a partir de anterior
    j.add_edge(ant, vertice)




plt.figure(figsize=(10,8))
nx.draw(g, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

plt.show()
plt.figure(figsize=(10,8))
nx.draw(j, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

plt.show()