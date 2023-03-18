import networkx as nx
import pydot
import matplotlib.pyplot as plt

def busca(g, inicio):
    fila = []
    visitados = set()
    anterior = {}

    fila.append(inicio)

    while len(fila):
        v = fila.pop(0)
        #print(g.edges(v))
        for vizinho in g.edges(v):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                visitados.add(vizinho)
                #print("vizinho:", vizinho)
                fila.append(vizinho)
                anterior[vizinho] = v

    return anterior

def caminho(raiz, destino):
    arra = []
    while destino != raiz:
        arra.insert(0, destino)
        destino = anterior[destino]
    arra.insert(0, raiz)
    print(arra)

adj_rj = open(r"Instancia RJ\Txts\normal (real)\adjacencias.txt", "r", encoding="utf-8")

g = nx.Graph()
arestas = [("a", "b"), ("b", "c"), ("a", "d")]
for linha in adj_rj:
    vertice, adj = linha.split(" ", maxsplit=1)
    adj = adj.strip().split(" ")
    #print(vertice, adj)
    if adj[0] != "":
        arestas = [(int(vertice), int(x)) for x in adj]
        g.add_edges_from(arestas)

#print(g.edges())
raiz = 1
anterior = busca(g, raiz)

caminho(raiz, 148)

j = nx.Graph()
adjacencias = open("adjacencias.txt", "w", encoding="utf-8")

adj = {}
for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
    try:
        adj[ant].append(vertice)
    except:
        adj[ant] = [vertice]
    adjacencias.write(f"{ant}, {vertice}\n")
    j.add_edge(ant, vertice)

#print(adj)




# plt.figure(figsize=(10,8))
# nx.draw(g, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

# plt.show()

# pos = nx.drawing.nx_pydot.graphviz_layout(j, prog="dot")
# plt.figure(figsize=(10,8))
# nx.draw(j, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

# plt.show()