import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

class Exemplos:
    def __init__(self, path_txt) -> None:
        self.gerarGrafo(path_txt)


    def gerarGrafo(self, path_txt):

        arquivo = open(path_txt, "r", encoding="utf-8")
        self.grafo = nx.Graph()

        for linha in arquivo:
            adj, coords = linha.split("/", maxsplit=1)
            bairro, *adjs = adj.split(", ")

            adjs = [(bairro, adj, {"pintado": False}) for adj in adjs]
            self.grafo.add_node(bairro,
                                coord=make_tuple(coords), 
            )
            self.grafo.add_edges_from(adjs)

    def buscaProfundidade(self, inicio):
        return
        visitados = set(inicio)

        for vizinho in self.grafo_original.edges(inicio):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                anterior[vizinho] = v
                self.busca_em_profundidade(vizinho, anterior, visitados, niveis, nivel+1)


    def buscaLargura(self, inicio):
        fila = []
        visitados = {inicio}
        anterior = {}
        fila.append(inicio)

        while len(fila):
            v = fila.pop(0)

            for vizinho in self.grafo.edges(v):
                vizinho = vizinho[1]
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
                    anterior[vizinho] = v

        return anterior
    
    def gerarArvore(self, tipo):
        arvore = dict()
        if tipo == "profundidade":
            arvore = self.buscaProfundidade()
        elif tipo == "largura":
            arvore = self.buscaLargura()

    def pintarAresta(self, aresta):
        pass

    def printarGrafo(self):

        pos = dict()

        #pos = nx.spring_layout(nx.Graph(self.grafo))

        for vertice in self.grafo.nodes(data=True):
            pos[vertice[0]] = vertice[1]["coord"]
      
        g = self.grafo

        nx.draw(g, pos, with_labels=True, font_weight='bold', font_size=11, node_size=300) #fonte 6 nodesize 200
        print(pos)
        #plt.savefig(fr"Exemplo grafo.png", format="png", dpi=300)

        plt.show()  


a = Exemplos("./Grafos/SP.txt")

print(a.grafo.nodes(data=True))
print(a.grafo.edges(data=True))
a.printarGrafo()