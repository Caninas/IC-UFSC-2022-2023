import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
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

            adjs = [(bairro, adj, {"color": "black"}) for adj in adjs]
            self.grafo.add_node(bairro,
                                coord=make_tuple(coords), 
            )
            self.grafo.add_edges_from(adjs)

    def buscaProfundidade(self, inicio):
        visitados = set(inicio)
        pilha = [inicio]
        anterior = dict()
        vertice_anterior = None

        while len(pilha):
            vertice = pilha[-1]
            pilha.pop()

            visitados.add(vertice)

            if vertice_anterior:
                anterior[vertice] = vertice_anterior
            vertice_anterior = vertice

            for vizinho in self.grafo.edges(vertice):
                vizinho = vizinho[1]
            
                if vizinho not in visitados:
                    pilha.append(vizinho)    

        return anterior


    def buscaLargura(self, inicio):
        fila = [inicio]
        visitados = {inicio}
        anterior = dict()

        while len(fila):
            v = fila.pop(0)

            for vizinho in self.grafo.edges(v):
                vizinho = vizinho[1]
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
                    anterior[vizinho] = v

        return anterior
    
    def gerarArvore(self, tipo, inicio):
        arvore = dict()
        if tipo == "profundidade":
            arvore = self.buscaProfundidade(inicio)
        elif tipo == "largura":
            arvore = self.buscaLargura(inicio)

        self.grafo.remove_edges_from(e for e in self.grafo.edges if e not in nx.Graph([(a, b) for a,b in arvore.items()]).edges)

    def pintarAresta(self, aresta):
        pass

    def printarGrafo(self, arvore=False):

        pos = dict()


        pos = nx.spring_layout(nx.Graph(self.grafo))
        
        #pos = graphviz_layout(self.grafo, prog="dot")
        for vertice in self.grafo.nodes(data=True):
            pos[vertice[0]] = vertice[1]["coord"]
        

        colors = nx.get_edge_attributes(self.grafo, 'color').values()
        print(colors)
        nx.draw(self.grafo, pos, edge_color=colors, with_labels=True, font_weight='bold', font_size=11, node_size=300,) #fonte 6 nodesize 200
        print(pos)
        #plt.savefig(fr"Exemplo grafo.png", format="png", dpi=300)

        plt.show()  


a = Exemplos("./Grafos/SP.txt")

# print(a.grafo.nodes(data=True))
# print(a.grafo.edges(data=True))
a.gerarArvore("profundidade", "Liberdade")
a.printarGrafo()