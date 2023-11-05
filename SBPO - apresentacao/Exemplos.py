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
            adj, props = linha.split("/", maxsplit=1)
            props = props.split(", ")
            coord = ", ".join(props[0:2])
            print(coord)
            id = props[-1]
            bairro, *adjs = adj.split(", ")

            adjs = [(bairro, adj, {"color": "black"}) for adj in adjs]
            self.grafo.add_node(bairro,
                                id=int(id),
                                coord=make_tuple(coord)
            )
            self.grafo.add_edges_from(adjs)

    def buscaProfundidade(self, inicio):
        visitados = set(inicio)
        vertice_anterior = None
        pilha = [{inicio: vertice_anterior}]
        anterior = dict()

        while len(pilha):
            print(list(pilha[-1].keys()))

            vertice, vertice_anterior = list(pilha[-1].items())[0]
            pilha.pop()

            if vertice not in visitados:
                if vertice_anterior:
                    anterior[vertice] = vertice_anterior

                visitados.add(vertice)
                vertice_anterior = vertice

                for vizinho in self.grafo.edges(vertice):
                    vizinho = vizinho[1]
                
                    if vizinho not in visitados:
                        pilha.append({vizinho: vertice_anterior})    

        print(anterior)
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

    def pintarAresta(self, aresta, cor):
        #self.grafo
        nx.set_edge_attributes(self.grafo, {aresta: {"color": cor}})

    def printarGrafo(self, arvore=False, nome_salvar=None):
        pos = dict()
        #pos = nx.spring_layout(nx.Graph(self.grafo))
        #self.grafo = nx.Graph(((1,2), (2,3), (2,4)))

        if arvore:
            g = self.grafo
            mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}            
            self.grafo = nx.relabel_nodes(self.grafo, mapping)
            # for vertice in self.grafo.nodes(data=True):
            #     del vertice[1]["coord"]
            
            pos = graphviz_layout(self.grafo, prog="dot")
            # pos = {'Liberdade': (0, 0), 'Cambuci': (7.5, -35), 'Sé': (7.5, -28), 
            #        'Bela Vista': (0, -7), 'Consolação': (-7.5, -35), 'República': (0, -14), 
            #        'Santa Cecília': (-7.5, -28), 'Bom Retiro': (0, -21)}
            # pos = {'Liberdade': (0, 0), 'Cambuci': (35, 7.5), 'Sé': (28, 7.5), 
            #        'Bela Vista': (7, 0), 'Consolação': (35, -7.5), 'República': (14, 0), 
            #        'Santa Cecília': (28, -7.5), 'Bom Retiro': (21, 0)}
            # for vertice in self.grafo.nodes(data=True):
            #     pos[vertice[0]] = pos[vertice[1]["id"]]
            #     del pos[vertice[1]["id"]]
            self.grafo = g

            for vertice in self.grafo.nodes(data=True):
                val = pos[vertice[1]["id"]]
                del pos[vertice[1]["id"]]
                pos[vertice[0]] = val

        else:
            for vertice in self.grafo.nodes(data=True):
                pos[vertice[0]] = vertice[1]["coord"]

        matplotlib.rcParams['figure.figsize'] = 6.0973, 3
        colors = nx.get_edge_attributes(self.grafo, 'color').values()
        nx.draw(self.grafo, pos, edge_color=colors, width=2, with_labels=True, font_weight='bold', font_size=11, node_size=300,) #fonte 6 nodesize 200
        
        if nome_salvar:
            plt.savefig(nome_salvar, format="png", dpi=300)
        
        #plt.show()  


a = Exemplos("./SBPO - apresentacao/Grafos/SP.txt")

# print(a.grafo.nodes(data=True))
# print(a.grafo.edges(data=True))
a.printarGrafo(nome_salvar="Exemplo_grafo_original.png")
a.printarGrafo()
#a.gerarArvore("largura", "Liberdade")
# a.grafo.add_edge("Liberdade", "Sé")
# a.pintarAresta(("Liberdade", "Sé"), "#5bfc6e")
# # a.printarGrafo(True, "Exemplo_adiçao_aresta.png")

# a.pintarAresta(("Liberdade", "Bela Vista"), "#fa3c3c")
# a.pintarAresta(("Bela Vista", "República"), "#fa3c3c")
# a.pintarAresta(("República", "Bom Retiro"), "#fa3c3c")
# a.pintarAresta(("Bom Retiro", "Sé"), "#fa3c3c")

a.printarGrafo(True, "Exemplo_adição_clico2.png")

# # 
# # 
# #
# props = a.grafo.get_edge_data("Liberdade", "Bela Vista")
# a.grafo.remove_edge("Liberdade", "Bela Vista")
# a.printarGrafo(True, "Exemplo_remoçao1.png")
# print(props)
# a.grafo.add_edge("Liberdade", "Bela Vista", **props)

# props = a.grafo.get_edge_data("Bela Vista", "República")
# a.grafo.remove_edge("Bela Vista", "República")
# a.printarGrafo(True, "Exemplo_remoçao2.png")
# a.grafo.add_edge("Bela Vista", "República", **props)

# props = a.grafo.get_edge_data("República", "Bom Retiro")
# a.grafo.remove_edge("República", "Bom Retiro")
# a.printarGrafo(True, "Exemplo_remoçao3.png")
# a.grafo.add_edge("República", "Bom Retiro", **props)

# props = a.grafo.get_edge_data("Bom Retiro", "Sé")
# a.grafo.remove_edge("Bom Retiro", "Sé")
# a.printarGrafo(True, "Exemplo_remoçao4.png")
# a.grafo.add_edge("Bom Retiro", "Sé", **props)