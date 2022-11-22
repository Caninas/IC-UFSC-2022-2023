import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# exportar para graphviz
# G = nx.complete_graph(5)
# PG = nx.nx_pydot.to_pydot(G)
# H = nx.nx_pydot.from_pydot(PG)

class Bairro:
    def __init__(self, cod, nome):
        self.cod = cod
        self.nome = nome

        self.S = 3000
        self.I = 300
        self.R = 0
        self.X = 0
        
        #self.taxa_virulencia = 29/200
        #self.taxa_recuperaçao = 91/200

        #self.taxa_distanciamento = 0

    def __repr__(self):
        return self.nome



G = nx.Graph()

flamengo = Bairro(15, "Flamengo")
gloria = Bairro(16, "Gloria")
laranjeiras = Bairro(17, "Laranjeiras")
catete = Bairro(18, "Catete")
cosme_velho = Bairro(19, "Cosme Velho")
botafogo = Bairro(20, "Botafogo")
humaita = Bairro(21, "HumaitÁ")
urca = Bairro(22, "Urca")
leme = Bairro(23, "Leme")
copacabana = Bairro(24, "Copacabana")
ipanema = Bairro(25, "Ipanema")
leblon = Bairro(26, "Leblon")
lagoa = Bairro(27, "Lagoa")
jardim_botanico = Bairro(28, "Jardim Botânico")
gavea = Bairro(29, "Gávea")
vidigal = Bairro(30, "Vidigal")
sao_conrado = Bairro(31, "Sao Conrado")
rocinha = Bairro(154, "Rocinha")

G.add_node(flamengo)
G.add_node(gloria)
G.add_node(laranjeiras)
G.add_node(catete)
G.add_node(cosme_velho)
G.add_node(botafogo)
G.add_node(humaita)
G.add_node(urca)
G.add_node(leme)
G.add_node(copacabana)
G.add_node(ipanema)
G.add_node(leblon)
G.add_node(lagoa)
G.add_node(leblon)
G.add_node(jardim_botanico)
G.add_node(gavea)
G.add_node(vidigal)
G.add_node(sao_conrado)
G.add_node(rocinha)


G.add_edge(catete, gloria)
G.add_edge(catete, flamengo)
G.add_edge(catete, laranjeiras)

G.add_edge(gloria, flamengo)

G.add_edge(laranjeiras, flamengo)
G.add_edge(laranjeiras, cosme_velho)
G.add_edge(laranjeiras, botafogo)

G.add_edge(botafogo, flamengo)
G.add_edge(botafogo, humaita)
G.add_edge(botafogo, urca)
G.add_edge(botafogo, copacabana)

G.add_edge(urca, botafogo)

G.add_edge(humaita, lagoa)
G.add_edge(humaita, jardim_botanico)

G.add_edge(copacabana, leme)
G.add_edge(copacabana, lagoa)
G.add_edge(copacabana, ipanema)

G.add_edge(lagoa, jardim_botanico)
G.add_edge(lagoa, ipanema)
G.add_edge(lagoa, leblon)
G.add_edge(lagoa, gavea)

G.add_edge(jardim_botanico, gavea)

G.add_edge(leblon, ipanema)
G.add_edge(leblon, gavea)
G.add_edge(leblon, vidigal)

G.add_edge(vidigal, rocinha)
G.add_edge(vidigal, gavea)

G.add_edge(rocinha, sao_conrado)
G.add_edge(rocinha, gavea)

# pos = {flamengo: np.array((0, 0)), gloria: np.array((0, 0)), laranjeiras: np.array((0, 0)),
#         catete: np.array((0, 0)), cosme_velho: np.array((0, 0)), botafogo: np.array((0, 0)),
#         humaita: np.array((0, 0)), urca: np.array((0, 0)), leme: np.array((0, 0)),
#         copacabana: np.array((0, 0)), ipanema: np.array((0, 0)), leblon: np.array((0, 0)),
#         lagoa: np.array((0, 0)), jardim_botanico: np.array((0, 0)), gavea: np.array((0, 0)),
#         vidigal: np.array((0, 0)), sao_conrado: np.array((0, 0)), rocinha: np.array((0, 0))}

#pos = nx.rescale_layout_dict(pos)
nx.draw(G, with_labels=True, font_weight='bold')# pos,

nx.draw 
plt.show()