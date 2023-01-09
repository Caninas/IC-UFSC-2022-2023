import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
from random import randint, random

from Txt import Txt

#? exportar para graphviz
#? pintar cada zona

#? G = nx.complete_graph(5)
#? PG = nx.nx_pydot.to_pydot(G)
#? H = nx.nx_pydot.from_pydot(PG)

class Modelo:
    # arquivo no formato | int(id), adjs(sep=", ")/prop1(name), prop2(populaçao), prop3(beta)... |
    def __init__(self, arq_final):
        self.arquivo = open(arq_final, "r", encoding="utf-8")

        # tempo atual
        self.t = 0

        # variaveis para a construçao do grafico posteriormente
        self.tempos = []
        self.SIRs = []


        # variaveis globais, se aplicam a todos os vértices
        # retiradas da pagina 8 e 9 do artigo do modelo

        self.frac_infect_inicial = 2/15     # 400/3000

        self.v = 91/200    # taxa_virulencia
        self.e = 29/200    # taxa_recuperaçao

        self.alpha = 2/5        # fraçao de pessoas que respeitam o distanciamento, fator de distanciamento
                                # aplicado por vertice ou globalmente

        self.lambdaS = 2/5       # fraçao de pessoas que respeitam distanciamento mas precisam sair (podem ser infectados)
        self.lambdaI = 1/10      
        self.lambdaR = 3/5

        # diz se o SIR que não respeita distanciamento esta ou nao em seu vertice original
        self.juntos = True
        self.grafo = self.GerarGrafo()


    def GerarGrafo(self):
        g = nx.Graph()

        for linha in self.arquivo:
            adj, propriedades = linha.strip().split("/")
            nome, *adj = adj.split(", ")
            numero, populaçao, beta = propriedades.split(", ")
            populaçao = int(populaçao)
            
            adj = [(nome, v) for v in adj] + [(v, nome) for v in adj]

            I = floor(populaçao * self.frac_infect_inicial)
            S = populaçao - I

            Sponto = floor(self.alpha * S)      # pessoas que respeitam o distanciamento social (ficam no vertice)
            Iponto = floor(self.alpha * I)

            S2pontos = S - Sponto               # pessoas que nao respeitam (podem sair do vertice)
            I2pontos = I - Iponto

            g.add_node(nome, 
                id=int(numero), 
                populaçao=int(populaçao),
                SIR_t0={"S": S, "I": I, "R": 0},
                SIRd={"S": Sponto, "I": Iponto, "R": 0},
                SIRdd={"S": S2pontos, "I": I2pontos, "R": 0},
                SIRddd={},       # SIR ESTRANGEIRO (SIRdd de outros) #
                SIRdddantes={},
                quant_vizinhos=len(adj),
                beta=float(beta))
                
            g.add_edges_from(adj)

        return g

    def PrintarGrafo(self):
        plt.figure(figsize=(10,8))

        nx.draw(self.grafo, with_labels=True, font_weight='bold', font_size=6, node_size=500, clip_on=True)

        plt.show()

    def PrintarEstadoVertice(self, vertice):
        print(f"{vertice}: {self.grafo.nodes[vertice]}")
    
    def PrintarEstadoGrafo(self):
        plt.gca().set_prop_cycle('color', ['red', 'green', 'blue'])
        plt.plot(self.tempos, self.SIRs)
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        self.juntos = True
        for node in list(self.grafo.nodes(data=True)):          # voltar pessoas para o proprio vertice
            nome, atributos = node

            for vizinho in self.grafo.edges(nome):
                nodevizinho = self.grafo.nodes[vizinho[1]]["SIRddd"][nome]
                atributos["SIRdd"]["S"] += nodevizinho["S"]
                atributos["SIRdd"]["I"] +=  nodevizinho["I"]
                atributos["SIRdd"]["R"] += nodevizinho["R"]
                self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {}

        for vertice in self.grafo.nodes():
            self.PrintarEstadoVertice(vertice)

        plt.show()


    def AvançarTempo(self, t):
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambdaS*))
        i = 0
        if self.t == 0 or self.juntos == True:                                             # distribuiçao inicial de pessoas
            self.juntos = False
            for node in list(self.grafo.nodes(data=True)):          # posso atribuir direto no node
                nome, atributos = node

                S2pontossaindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                I2pontossaindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                R2pontossaindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                atributos["SIRdd"]["S"] -= S2pontossaindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["I"] -= I2pontossaindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["R"] -= R2pontossaindo * len(self.grafo.edges(nome))

                for vizinho in self.grafo.edges(nome):
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S2pontossaindo, "I": I2pontossaindo, "R": R2pontossaindo}

        for tempo in range(t):
            self.tempos.append(tempo)
            for node in list(self.grafo.nodes(data=True)):
                # x = tempo
                # y = [S, I, R]
                nome, atributos = node

                atributos["SIRdddantes"]["S"] = atributos["SIRdddantes"]["I"] = atributos["SIRdddantes"]["R"] = 0

                for vizinho in atributos["SIRddd"].values():
                    atributos["SIRdddantes"]["S"] += vizinho["S"]
                    atributos["SIRdddantes"]["I"] += vizinho["I"]
                    atributos["SIRdddantes"]["R"] += vizinho["R"]


            self.SIRs.append([])
            somaSIR = [0, 0, 0]
            for node in list(self.grafo.nodes(data=True)):          # posso atribuir direto no node
                nome, atributos = node

                # guardando valores SIR para usar no grafico
                somaSIR[0] += atributos["SIRd"]["S"] + atributos["SIRdd"]["S"] + atributos["SIRdddantes"]["S"]
                somaSIR[1] += atributos["SIRd"]["I"] + atributos["SIRdd"]["I"] + atributos["SIRdddantes"]["I"]
                somaSIR[2] += atributos["SIRd"]["R"] + atributos["SIRdd"]["R"] + atributos["SIRdddantes"]["R"]
                
                #! Calculo Xponto + Sdd que fica (Ypi = Ypi -> i)

                Nponto = floor(self.lambdaS * atributos["SIRd"]["S"]) + floor(self.lambdaI * atributos["SIRd"]["I"]) + floor(self.lambdaR * atributos["SIRd"]["R"])
                N2pontos = atributos["SIRdd"]["S"] + atributos["SIRdd"]["I"] + atributos["SIRdd"]["R"]
                N3pontos = atributos["SIRdddantes"]["S"] + atributos["SIRdddantes"]["I"] + atributos["SIRdddantes"]["R"]

                # prob de acontecer um encontro de Sd / Sdd com Infectados no geral (Ypi = Ypi -> i)
                Yponto = (floor(self.lambdaI * atributos["SIRd"]["I"]) + atributos["SIRdd"]["I"] + atributos["SIRdddantes"]["I"]) / ((Nponto + N2pontos + N3pontos) - 1)
                
                Xponto = 0          # numero de encontros de S com I no vertice i
                X2pontosii = 0

                # range S bonito 1 ponto
                for i in range(floor(self.lambdaS * atributos["SIRd"]["S"])):    # calculo do numero de encontros na pop de suscetiveis que respeitam mas tem q sair
                    Xponto += random() < Yponto                                 # baseado na probabilidade de Yponto

                # X2pontos = X2pontosii + X2pontosij (somatorio Y2pontosii(=Yponto) + somario dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                for i in range(atributos["SIRdd"]["S"]):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X2pontosii += random() < Yponto                         # baseado na probabilidade de Yponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (3 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                for vizinho in atributos["SIRddd"].values():                # SIR t+1 vizinhos
                    X2pontosij = 0

                    for i in range(vizinho["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                        X2pontosij += random() < Yponto                                            # baseado na probabilidade de Y2pontos
                        
                    recuperados_novosddd = ceil(self.e * vizinho["I"])

                    vizinho["S"] = vizinho["S"] - floor(self.v * X2pontosij)
                    vizinho["I"] = vizinho["I"] - recuperados_novosddd + floor(self.v * X2pontosij)
                    vizinho["R"] = vizinho["R"] + recuperados_novosddd
        
                # SIR t+1 SIRponto e SIR2pontos
                recuperados_novosd = ceil(self.e * atributos["SIRd"]["I"])
                recuperados_novosdd = ceil(self.e * atributos["SIRdd"]["I"])
                
                if (nome == "Bangu"):
                    i += 1
                    print("tempo=",self.t)
                    print(recuperados_novosd, recuperados_novosdd, floor(self.v * Xponto))

                atributos["SIRd"]["S"] = atributos["SIRd"]["S"] - floor(self.v * Xponto)              # X = quantidade total de encontros, v*X pessoas suscetiveis de fato infectadas
                atributos["SIRdd"]["S"] = atributos["SIRdd"]["S"] - floor(self.v * X2pontosii)

                atributos["SIRd"]["I"] = atributos["SIRd"]["I"] - recuperados_novosd + floor(self.v * Xponto)
                atributos["SIRdd"]["I"] = atributos["SIRdd"]["I"] - recuperados_novosdd + floor(self.v * X2pontosii)

                atributos["SIRd"]["R"] = atributos["SIRd"]["R"] + recuperados_novosd
                atributos["SIRdd"]["R"] = atributos["SIRdd"]["R"] + recuperados_novosdd

            self.SIRs[tempo] = [somaSIR[0], somaSIR[1], somaSIR[2]]
            self.t += 1


os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

adjacencias = "./txts/adjacencias.txt"
nomes = "./txts/bairros apenas.txt"
populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"
arquivo_final = "./txts/arquivo_final.txt"

# txt = Txt(adjacencias, nomes, arquivo_final, populaçao)
# txt.gerar_arquivo_destino()

plt.xlim(left=0, right=199)

m = Modelo(arquivo_final)
m.GerarGrafo()
m.PrintarEstadoVertice("Bangu")
m.AvançarTempo(200)
m.PrintarEstadoVertice("Bangu")
m.PrintarEstadoGrafo()
#m.PrintarGrafo()
