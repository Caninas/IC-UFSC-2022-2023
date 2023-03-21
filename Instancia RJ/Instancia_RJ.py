import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
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
        self.t = 1

        # variaveis para a construçao do grafico posteriormente
        self.tempos = []
        self.SIRs = []
        self.pico_infectados = 0

        # variaveis globais, se aplicam a todos os vértices
        # retiradas da pagina 8 e 9 do artigo do modelo

        self.frac_infect_inicial = 2/15     # 400/3000

        self.v = 91/200    # taxa_virulencia
        self.e = 29/200    # taxa_recuperaçao

        self.alpha = 2/5        # fraçao de pessoas que respeitam o distanciamento, fator de distanciamento
                                # aplicado por vertice ou globalmente

        self.lambda_S = 2/5       # fraçao de pessoas que respeitam distanciamento mas precisam sair (podem ser infectados)
        self.lambda_I = 1/10      
        self.lambda_R = 3/5

        # diz se o SIR que não respeita distanciamento esta ou nao em seu vertice original
        self.juntos = True
        self.isolados = False

        self.array_top_populaçao = []

        self.grafo = self.gerar_grafo()


    def gerar_grafo(self):
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
                beta=float(beta))
            g.add_edges_from(adj)

        return g

    def printar_grafo(self):
        # #pos = nx.circular_layout(self.grafo.subgraph(("Ipanema"...)))
        # pos = {'Ipanema': array([1.0000000e+00, 1.4702742e-08]), 'Glória': array([0.809017  , 0.58778526]), 'Catete': array([0.30901698, 0.95105655]),
        # 'Laranjeiras': array([-0.30901702,  0.95105649]), 'Cosme Velho': array([-0.80901699,  0.58778526]), 'Urca': array([-9.99999988e-01, -7.27200340e-08]),
        # 'Leme': array([-0.80901693, -0.58778529]), 'São Conrado': array([-0.30901711, -0.95105646]), 'Vidigal': array([ 0.30901713, -0.95105646]),
        # 'Leblon': array([ 0.80901694, -0.58778529]),
        # 'Gávea': (0, -0.2), 'Flamengo': (0, 0.4), 'Botafogo': (-0.5, 0.2), 'Humaitá': (0.25, 0.1), 'Copacabana': (-0.2, -0.5),
        # 'Lagoa': (0.4, -0.25), 'Jardim Botânico': (0.6, 0.6), 'Rocinha': (0.7, 0.1)}

        plt.figure(figsize=(10,8))
        #nx.draw(self.grafo, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)
        nx.draw(self.grafo, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

        plt.show()

    def printar_estado_vertice(self, vertice):
        print(f"{vertice}: {self.grafo.nodes[vertice]}")
    
    def printar_estados_vertices(self):
        for vertice in self.grafo.nodes:
            print(f"{vertice}: SIR_t0: {self.grafo.nodes[vertice]['SIR_t0']}\n"
                f"        SIRd:  {self.grafo.nodes[vertice]['SIRd']}\n"
                f"        SIRdd:  {self.grafo.nodes[vertice]['SIRdd']}\n"
                f"        SIRddd:  {self.grafo.nodes[vertice]['SIRddd']}")

    
    def printar_grafico_SIRxT(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        plt.xlim(left=self.tempos[0], right=self.t-1)
        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.plot(self.tempos, self.SIRs)
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)


        ax.legend(["S", "I", "R"], loc='center right', bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Pessoas')    

        plt.show()

        # self.juntos = True
        # for node in list(self.grafo.nodes(data=True)):          # voltar pessoas para o proprio vertice
        #     nome, atributos = node

        #     for vizinho in self.grafo.edges(nome):
        #         node_vizinho = self.grafo.nodes[vizinho[1]]["SIRddd"][nome]
        #         atributos["SIRdd"]["S"] += node_vizinho["S"]
        #         atributos["SIRdd"]["I"] +=  node_vizinho["I"]
        #         atributos["SIRdd"]["R"] += node_vizinho["R"]
        #         self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {}

        # print console todos os vertices
        # for vertice in self.grafo.nodes():
        #     self.printar_estado_vertice(vertice)


    def mergesort(self, array):
        if len(array) > 1:
            m = len(array)//2
            esquerda = array[:m]
            direita = array[m:]

            self.mergesort(esquerda)
            self.mergesort(direita)
            
            i = 0
            j = 0
            k = 0

            while i < len(esquerda) and j < len(direita):
                if esquerda[i] > direita[j]:
                    array[k]=esquerda[i]
                    i=i+1
                else:
                    array[k]=direita[j]
                    j=j+1
                k=k+1

            while i < len(esquerda):
                array[k]=esquerda[i]
                i=i+1
                k=k+1

            while j < len(direita):
                array[k]=direita[j]
                j=j+1
                k=k+1


    def isolar_vertices_mais_populosos(self):
        if not self.array_top_populaçao:
            pop = {}

            self.quantidade_bairros_selecionados = round(0.4 * len(self.grafo.nodes))

            for vertice in self.grafo.nodes(data=True):
                pop[vertice[1]["populaçao"]] = vertice[0]

                self.array_top_populaçao.append(vertice[1]["populaçao"])
            
            self.mergesort(self.array_top_populaçao)
            
            for i in range(len(self.array_top_populaçao)):
                self.array_top_populaçao[i] = pop[self.array_top_populaçao[i]]

        if not self.isolados and not self.juntos:
            print("Isolando vértices!")
            for bairro in self.array_top_populaçao:
                for vizinho, populaçao_de_fora in self.grafo.nodes[bairro]["SIRddd"].items():   # mandar a pop de outros de dentro para fora
                    self.grafo.nodes[vizinho]["SIRdd"]["S"] += populaçao_de_fora["S"]
                    populaçao_de_fora["S"] = 0

                    self.grafo.nodes[vizinho]["SIRdd"]["I"] += populaçao_de_fora["I"]
                    populaçao_de_fora["I"] = 0

                    self.grafo.nodes[vizinho]["SIRdd"]["R"] += populaçao_de_fora["R"]
                    populaçao_de_fora["R"] = 0


                for vizinho in self.grafo.edges(bairro):            # receber pop de fora de volta
                    self.grafo.nodes[bairro]["SIRdd"]["S"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["S"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["S"] = 0

                    self.grafo.nodes[bairro]["SIRdd"]["I"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["I"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["I"] = 0

                    self.grafo.nodes[bairro]["SIRdd"]["R"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["R"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["R"] = 0

            self.isolados = True

        else:
            if self.juntos:
                print("População ainda não se movimentou!")
                return
            
            print("Tirando isolamento!")

            for bairro in self.array_top_populaçao:
                nome = bairro
                atributos = self.grafo.nodes[nome]

                S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))

                for vizinho in self.grafo.edges(nome):
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}

            self.isolados = False

        #self.printar_estados_vertices()

    def avançar_tempo(self, t):
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))
        if self.t == 0 or self.juntos == True:                                             # distribuiçao inicial de pessoas
            self.juntos = False
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node
                S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))

                for vizinho in self.grafo.edges(nome):
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}

        for tempo in range(t):
            print("tempo=", self.t)

            self.tempos.append(self.t)

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
            soma_SIR = [0, 0, 0]

            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                # guardando valores SIR para usar no grafico
                soma_SIR[0] += atributos["SIRd"]["S"] + atributos["SIRdd"]["S"] + atributos["SIRdddantes"]["S"]
                soma_SIR[1] += atributos["SIRd"]["I"] + atributos["SIRdd"]["I"] + atributos["SIRdddantes"]["I"]
                soma_SIR[2] += atributos["SIRd"]["R"] + atributos["SIRdd"]["R"] + atributos["SIRdddantes"]["R"]
                
                #! Calculo X_ponto + Sdd que fica (Ypi = Ypi -> i)

                Nponto = floor(self.lambda_S * atributos["SIRd"]["S"]) + floor(self.lambda_I * atributos["SIRd"]["I"]) + floor(self.lambda_R * atributos["SIRd"]["R"])
                N2pontos = atributos["SIRdd"]["S"] + atributos["SIRdd"]["I"] + atributos["SIRdd"]["R"]
                N3pontos = atributos["SIRdddantes"]["S"] + atributos["SIRdddantes"]["I"] + atributos["SIRdddantes"]["R"]

                # prob de acontecer um encontro de Sd / Sdd com Infectados no geral (Ypi = Ypi -> i)
                Y_ponto = (floor(self.lambda_I * atributos["SIRd"]["I"]) + atributos["SIRdd"]["I"] + atributos["SIRdddantes"]["I"]) / ((Nponto + N2pontos + N3pontos) - 1)
                
                X_ponto = 0          # numero de encontros de S com I no vertice i
                X_2pontos_ii = 0

                # range S bonito 1 ponto
                for i in range(floor(self.lambda_S * atributos["SIRd"]["S"])):    # calculo do numero de encontros na pop de suscetiveis que respeitam mas tem q sair
                    X_ponto += random() < Y_ponto                                 # baseado na probabilidade de Y_ponto

                # X_2pontos = X_2pontos_ii + X_2pontos_ij (somatorio Y_2pontos_ii(=Y_ponto) + somatorio dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                for i in range(atributos["SIRdd"]["S"]):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X_2pontos_ii += random() < Y_ponto                      # baseado na probabilidade de Y_ponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (2 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                for vizinho in atributos["SIRddd"].values():                # SIR t+1 vizinhos
                    X_2pontos_ij = 0

                    for i in range(vizinho["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                        X_2pontos_ij += random() < Y_ponto                                            # baseado na probabilidade de Y_2pontos
                        
                    recuperados_novos_ddd = ceil(self.e * vizinho["I"])

                    vizinho["S"] = vizinho["S"] - floor(self.v * X_2pontos_ij)
                    vizinho["I"] = vizinho["I"] - recuperados_novos_ddd + floor(self.v * X_2pontos_ij)
                    vizinho["R"] = vizinho["R"] + recuperados_novos_ddd
        
                # SIR t+1 SIRponto e SIR2pontos
                recuperados_novos_d = ceil(self.e * atributos["SIRd"]["I"])
                recuperados_novos_dd = ceil(self.e * atributos["SIRdd"]["I"])

                atributos["SIRd"]["S"] = atributos["SIRd"]["S"] - floor(self.v * X_ponto)              # X = quantidade total de encontros, v*X pessoas suscetiveis de fato infectadas
                atributos["SIRdd"]["S"] = atributos["SIRdd"]["S"] - floor(self.v * X_2pontos_ii)

                atributos["SIRd"]["I"] = atributos["SIRd"]["I"] - recuperados_novos_d + floor(self.v * X_ponto)
                atributos["SIRdd"]["I"] = atributos["SIRdd"]["I"] - recuperados_novos_dd + floor(self.v * X_2pontos_ii)

                atributos["SIRd"]["R"] = atributos["SIRd"]["R"] + recuperados_novos_d
                atributos["SIRdd"]["R"] = atributos["SIRdd"]["R"] + recuperados_novos_dd

            self.SIRs[self.t-1] = [soma_SIR[0], soma_SIR[1], soma_SIR[2]]

            if not any(i[1] > soma_SIR[1] for i in self.SIRs):
                self.pico_infectados = soma_SIR[1]

            #print(self.grafo.nodes["Flamengo"])
            self.t += 1

    def printar_grafico_arvore(tipo_arvore):
        # self.picos_infectados_arvores = {"largura": [], "profundidade": []}  (eixo y)
        # self.indice 1-159 (eixo x)
        if tipo_arvore in {"largura", "profundidade"} or self.picos_infectados_arvores: 
            eixo_x = [for indice in range(1, len(self.picos_infectados_arvores[tipo_arvore]) + 1)]
            eixo_y = self.picos_infectados_arvores[tipo_arvore]

            fig = plt.figure(1)
            ax = fig.add_subplot(111)

            plt.xlim(left=eixo_x[0], right=eixo_x[-1])
            #plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
            plt.plot(eixo_x, eixo_y)
            plt.gca().get_yaxis().get_major_formatter().set_scientific(False)


            #ax.legend(["Arvores"], loc='center right', bbox_to_anchor=(1.1, 0.5))
            ax.set_xlabel('Arvore')
            ax.set_ylabel('Pico de Infectados')    

            plt.show()
        else:
            print("Tipo de árvore inválida") if self.picos_infectados_arvores /
            else print("É necessário rodar o modelo sobre as árvores primeiro")

#os.chdir(r"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ")
os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

# "./txts/normal (real)/adjacencias.txt"
# "./txts/normal (real)/arquivo_final.txt"
# "./txts/otimizado/adjacencias.txt"
arquivo_adjacencias = "./txts/zona sul/adjacencias_zona_sul.txt"
arquivo_final = "./txts/normal (real)/arquivo_final.txt"#"./txts/zona sul/arquivo_final.txt"
arquivo_ID_nomes = "./txts/relaçao ID - bairros.txt"
tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"

#txt = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_final, tabela_populaçao)
#txt.gerar_arquivo_destino()


m = Modelo(arquivo_final)
m.gerar_grafo()
m.avançar_tempo(4)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(2)
m.isolar_vertices_mais_populosos()
m.avançar_tempo(200)
# print(len(m.grafo.edges))
print(m.pico_infectados)
m.printar_grafo()


m.printar_grafico_SIRxT()


# arquivo_pico_infectados = "./txts/pico_infectados.txt"
# pico_infec = open(arquivo_pico_infectados, "w")

# pico_infec.write(f"Normal\n")
# for i in range(50):
#     m = Modelo(arquivo_final)
#     m.gerar_grafo()
#     m.avançar_tempo(75)
#     pico_infec.write(f"{str(m.pico_infectados)}\n")


# pico_infec.write(f"\nOtimizado\n")
# arquivo_final = "./txts/zona sul/arquivo_final_otimizado_circulo.txt"

# for i in range(50):
#     m = Modelo(arquivo_final)
#     m.gerar_grafo()
#     m.avançar_tempo(75)
#     pico_infec.write(f"{str(m.pico_infectados)}\n")
