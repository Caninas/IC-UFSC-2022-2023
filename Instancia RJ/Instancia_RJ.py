import os
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
from numpy import array
from math import floor, ceil
from random import randint, random
import multiprocessing as mp
import ast

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
        self.vertice_de_inicio = "Flamengo"
        self.SIRxTdeVertices = dict()

        # variaveis globais, se aplicam a todos os vértices
        # retiradas da pagina 8 e 9 do artigo do modelo

        self.frac_infect_inicial = 0.05   #2/15     # 400/3000

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
        self.grafo_arvores = 0


    def gerar_grafo(self):
        g = nx.Graph()

        for linha in self.arquivo:
            adj, propriedades = linha.strip().split("/")
            nome, *adj = adj.split(", ")
            numero, populaçao, beta = propriedades.split(", ")
            populaçao = int(populaçao)

            adj = [(nome, v) for v in adj] + [(v, nome) for v in adj]

            if nome == self.vertice_de_inicio:
                I = floor(self.frac_infect_inicial * populaçao)
            else:
                I = 0
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


    def resetar_grafo(self):
        for vertice in self.grafo.nodes:
            populaçao = self.grafo.nodes[vertice]["populaçao"]
            
            if vertice == self.vertice_de_inicio:
                I = floor(self.frac_infect_inicial * populaçao)
            else:
                I = 0
            S = populaçao - I

            Sponto = floor(self.alpha * S)      # pessoas que respeitam o distanciamento social (ficam no vertice)
            Iponto = floor(self.alpha * I)

            S2pontos = S - Sponto               # pessoas que nao respeitam (podem sair do vertice)
            I2pontos = I - Iponto

            self.grafo.nodes[vertice]["SIRd"] = {"S": Sponto, "I": Iponto, "R": 0}
            
            self.grafo.nodes[vertice]["SIRdd"] = {"S": S2pontos, "I": I2pontos, "R": 0}
            self.grafo.nodes[vertice]["SIRddd"] = {}
            self.grafo.nodes[vertice]["SIRdddantes"] = {}
            
            self.SIRs = []
            self.tempos = []
            self.t = 1
            self.pico_infectados = 0
            self.juntos = True

    def printar_grafo(self, tipo=None):
        # #pos = nx.circular_layout(self.grafo.subgraph(("Ipanema"...)))
        # pos = {'Ipanema': array([1.0000000e+00, 1.4702742e-08]), 'Glória': array([0.809017  , 0.58778526]), 'Catete': array([0.30901698, 0.95105655]),
        # 'Laranjeiras': array([-0.30901702,  0.95105649]), 'Cosme Velho': array([-0.80901699,  0.58778526]), 'Urca': array([-9.99999988e-01, -7.27200340e-08]),
        # 'Leme': array([-0.80901693, -0.58778529]), 'São Conrado': array([-0.30901711, -0.95105646]), 'Vidigal': array([ 0.30901713, -0.95105646]),
        # 'Leblon': array([ 0.80901694, -0.58778529]),
        # 'Gávea': (0, -0.2), 'Flamengo': (0, 0.4), 'Botafogo': (-0.5, 0.2), 'Humaitá': (0.25, 0.1), 'Copacabana': (-0.2, -0.5),
        # 'Lagoa': (0.4, -0.25), 'Jardim Botânico': (0.6, 0.6), 'Rocinha': (0.7, 0.1)}

        #T = nx.balanced_tree(2, 5)

        mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}
        
        self.grafo = nx.relabel_nodes(self.grafo, mapping)

        for vertice in self.grafo.nodes(data=True):
            del vertice[1]["id"]
            del vertice[1]["populaçao"]
            del vertice[1]["SIR_t0"]
            del vertice[1]["SIRd"]
            del vertice[1]["SIRdd"]
            del vertice[1]["SIRddd"]
            del vertice[1]["SIRdddantes"]
            del vertice[1]["beta"]
        
        pos = graphviz_layout(self.grafo, prog="dot")

        nx.draw(self.grafo, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)
        plt.show()  


        # plt.figure(figsize=(10,8))

        # #nx.draw(self.grafo, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)
        # nx.draw(self.grafo, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

        # plt.show()

    def printar_estado_vertice(self, vertice):
        print(f"{vertice}: {self.grafo.nodes[vertice]}")
    
    def printar_estados_vertices(self):
        for vertice in self.grafo.nodes:
            print(f"{vertice}: SIR_t0: {self.grafo.nodes[vertice]['SIR_t0']}\n"
                f"        SIRd:  {self.grafo.nodes[vertice]['SIRd']}\n"
                f"        SIRdd:  {self.grafo.nodes[vertice]['SIRdd']}\n"
                f"        SIRddd:  {self.grafo.nodes[vertice]['SIRddd']}")

    
    def printar_grafico_SIRxT(self, x=None, y=None, path=None):
        fig = plt.figure(1)#.set_fig
        ax = fig.add_subplot(111)
        fig.set_size_inches([10, 7])


        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        if path:
            plt.xlim(left=1, right=len(x))
            plt.plot(x, y)
        else:
            plt.xlim(left=self.tempos[0], right=self.t-1)
            plt.plot(self.tempos, self.SIRs)

        ax.legend(["S", "I", "R"], loc='center right', bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Pessoas')


        if not path:
            plt.show()
            plt.close()
            return
        else:
            inicio = path.split("\\")[-1]
            plt.title(f'Raiz: {inicio.split(".png")[0]}')
            plt.savefig(path, format="png", bbox_inches='tight')
            plt.close()


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

        if self.t == 1 or self.juntos == True:                                             # distribuiçao inicial de pessoas
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
            print(self.t)
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
                self.tempo_pico = self.t
                self.pico_infectados = soma_SIR[1]


            self.t += 1


    def busca_em_largura(self, inicio):
        fila = []
        visitados = set()
        anterior = {}

        fila.append(inicio)
        while len(fila):
            v = fila.pop(0)
            for vizinho in self.grafo_original.edges(v):
                vizinho = vizinho[1]
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
                    anterior[vizinho] = v

        return anterior

    def gerar_grafos_arvore_largura(self, tempo, iteraçoes):
        self.resultados_arvore_largura = open(resultados_arvore_largura, "w", encoding="utf-8")
        self.SIRxTdeVerticesTXT_largura = open(SIRxTdeVerticesTXT_largura, "w", encoding="utf-8")

        g = self.grafo.nodes
        self.grafo_original = self.grafo.copy()
        
        menor_media = 99999999

        for inicio in g:
            self.vertice_de_inicio = inicio
            self.resetar_grafo()
            soma = 0
            tempo_pico = 0

            anterior = self.busca_em_largura(inicio)

            adj = {}

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
                try:
                    adj[ant].append(vertice)
                except:
                    adj[ant] = [vertice]

                self.grafo.add_edge(ant, vertice)


            for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)


            for i in range(iteraçoes):
                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao_dinamica(tempo)
                print("Pico:", self.pico_infectados)

                soma += self.pico_infectados
                tempo_pico += self.tempo_pico

                self.resetar_grafo()
            
            tempo_pico = tempo_pico / (i + 1)
            media = soma / (i + 1)
            self.resultados_arvore_largura.write(f"{self.grafo.nodes[inicio]['id']}, {tempo_pico}, {media}\n")   

            self.SIRxTdeVerticesTXT_largura.write(f"{inicio}, ")

            self.SIRxTdeVerticesTXT_largura.write(f"{self.SIRxTdeVertices}\n")
                
            menor_media = media if media < menor_media else menor_media
        
        self.resultados_arvore_largura.write(f"\nMenor média: {menor_media}")

            
    def printar_grafico_arvore(self, tipo_arvore):
        # self.picos_infectados_arvores = {"largura": [], "profundidade": []}  (eixo y)
        # self.indice 1-159 (eixo x)
        if tipo_arvore in {"largura", "profundidade"} or self.picos_infectados_arvores: 
            eixo_x = [indice for indice in range(1, len(self.picos_infectados_arvores[tipo_arvore]) + 1)]
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
            print("Tipo de árvore inválida") if self.picos_infectados_arvores \
            else print("É necessário rodar o modelo sobre as árvores primeiro")

    def busca_em_profundidade(self, v):

        for vizinho in self.grafo_original.edges(v):
            vizinho = vizinho[1]
            if vizinho not in self.visitados:
                self.visitados.add(vizinho)
                self.anterior_profundidade[vizinho] = v
                self.busca_em_profundidade(vizinho)

        return

    def gerar_grafos_arvore_profundidade(self, tempo, iteraçoes):
        self.resultados_arvore_profundidade = open(resultados_arvore_profundidade, "w", encoding="utf-8")
        self.SIRxTdeVerticesTXT_profundidade = open(SIRxTdeVerticesTXT_profundidade, "w", encoding="utf-8")
        
        g = self.grafo.nodes
        self.grafo_original = self.grafo.copy()
        
        menor_media = 99999999
        for inicio in g:
            self.vertice_de_inicio = inicio
            self.resetar_grafo()
            soma = 0
            tempo_pico = 0

            self.anterior_profundidade = {}
            self.visitados = set()
            self.busca_em_profundidade(inicio)
            
            adj = {}

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            for vertice, ant in self.anterior_profundidade.items():   # recriar grafo a partir de anterior
                try:
                    adj[ant].append(vertice)
                except:
                    adj[ant] = [vertice]

                self.grafo.add_edge(ant, vertice)


            for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)
            

            for i in range(iteraçoes):
                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao_dinamica(tempo)
                print("Pico:", self.pico_infectados)

                soma += self.pico_infectados
                tempo_pico += self.tempo_pico

                self.resetar_grafo()
            
            tempo_pico = tempo_pico / (i + 1)
            media = soma / (i + 1)
            self.resultados_arvore_profundidade.write(f"{self.grafo.nodes[inicio]['id']}, {tempo_pico}, {media}\n")      
            
            self.SIRxTdeVerticesTXT_profundidade.write(f"{inicio}, ")

            self.SIRxTdeVerticesTXT_profundidade.write(f"{self.SIRxTdeVertices}\n")

            menor_media = media if media < menor_media else menor_media
        
        self.resultados_arvore_profundidade.write(f"\nMenor média: {menor_media}")


    def printar_grafico_ID_MAXINFECT_arvore(self, tipo_arvore):
        if tipo_arvore == "largura":
            resultados = open("./Resultados/resultados_arvore_largura.txt", "r")
            titulo = 'Pico de Infectados Árvore de Busca em Largura'
        else:
            resultados = open("./Resultados/resultados_arvore_profundidade.txt", "r")
            titulo = 'Pico de Infectados Árvore de Busca em Profundidade'

        resultados_lista = [x for x in range(160)]


        for linha in resultados:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)] = int(float(max_infect))

        resultados_lista[0] = 816398 # INICIO FLAMENGO    #1651756 # resultado original mudar
        #print(resultados_lista)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        plt.xlim(left=0, right=161)
        plt.xticks([x for x in range(0, 160, 4)])
        
        plt.plot(0, resultados_lista[0], "o", color="red")      # valor grafo normal
        resultados_lista.pop(0)
        plt.plot([x for x in range(1, 160)], resultados_lista, "o", color="C0")    # valores arvores

        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 - Grafo Real)')
        ax.set_ylabel('Pico de Infectados')

        fig.set_size_inches([13.3, 7.5])

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        #plt.savefig(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Pico Infectados Arvores {tipo_arvore.title()}.png", format="png")
        plt.show()

    def avançar_tempo_movimentacao_dinamica(self, t):    # s = nome vertice de origem (no caso de utilizar um grafo arvore)
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))
        for tempo in range(t):
            print(self.t)
            # distribuiçao de pessoas
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                try:
                    self.SIRxTdeVertices[nome][self.t] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]
                except:
                    self.SIRxTdeVertices[nome] = dict()
                    self.SIRxTdeVertices[nome][self.t] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]


                S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))


                for vizinho in self.grafo.edges(nome):
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}


            #print("tempo=", self.t)
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
                self.tempo_pico = self.t
                self.pico_infectados = soma_SIR[1]


            for node in list(self.grafo.nodes(data=True)):          # voltar pessoas para o proprio vertice
                nome, atributos = node

                for vizinho in self.grafo.edges(nome):
                    node_vizinho = self.grafo.nodes[vizinho[1]]["SIRddd"][nome]
                    atributos["SIRdd"]["S"] += node_vizinho["S"]
                    atributos["SIRdd"]["I"] +=  node_vizinho["I"]
                    atributos["SIRdd"]["R"] += node_vizinho["R"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {}


            self.t += 1

    def printar_grafico_SIRxTdeVerticesPizza(self):
        plt.style.use('_mpl-gallery-nogrid')

        colors = ["blue", "#55eb3b", "red"]
        for t in range(1, self.t):
            print("imagem", t)
            coluna = 0
            linha = 0
            index = 1
            fig = plt.figure(num=1, clear=True)
            fig.set_size_inches([40, 40])
            
            for key, value in self.SIRxTdeVertices.items():
                x = value[t]
                ax = plt.subplot(14, 12, index)

                #[linha][coluna]
                ax.set_title(key)
                ax.pie(x, colors=colors, radius=6, center=(4, 4),
                    wedgeprops={"linewidth": 0, "edgecolor": "white"}, frame=True)

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                index += 1
                if coluna + 1 < 12:
                    coluna += 1
                else:
                    coluna = 0
                    linha += 1
    
            plt.savefig(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Grafico pizza grafo original\tempo {t}.png", format="png")
            plt.close(fig="all")

    def printar_grafico_SIRxTdeVerticesPizzaTXT(self, path, tipo):
        self.SIRxTdeVerticesTXT = open(path, "r", encoding="utf-8")
        for linha in self.SIRxTdeVerticesTXT:
            inicio, dicionario_dados = linha.split(", ", maxsplit=(1))
            self.SIRxTdeVertices = ast.literal_eval(dicionario_dados)
            
            #plt.style.use('_mpl-gallery-nogrid')
            #colors = ["blue", "#55eb3b", "red"]

            #os.mkdir(fr"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ\Resultados\figs\{tipo}\inicio {inicio}")
            y_grafico = []
            
            print(inicio, "imagem")
            for t in range(1, max(self.SIRxTdeVertices[inicio]) + 1):
                y_grafico.append([0,0,0])
                #print(inicio, "imagem", t)
                #coluna = 0
                #linha = 0

                #fig, ax = plt.subplots(14, 12)
                
                for key, value in self.SIRxTdeVertices.items():
                    x = value[t]
                    y_grafico[t-1][0] += x[0]
                    y_grafico[t-1][1] += x[1]
                    y_grafico[t-1][2] += x[2]

                    # fig.set_size_inches([40, 40])
                    # ax[linha][coluna].set_title(key)
                    # ax[linha][coluna].pie(x, colors=colors, radius=6, center=(4, 4),
                    #     wedgeprops={"linewidth": 0, "edgecolor": "white"}, frame=True)

                    # ax[linha][coluna].get_xaxis().set_visible(False)
                    # ax[linha][coluna].get_yaxis().set_visible(False)

                    # if coluna + 1 < 12:
                    #     coluna += 1
                    # else:
                    #     coluna = 0
                    #     linha += 1

                # plt.savefig(fr"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ\Resultados\figs\{tipo}\inicio {inicio}\tempo {t}.png", format="png")
                # plt.close()
            
            x_grafico = [x for x in range(1, max(self.SIRxTdeVertices[inicio]) + 1)]
            self.printar_grafico_SIRxT(x_grafico, y_grafico, fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Graficos SIRxT arvores {tipo}\{inicio}.png")

#? Escrever resultados etc
#? Salvar arquivos relevantes drive e separado

#os.chdir(r"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ")
os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

# "./txts/normal (real)/adjacencias.txt"
# "./txts/normal (real)/arquivo_final.txt"
# "./txts/zona sul/adjacencias_zona_sul.txt"
# "./txts/otimizado/adjacencias.txt"
arquivo_adjacencias = "./txts/normal (real)/adjacencias.txt"
arquivo_final = "./txts/normal (real)/arquivo_final.txt"#"./txts/zona sul/arquivo_final.txt"
arquivo_ID_nomes = "./txts/nova relaçao ID - bairros.txt"
tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"


resultados_arvore_profundidade = "./Resultados/resultados_arvore_profundidade.txt"
SIRxTdeVerticesTXT_profundidade = "./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt"
resultados_arvore_largura = "./Resultados/resultados_arvore_largura.txt"
SIRxTdeVerticesTXT_largura = "./Resultados/SIR_vertice_por_tempo_LARGURA.txt"


# txt = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_final, tabela_populaçao)
# txt.gerar_arquivo_destino()


m = Modelo(arquivo_final)

#m.avançar_tempo_movimentacao_dinamica(200)

#m.printar_grafico_SIRxT()

#m.printar_grafico_SIRxTdeVerticesPizza()

# m.gerar_grafos_arvore_largura(200, 1) # FEITO
# m.printar_grafico_SIRxTdeVerticesPizzaTXT(SIRxTdeVerticesTXT_largura, "largura") # FEITO
# m.gerar_grafos_arvore_profundidade(200, 1) # FEITO
# m.printar_grafico_SIRxTdeVerticesPizzaTXT(SIRxTdeVerticesTXT_profundidade, "profundidade") # FEITO

#m.printar_grafico_ID_MAXINFECT_arvore("largura")

#print(m.pico_infectados)
