import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
from numpy import array
from math import floor, ceil
from random import randint, random
import multiprocessing as mp
import ast
import time
import pandas as pd
import openpyxl
import seaborn as sns
#from blessed import Terminal

from Txt import Txt

#? exportar para graphviz
#? pintar cada zona

#? G = nx.complete_graph(5)
#? PG = nx.nx_pydot.to_pydot(G)
#? H = nx.nx_pydot.from_pydot(PG)



class Modelo:
    # arquivo no formato | int(id), adjs(sep=", ")/prop1(name), prop2(populaçao), prop3(beta)... |
    def __init__(self, arq_final, flo=False):
        self.arquivo = open(arq_final, "r", encoding="utf-8")
        # tempo atual
        self.floripa = flo
        self.t = 1

        # variaveis para a construçao do grafico posteriormente
        self.tempos = []
        self.SIRs = []
        self.pico_infectados = 0
        self.vertice_de_inicio = ""
        self.SIRxTdeVertices = dict()

        # variaveis globais, se aplicam a todos os vértices
        # retiradas da pagina 8 e 9 do artigo do modelo

        self.frac_infect_inicial = 0.05     # 400/3000 2/15  

        self.v = 91/200    # taxa_virulencia
        self.e = 29/200    # taxa_recuperaçao

        self.alpha = 2/5        # fraçao de pessoas que respeitam o distanciamento, fator de distanciamento

        self.lambda_S = 2/5       # fraçao de pessoas que respeitam distanciamento mas precisam sair (podem ser infectados)
        self.lambda_I = 1/10      
        self.lambda_R = 3/5

        # diz se o SIR que não respeita distanciamento esta ou nao em seu vertice original
        self.juntos = False
        self.isolados = False
        self.edges_isoladas = []

        self.array_top_populaçao = []

        self.grafo = (self.gerar_grafo_florianopolis if self.floripa else self.gerar_grafo)()    # grafo utilizado atualmente
        self.grafo_original = 0             # copia do grafo original, usado na criaçao de arvores

        #self.terminal = Terminal()

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
            #I = floor(self.frac_infect_inicial * populaçao)    # distribuir igualmente
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
                beta=float(beta),
                isolado=False)
            g.add_edges_from(adj)

        return g


    def resetar_grafo(self):
        for vertice in self.grafo:
            if self.floripa:
                populaçao_s, populaçao_i = self.grafo.nodes[vertice]["populaçao_s"], self.grafo.nodes[vertice]["populaçao_i"]
                I = populaçao_i
                S = populaçao_s
            
            
            else:
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
            #self.juntos = True

    def estimar_tempo_restante(self, percorrido, total):
        if percorrido:
            estimativa = ((time.perf_counter() - self.estimativa_tempo_inicial) / percorrido) * (total - percorrido)
        else:
            self.estimativa_tempo_inicial = time.perf_counter()
            estimativa = 0

        #with self.terminal.location(y=self.terminal.height):
            #print("Tempo restante:", self.tempo_restante, "segundos")
            #print(self.terminal.on_firebrick2("   "))
        print(f"Estimativa {estimativa/60:.2f} minutos | {total - percorrido} arvores restantes")


    def salvar_grafo_arvore(self, path):
        mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}            
        g = nx.relabel_nodes(self.grafo, mapping)

        if self.floripa:
            for vertice in g.nodes(data=True):
                del vertice[1]["id"]
                del vertice[1]["populaçao_s"]
                del vertice[1]["populaçao_i"]
                del vertice[1]["SIR_t0"]
                del vertice[1]["SIRd"]
                del vertice[1]["SIRdd"]
                del vertice[1]["SIRddd"]
                del vertice[1]["SIRdddantes"]
        else:
            for vertice in g.nodes(data=True):
                del vertice[1]["id"]
                del vertice[1]["populaçao"]
                del vertice[1]["SIR_t0"]
                del vertice[1]["SIRd"]
                del vertice[1]["SIRdd"]
                del vertice[1]["SIRddd"]
                del vertice[1]["SIRdddantes"]
                del vertice[1]["beta"]
    
        pos = graphviz_layout(g, prog="dot")

        nx.draw(g, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

        fig = plt.figure(1)
        plt.savefig(path, format="png", dpi=300)
        plt.close()

    def printar_grafo(self, tipo=None):
        # # pos = nx.circular_layout(self.grafo.subgraph(("Ipanema"...)))
        #self.grafo = nx.Graph(self.grafo)

        pos = nx.spring_layout(nx.Graph(self.grafo))
        if tipo == "zonasul":
            pos = {'Ipanema': array([1.0000000e+00, 1.4702742e-08]), 'Copacabana': array([0.809017  , 0.58778526]), 'Botafogo': array([0.30901698, 0.95105655]),
            'Humaitá': array([-0.30901702,  0.95105649]), 'Jardim Botânico': array([-0.80901699,  0.58778526]), 'Gávea': array([-9.99999988e-01, -7.27200340e-08]),
            'Rocinha': array([-0.80901693, -0.58778529]), 'São Conrado': array([-0.30901711, -0.95105646]), 'Vidigal': array([ 0.30901713, -0.95105646]),
            'Leblon': array([ 0.80901694, -0.58778529])}
            pos2 = {'Glória': array([1, 3.5]), 'Catete': array([0, 3.5]), 'Laranjeiras': array([0, 2.5]), 'Flamengo': array([1, 2.5])}
            pos3 = { 'Lagoa': array([-0.15, 0.8]), 'Cosme Velho': array([0.5, 3]), 'Leme': array([0.6, 0.5]), 'Urca': array([0.5,  2.3])}
            pos_b = array([0.5, 1.5])
            print(pos)

            for bairro, posiçao in pos.items():
                if bairro != "Botafogo":
                    pos[bairro][0] = pos_b[0] + (posiçao[0] - pos["Botafogo"][0])
                    pos[bairro][1] = pos_b[1] + (posiçao[1] - pos["Botafogo"][1])
            pos["Botafogo"] = pos_b
            #pos2 = {'Glória': array([3, 3]), 'Flamengo': array([0.70710678, 0.70710677]), 'Catete': array([-1.73863326e-08,  9.99999992e-01]), 'Laranjeiras': array([-9.99999947e-01, -6.90443471e-08])}
            pos = {**pos2, **pos, **pos3}
        
        #T = nx.balanced_tree(2, 5)
        
        #pos = nx.circular_layout(self.grafo.subgraph(["São Conrado", "Rocinha", "Gávea", "Vidigal", "Leblon", "Ipanema", "Copacabana", "Jardim Botânico", "Humaitá", "Botafogo"]))
        #pos = {'Flamengo': array([0.6043461, 0.4442784]), 'Laranjeiras': array([0.45074005, 0.55273503]), 'Glória': array([0.8534418 , 0.58982338]), 'Botafogo': array([0.31947341, 0.18126152]), 'Catete': array([0.68333495, 0.64406827]), 'Cosme Velho': array([0.42134842, 0.85813163]), 'Humaitá': array([-0.04907372,  0.02847084]), 'Copacabana': array([ 0.11292418, -0.20412333]), 'Urca': array([0.57723837, 0.07557802]), 'Jardim Botânico': array([-0.34296672, -0.06957464]), 'Lagoa': array([-0.2287956 , -0.22462745]), 'Leme': array([ 0.30783336, -0.43993586]), 'Ipanema': array([-0.13604816, -0.39443646]), 'Leblon': array([-0.42540793, -0.42112642]), 'Gávea': array([-0.55692957, -0.28087638]), 'Vidigal': array([-0.72900454, -0.46273238]), 'Rocinha': array([-0.8624544 , -0.36491218]), 'São Conrado': array([-1.        , -0.51200198])}
        #pos = {'Flamengo': array([0.6043461, 0.4442784]), 'Laranjeiras': array([0.45074005, 0.55273503]), 'Glória': array([0.8534418 , 0.58982338]), 'Catete': array([0.68333495, 0.64406827]), 'Cosme Velho': array([0.42134842, 0.85813163]), 'Urca': array([0.57723837, 0.07557802]), 'Lagoa': array([-0.2287956 , -0.22462745]), 'Leme': array([ 0.30783336, -0.43993586])}

        g = self.grafo
        
        if tipo == "arvore":
            mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}
            
            g = nx.relabel_nodes(self.grafo, mapping)

            if self.floripa:
                for vertice in g.nodes(data=True):
                    del vertice[1]["id"]
                    del vertice[1]["populaçao_s"]
                    del vertice[1]["populaçao_i"]
                    del vertice[1]["SIR_t0"]
                    del vertice[1]["SIRd"]
                    del vertice[1]["SIRdd"]
                    del vertice[1]["SIRddd"]
                    del vertice[1]["SIRdddantes"]
            else:
                for vertice in g.nodes(data=True):
                    del vertice[1]["id"]
                    del vertice[1]["populaçao"]
                    del vertice[1]["SIR_t0"]
                    del vertice[1]["SIRd"]
                    del vertice[1]["SIRdd"]
                    del vertice[1]["SIRddd"]
                    del vertice[1]["SIRdddantes"]
                    del vertice[1]["beta"]
        
            pos = graphviz_layout(g, prog="dot")

        #font_size=15
        nx.draw(g, pos, with_labels=True, font_weight='bold', font_size=11, node_size=300) #fonte 6 nodesize 200
        print(pos)
        plt.savefig(fr"Exemplo grafo zs novo.png", format="png", dpi=300)

        plt.show()  


        # plt.figure(figsize=(10,8))

        # #nx.draw( nx.draw(self.grafo, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)

        # plt.show()self.grafo, pos, with_labels=True, font_weight='bold', font_size=6, node_size=200, clip_on=True)
        #

    def printar_estado_vertice(self, vertice):
        print(f"{vertice}: {self.grafo.nodes[vertice]}")
    
    def printar_estados_vertices(self):
        for vertice in self.grafo.nodes:
            print(f"{vertice}: SIR_t0: {self.grafo.nodes[vertice]['SIR_t0']}\n"
                f"        SIRd:  {self.grafo.nodes[vertice]['SIRd']}\n"
                f"        SIRdd:  {self.grafo.nodes[vertice]['SIRdd']}\n"
                f"        SIRddd:  {self.grafo.nodes[vertice]['SIRddd']}")

    def  printar_grafico_SIRxT(self, x=None, y=None, path=None):
        fig = plt.figure(1)#.set_fig
        ax = fig.add_subplot(111)
        fig.set_size_inches([9, 6])


        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        if x:
            plt.xlim(left=1, right=len(x))
            plt.plot(x, y)
        else:
            plt.xlim(left=self.tempos[0], right=self.tempos[-1])
            plt.plot(self.tempos, self.SIRs)

        ax.legend(["S", "I", "R"], loc='center right', bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Pessoas')


        if path and not x:
            plt.savefig(path, format="png", bbox_inches='tight')
        elif x:
            inicio = path.split("\\")[-1]
            plt.title(f'Raiz: {inicio.split(".png")[0]}')
            plt.savefig(path, format="png", bbox_inches='tight')
        else:
            plt.show()
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
                if self.floripa:
                    pop[vertice[1]["populaçao_s"] + vertice[1]["populaçao_i"]] = vertice[0]

                else:
                    pop[vertice[1]["populaçao"]] = vertice[0]

                self.array_top_populaçao.append(vertice[1]["populaçao"])
            
            self.mergesort(self.array_top_populaçao)
            
            self.array_top_populaçao = [pop[self.array_top_populaçao[i]] for i in range(self.quantidade_bairros_selecionados)]

        if not self.isolados and not self.juntos:
            print("Isolando vértices!:", self.array_top_populaçao)

            for bairro in self.array_top_populaçao:
                self.grafo.nodes[bairro]["isolado"] = True

                for vizinho, populaçao_de_fora in self.grafo.nodes[bairro]["SIRddd"].items():   # mandar a pop de outros de dentro para fora
                    #print(vizinho,populaçao_de_fora)
                    try:
                        self.grafo.nodes[vizinho]["SIRdd"]["S"] += populaçao_de_fora["S"]
                        populaçao_de_fora["S"] = 0

                        self.grafo.nodes[vizinho]["SIRdd"]["I"] += populaçao_de_fora["I"]
                        populaçao_de_fora["I"] = 0

                        self.grafo.nodes[vizinho]["SIRdd"]["R"] += populaçao_de_fora["R"]
                        populaçao_de_fora["R"] = 0
                    except:
                        pass

                for vizinho in self.grafo.edges(bairro):
                    self.edges_isoladas.append(vizinho)

                self.grafo.remove_edges_from(self.edges_isoladas)

                for vizinho in self.grafo.edges(bairro):            # receber pop de fora de volta

                    try:
                        self.grafo.nodes[bairro]["SIRdd"]["S"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["S"]
                        self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["S"] = 0

                        self.grafo.nodes[bairro]["SIRdd"]["I"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["I"]
                        self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["I"] = 0

                        self.grafo.nodes[bairro]["SIRdd"]["R"] += self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["R"]
                        self.grafo.nodes[vizinho[1]]["SIRddd"][bairro]["R"] = 0
                    except:
                        pass

            
            self.isolados = True

        else:
            if self.juntos:
                print("População ainda não se movimentou!")
                return
            
            print("Tirando isolamento!")

            for aresta in self.edges_isoladas:
                self.grafo.add_edge(*aresta)
            
            self.edges_isoladas = []

            for bairro in self.array_top_populaçao:
                nome = bairro
                atributos = self.grafo.nodes[nome]
                atributos["isolado"] = False

                # S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                # I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                # R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                # atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                # atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                # atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))

                # for vizinho in self.grafo.edges(nome):
                #     self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}

            self.isolados = False

    def busca_em_largura(self, inicio):
        fila = []
        visitados = {inicio}
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

        self.grafo_original = self.grafo.copy()
        
        menor_media = 99999999
        arvore = 0
        quant_arvores = len(self.grafo_original.nodes)
        
        for inicio in self.grafo_original.nodes:
            self.estimar_tempo_restante(arvore, quant_arvores)
            arvore += 1
            self.vertice_de_inicio = inicio
            self.resetar_grafo()
            soma_pico = 0
            tempo_pico = 0

            anterior = self.busca_em_largura(inicio)

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
                self.grafo.add_edge(ant, vertice)


            for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)


            for i in range(iteraçoes):
                # qtd_arestas_iguais = 0            # qtd arestas iguais a saude
                # if inicio == "Saúde":
                #     self.arestas_primeira_arvore = (self.grafo.copy()).edges()
                # else:
                #     for aresta in self.grafo.edges():
                #         if aresta in self.arestas_primeira_arvore:
                #             qtd_arestas_iguais += 1
                #print(self.grafo.nodes[inicio]["id"], "| Arestas iguais à Saúde:", qtd_arestas_iguais, "/158")

                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao_dinamica(tempo)
                print("Pico:", self.pico_infectados)

                soma_pico += self.pico_infectados
                tempo_pico += self.tempo_pico

                self.resetar_grafo()
            
            tempo_pico = tempo_pico / (i + 1)
            media = soma_pico / (i + 1)
            self.resultados_arvore_largura.write(f"{self.grafo.nodes[inicio]['id']}, {tempo_pico}, {media}\n")   

            self.SIRxTdeVerticesTXT_largura.write(f"{inicio}, ")

            self.SIRxTdeVerticesTXT_largura.write(f"{self.SIRxTdeVertices}\n")
                
            menor_media = media if media < menor_media else menor_media
        
        self.resultados_arvore_largura.write(f"\nMenor média: {menor_media}")
        self.resultados_arvore_largura.close()
        self.SIRxTdeVerticesTXT_largura.close()
        self.grafo = self.grafo_original 

            
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

    def busca_em_profundidade(self, v, anterior, visitados, niveis, nivel=0):
        visitados.add(v)
        niveis[v] = nivel

        for vizinho in self.grafo_original.edges(v):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                anterior[vizinho] = v
                self.busca_em_profundidade(vizinho, anterior, visitados, niveis, nivel+1)

    def gerar_grafos_arvore_profundidade(self, tempo, iteraçoes):
        self.resultados_arvore_profundidade = open(resultados_arvore_profundidade, "w", encoding="utf-8")
        self.SIRxTdeVerticesTXT_profundidade = open(SIRxTdeVerticesTXT_profundidade, "w", encoding="utf-8")
        
        self.grafo_original = self.grafo.copy()
        
        menor_media = 99999999
        arvore = 0
        quant_arvores = len(self.grafo_original.nodes)
        for inicio in self.grafo_original.nodes:
            self.estimar_tempo_restante(arvore, quant_arvores)
            arvore += 1
            self.vertice_de_inicio = inicio
            self.resetar_grafo()
            soma_pico = 0
            tempo_pico = 0

            anterior = {}
            visitados = set()
            self.busca_em_profundidade(inicio, anterior, visitados)

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
                self.grafo.add_edge(ant, vertice)


            for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)
            

            for i in range(iteraçoes):
                # qtd_arestas_iguais = 0            # qtd arestas iguais a saude
                # if inicio == "Saúde":
                #     self.arestas_primeira_arvore = (self.grafo.copy()).edges()
                # else:
                #     for aresta in self.grafo.edges():
                #         if aresta in self.arestas_primeira_arvore:
                #             qtd_arestas_iguais += 1
                #print(self.grafo.nodes[inicio]["id"], "| Arestas iguais à Saúde:", qtd_arestas_iguais, "/158")

                print("\nInicio:", inicio, "| Iteração:", i+1)

                self.avançar_tempo_movimentacao_dinamica(tempo)
                print("Pico:", self.pico_infectados)

                soma_pico += self.pico_infectados
                tempo_pico += self.tempo_pico

                self.resetar_grafo()
            
            tempo_pico = tempo_pico / (i + 1)
            media = soma_pico / (i + 1)
            #self.salvar_grafo_arvore(fr"C:\Users\rasen\Desktop\Resultados\arvores profundidade\{inicio}.png")
            self.resultados_arvore_profundidade.write(f"{self.grafo.nodes[inicio]['id']}, {tempo_pico}, {media}\n")      
            
            self.SIRxTdeVerticesTXT_profundidade.write(f"{inicio}, ")

            self.SIRxTdeVerticesTXT_profundidade.write(f"{self.SIRxTdeVertices}\n")

            menor_media = media if media < menor_media else menor_media

        self.resultados_arvore_profundidade.write(f"\nMenor média: {menor_media}")
        self.resultados_arvore_profundidade.close()
        self.SIRxTdeVerticesTXT_profundidade.close()
        self.grafo = self.grafo_original 

    def printar_grafico_ID_MAXINFECT_arvore(self, tipo_arvore):
        if tipo_arvore == "largura":
            resultados = open("./Resultados/resultados_arvore_largura.txt", "r")
        else:
            resultados = open("./Resultados/resultados_arvore_profundidade.txt", "r")

        titulo = f'Picos de Infectados das Árvores de Busca em {tipo_arvore.title()}'
        resultados_lista = [x for x in range(160)]


        for linha in resultados:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)] = [int(float(max_infect))]

        resultados_lista[0] = 816398 # INICIO FLAMENGO    #1651756 # resultado original mudar

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([15, 7.5])

        plt.xlim(left=-5, right=164)
        plt.xticks([x for x in range(0, 160, 4)])
        plt.yticks([x for x in range(0, 1000001, 100000)])
        
        plt.plot(0, resultados_lista[0], "o", color="red")      # valor grafo normal
        resultados_lista.pop(0)

        plt.gca().set_prop_cycle('color', ['0d66a3'])
        plt.plot([x for x in range(1, 160)], resultados_lista, "o")    # valores arvores
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend(["Grafo Original", tipo_arvore.title()], loc='center right', bbox_to_anchor=(1.130, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Original)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(fr"C:\Users\rasen\Desktop\Pico Infectados Arvores {tipo_arvore.title()} FINAL.png", format="png", dpi=300)
        plt.close()

    def printar_grafico_ID_MAXINFECT_arvores_largura_profundidade(self):

        resultadosL = open("./Resultados/resultados_arvore_largura.txt", "r")
        resultadosP = open("./Resultados/resultados_arvore_profundidade.txt", "r")
        titulo = f'Picos de Infectados das Árvores de Busca em Largura e Profundidade'

        resultados_lista = [x for x in range(160)]

        for linha in resultadosL:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)] = [int(float(max_infect))]


        for linha in resultadosP:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)].append(int(float(max_infect)))

        resultados_lista[0] = 816398 # INICIO FLAMENGO    #1651756 # resultado original mudar


        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([15, 7.5])

        plt.xlim(left=-5, right=164)
        plt.xticks([x for x in range(0, 160, 4)])
        plt.yticks([x for x in range(0, 1000001, 100000)])
        
        plt.plot(0, resultados_lista[0], "o", color="red")      # valor grafo normal
        resultados_lista.pop(0)

        plt.gca().set_prop_cycle('color', ['green', '0d66a3'])

        plt.plot([x for x in range(1, 160)], resultados_lista, "o")    # valores arvores
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend(["Grafo Original", "Largura", "Profundidade"], loc='center right', bbox_to_anchor=(1.130, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Original)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(fr"C:\Users\rasen\Desktop\Pico Infectados Arvores Largura e Profundidade 400 dias FINAL.png", format="png", dpi=300)

    def avançar_tempo(self, t):
        #self.printar_estados_vertices()

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

                # X_2pontos = X_2pontos_ii + X_2pontos_ji (somatorio Y_2pontos_ii(=Y_ponto) + somatorio dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                for i in range(atributos["SIRdd"]["S"]):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X_2pontos_ii += random() < Y_ponto                      # baseado na probabilidade de Y_ponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (2 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                for vizinho in atributos["SIRddd"].values():                # SIR t+1 vizinhos
                    X_2pontos_ji = 0

                    for i in range(vizinho["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                        X_2pontos_ji += random() < Y_ponto                                            # baseado na probabilidade de Y_2pontos
                        
                    recuperados_novos_ddd = ceil(self.e * vizinho["I"])

                    vizinho["S"] = vizinho["S"] - floor(self.v * X_2pontos_ji)
                    vizinho["I"] = vizinho["I"] - recuperados_novos_ddd + floor(self.v * X_2pontos_ji)
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

    def avançar_tempo_movimentacao_dinamica(self, t, printar=True):    # s = nome vertice de origem (no caso de utilizar um grafo arvore)
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))

        #self.resetar_grafo()
        for tempo in range(t):
            if printar:
                print(self.t)
            # distribuiçao de pessoas
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node
                if not atributos["isolado"]:
                    try:
                        self.SIRxTdeVertices[nome][self.t] = [[
                            [atributos["SIRd"]["S"], atributos["SIRdd"]["S"]],
                            [atributos["SIRd"]["I"], atributos["SIRdd"]["I"]],
                            [atributos["SIRd"]["R"], atributos["SIRdd"]["R"]],
                            [atributos["SIRddd"].copy()]
                        ]]
                    except:
                        self.SIRxTdeVertices[nome] = dict()
                        self.SIRxTdeVertices[nome][self.t] = [[
                            [atributos["SIRd"]["S"], atributos["SIRdd"]["S"]],
                            [atributos["SIRd"]["I"], atributos["SIRdd"]["I"]],
                            [atributos["SIRd"]["R"], atributos["SIRdd"]["R"]],
                            [atributos["SIRddd"].copy()]
                        ]]
                        # self.SIRxTdeVertices[nome][self.t] = [
                        #     atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        #     atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        #     atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                        # ]


                    S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                    I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                    R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                    atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                    atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                    atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))


                    for vizinho in self.grafo.edges(nome):
                        if not self.grafo.nodes[vizinho[1]]["isolado"]:
                            self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}
                        else:
                            self.grafo.nodes[vizinho[1]]["SIRddd"] = {}
                            atributos["SIRdd"]["S"] += S_2pontos_saindo
                            atributos["SIRdd"]["I"] += I_2pontos_saindo
                            atributos["SIRdd"]["R"] += R_2pontos_saindo

                    #try:
                    # except:
                    #     self.SIRxTdeVertices[nome] = dict()
                    #     self.SIRxTdeVertices[nome][self.t].append([
                    #         [atributos["SIRd"]["S"], atributos["SIRdd"]["S"]],
                    #         [atributos["SIRd"]["I"], atributos["SIRdd"]["I"]],
                    #         [atributos["SIRd"]["R"], atributos["SIRdd"]["R"]],
                    #         [atributos["SIRddd"]]
                    #     ])

            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node
                #print(nome,  [atributos["SIRddd"]])
                self.SIRxTdeVertices[nome][self.t].append([
                        [atributos["SIRd"]["S"], atributos["SIRdd"]["S"]],
                        [atributos["SIRd"]["I"], atributos["SIRdd"]["I"]],
                        [atributos["SIRd"]["R"], atributos["SIRdd"]["R"]],
                        [atributos["SIRddd"].copy()]
                    ])
                
            self.tempos.append(self.t)

            for node in list(self.grafo.nodes(data=True)):
                # x = tempo
                # y = [S, I, R]
                nome, atributos = node

                atributos["SIRdddantes"]["S"] = atributos["SIRdddantes"]["I"] = atributos["SIRdddantes"]["R"] = 0

                for vizinho in atributos["SIRddd"].values():
                    try:
                        atributos["SIRdddantes"]["S"] += vizinho["S"]
                        atributos["SIRdddantes"]["I"] += vizinho["I"]
                        atributos["SIRdddantes"]["R"] += vizinho["R"]
                    except:
                        pass

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

                # X_2pontos = X_2pontos_ii + X_2pontos_ji (somatorio Y_2pontos_ii(=Y_ponto) + somatorio dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                for i in range(atributos["SIRdd"]["S"]):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X_2pontos_ii += random() < Y_ponto                      # baseado na probabilidade de Y_ponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (2 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                
                if not atributos["isolado"]:
                    for vizinho, atributos_v in atributos["SIRddd"].items():                # SIR t+1 vizinhos
                        if not self.grafo.nodes[vizinho]["isolado"]:
                            X_2pontos_ji = 0

                            for i in range(atributos_v["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                                X_2pontos_ji += random() < Y_ponto                                            # baseado na probabilidade de Y_2pontos
                                
                            recuperados_novos_ddd = ceil(self.e * atributos_v["I"])

                            atributos_v["S"] = atributos_v["S"] - floor(self.v * X_2pontos_ji)
                            atributos_v["I"] = atributos_v["I"] - recuperados_novos_ddd + floor(self.v * X_2pontos_ji)
                            atributos_v["R"] = atributos_v["R"] + recuperados_novos_ddd
                
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
            if self.SIRs[self.t-1][1] - self.SIRs[self.t-2][1] > 5000 and printar:
                print("diferença maior que 5000")

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
            if t in [1, 50, 75, 200]:
                print("imagem tempo", t)
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
                plt.text(x=0.5, y=1.005, s=f"Dia {t}", fontsize=70, ha="center", transform=fig.transFigure)
                #plt.suptitle(, y=1.02, fontsize='xx-large')
                plt.savefig(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Grafico pizza grafo dinamico flamengo\tempo {t}.png", format="png", bbox_inches="tight")
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
        self.SIRxTdeVerticesTXT.close()

    def avançar_tempo_movimentacao_dinamica_nao_discreto(self, deltaT: float, t: int):
          # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))

        for tempoAtual in range(1, int(t/deltaT) + 1):
            print(round(deltaT * tempoAtual, ndigits=1))
            # distribuiçao de pessoas
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                try:
                    self.SIRxTdeVertices[nome][tempoAtual] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]
                except:
                    self.SIRxTdeVertices[nome] = dict()
                    self.SIRxTdeVertices[nome][tempoAtual] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]

                #? ALTERAR?
                S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))


                for vizinho in self.grafo.edges(nome):
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}


            self.tempos.append(round(deltaT * tempoAtual, ndigits=1))

            for node in list(self.grafo.nodes(data=True)):
                # x = tempo
                # y = [S, I, R]
                nome, atributos = node


                #? COLOCAR ESSE LOOP NO LOOP SEGUINTE
                atributos["SIRdddantes"]["S"] = atributos["SIRdddantes"]["I"] = atributos["SIRdddantes"]["R"] = 0

                for vizinho in atributos["SIRddd"].values():
                    atributos["SIRdddantes"]["S"] += vizinho["S"]
                    atributos["SIRdddantes"]["I"] += vizinho["I"]
                    atributos["SIRdddantes"]["R"] += vizinho["R"]

            self.SIRs.append([])
            soma_SIR = [0, 0, 0]

            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node
                #? AQUI
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

                # X_2pontos = X_2pontos_ii + X_2pontos_ji (somatorio Y_2pontos_ii(=Y_ponto) + somatorio dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                #? ADICIONADO FLOOR JA QUE S PODE SER FLOAT
                for i in range(floor(atributos["SIRdd"]["S"])):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X_2pontos_ii += random() < Y_ponto                      # baseado na probabilidade de Y_ponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (2 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                for vizinho in atributos["SIRddd"].values():                # SIR t+1 vizinhos
                    X_2pontos_ji = 0

                    for i in range(vizinho["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                        X_2pontos_ji += random() < Y_ponto                                            # baseado na probabilidade de Y_2pontos
                        
                    #? ALTERADO
                    recuperados_novos_ddd = deltaT * (self.e * vizinho["I"])

                    vizinho["S"] = vizinho["S"] - deltaT * (self.v * X_2pontos_ji)
                    vizinho["I"] = vizinho["I"] - recuperados_novos_ddd + deltaT * (self.v * X_2pontos_ji)
                    vizinho["R"] = vizinho["R"] + recuperados_novos_ddd
        
                # SIR t+1 SIRponto e SIR2pontos
                #? ALTERADO
                recuperados_novos_d = deltaT * (self.e * atributos["SIRd"]["I"])
                recuperados_novos_dd = deltaT * (self.e * atributos["SIRdd"]["I"])

                atributos["SIRd"]["S"] = atributos["SIRd"]["S"] - deltaT * (self.v * X_ponto)              # X = quantidade total de encontros, v*X pessoas suscetiveis de fato infectadas
                atributos["SIRdd"]["S"] = atributos["SIRdd"]["S"] - deltaT * (self.v * X_2pontos_ii)

                atributos["SIRd"]["I"] = atributos["SIRd"]["I"] - recuperados_novos_d + deltaT * (self.v * X_ponto)
                atributos["SIRdd"]["I"] = atributos["SIRdd"]["I"] - recuperados_novos_dd + deltaT * (self.v * X_2pontos_ii)

                atributos["SIRd"]["R"] = atributos["SIRd"]["R"] + recuperados_novos_d
                atributos["SIRdd"]["R"] = atributos["SIRdd"]["R"] + recuperados_novos_dd

            self.SIRs[tempoAtual-1] = [soma_SIR[0], soma_SIR[1], soma_SIR[2]]

            if not any(i[1] > soma_SIR[1] for i in self.SIRs):
                self.tempo_pico = deltaT * tempoAtual
                self.pico_infectados = soma_SIR[1]


            for node in list(self.grafo.nodes(data=True)):          # voltar pessoas para o proprio vertice
                nome, atributos = node

                for vizinho in self.grafo.edges(nome):
                    node_vizinho = self.grafo.nodes[vizinho[1]]["SIRddd"][nome]
                    atributos["SIRdd"]["S"] += node_vizinho["S"]
                    atributos["SIRdd"]["I"] +=  node_vizinho["I"]
                    atributos["SIRdd"]["R"] += node_vizinho["R"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {}

    def printar_grafico_ID_MAXINFECT_arvores_profundidade_antes_depois(self):
        resultadosAntes = open("./Resultados/Graficos SIRxT arvores profundidade 200/resultados_arvore_profundidade.txt", "r")
        resultadosDepois = open("./Resultados/Graficos SIRxT arvores profundidade 400/resultados_arvore_profundidade.txt", "r")

        titulo = f'Picos de Infectados das Árvores de Busca em Profundidade'
        resultados_lista = [x for x in range(160)]


        for linha in resultadosAntes:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)] = [int(float(max_infect))]


        for linha in resultadosDepois:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)].append(int(float(max_infect)))

        resultados_lista[0] = 816398 # INICIO FLAMENGO    #1651756 # resultado original


        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([15, 7.5])

        plt.xlim(left=-5, right=164)
        plt.xticks([x for x in range(0, 160, 4)])
        plt.yticks([x for x in range(0, 900001, 100000)])
        
        plt.plot(0, resultados_lista[0], "o", color="red")      # valor grafo normal
        resultados_lista.pop(0)

        plt.gca().set_prop_cycle('color', ['brown', '0d66a3'])
        #plt.gca().set_prop_cycle('color', ['0d66a3'])
        plt.plot([x for x in range(1, 160)], resultados_lista, "o")    # valores arvores
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend(["Grafo Original", "Antes", "Depois"], loc='center right', bbox_to_anchor=(1.130, 0.5))
        #ax.legend(["Grafo Original", tipo_arvore.title()], loc='center right', bbox_to_anchor=(1.127, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Original)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        #C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Pico Infectados Arvores {tipo_arvore.title()}.png
        plt.savefig(fr"C:\Users\rasen\Desktop\Resultados\com betas\Pico Infectados Arvores Profundidade 200x400 dias.png", format="png", dpi=300)
        plt.close

    def heuristica_arvores_vizinhas(self, tipo_arvore, path_arquivo_log: str, path_arquivo_picos: str):
        path_arquivo_log = f"{path_arquivo_log}_{tipo_arvore}.txt"
        path_arquivo_picos = f"{path_arquivo_picos}_{tipo_arvore}.txt"

        arquivo_log = open(path_arquivo_log, "a", encoding="utf-8", buffering=1)
        arquivo_picos = open(path_arquivo_picos, "a", encoding="utf-8", buffering=1)
        arquivo_picos.write(f"Inicio da arvore:\n v-u adicionada (cria ciclo) pico dia_pico fim_espalhamento\n  x-y removida (volta a ser arvore) pico dia_pico fim_espalhamento\n")

        self.grafo_original = self.grafo.copy()
        arvore = 0
        quant_arvores = len(self.grafo_original.nodes)
        indice_arvore = 0

        for inicio in self.grafo_original.nodes:                # para cada arvore original
            arquivo_picos.write(f"\nInicio {inicio}:\n")

            #self.estimar_tempo_restante(arvore, quant_arvores)
            arvore += 1
            self.vertice_de_inicio = inicio
            self.resetar_grafo()
            print("Inicio:", inicio)

            anterior = {}
            niveis = {}
            if tipo_arvore == "largura":
                anterior = self.busca_em_largura(inicio)
            else:
                visitados = set()
                self.busca_em_profundidade(inicio, anterior, visitados, niveis, 0)

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            if self.floripa:
                for vertice, ant in anterior.items():
                    self.grafo.add_weighted_edges_from([(ant, vertice, self.grafo_original.get_edge_data(ant, vertice)["beta"]), (vertice, ant, self.grafo_original.get_edge_data(vertice, ant)["beta"])], weight="beta")
            else:
                for vertice, ant in anterior.items():
                    self.grafo.add_edge(ant, vertice)

                for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                    vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)


            grafo_complemento = nx.complement(self.grafo).edges()
            print(len(grafo_complemento))

            for v, u in grafo_complemento:      # adiçao da aresta que cria ciclo
                print("ADIÇÃO ARESTA: ", end="")
                self.estimar_tempo_restante(indice_arvore, len(self.grafo_original.nodes)*len(grafo_complemento))
                indice_arvore += 1

                #anterior = {}
                #visitados = set()
                #self.encontrou_ciclo = False
                ciclo = []

                # def achar_ciclo(k, anterior, visitados):          # achar ciclo
                #     visitados.add(k)
                #     for vizinho in self.grafo.edges(k):
                #         if not self.encontrou_ciclo:
                #             vizinho = vizinho[1]

                #             if vizinho not in visitados:
                #                 anterior[vizinho] = k
                #                 if vizinho == v:
                #                     print("Encontrou ciclo:", end="")
                #                     self.encontrou_ciclo = True
                #                     break
                #                 achar_ciclo(vizinho, anterior, visitados)

                def achar_ciclo_a(anterior):
                    # w vertice de maior nivel, z menor
                    w, z = (v, u) if (max(niveis[v], niveis[u]), min(niveis[v], niveis[u])) == (niveis[v], niveis[u]) else (u, v)

                    nivel = niveis[w]

                    while nivel > niveis[z]:
                        ciclo.append(w)
                        w = anterior[w]
                        nivel -= 1
                    
                    #print("W e Z mesmo nivel:", w, z)
                    
                    ciclo2 = []
                    while w != z:
                        ciclo.append(w)
                        w = anterior[w]
                        ciclo2.append(z)
                        z = anterior[z]
                    ciclo.append(w)     # raiz
                    print("ciclo1:", ciclo, "ciclo2", ciclo2)
                    if ciclo2:
                        ciclo2.reverse()            # reverse n retorna nada
                        ciclo.extend(ciclo2)  # append outro lado do caminho na arvore
                    #print("W e Z dps de achar ciclo:", w, z)
                    #with self.terminal.location(x=0, y=self.terminal.height-4):
                    print("Criando", v, "-->", u, "/ ciclo:", ciclo)

                    # salvar niveis dos vertices na arvore (pair<nivel, vertice>)
                    # busca em largura e profundidade linha 1170 - 1173
                    # igualar niveis dos vertices v, u atraves dos anteriores 
                    # depois ir voltando com os dois ate chegarem em um antecessor comum
                    # salvar caminho de v em um array normal e de u em um array de forma inversa,
                    # depois juntar colocando vertice igual no final do primeiro
                
                achar_ciclo_a(anterior)
                #achar_ciclo(u, anterior, visitados)

                # ciclo.append(v)
                # while True:                 # montar ciclo
                #     try:
                #         if anterior[ciclo[-1]] != v:
                #             ciclo.append(anterior[ciclo[-1]])
                #         else:
                #             raise Exception
                #     except:
                #         break

                #print(ciclo)
                #self.printar_grafo("arvore")
                if self.floripa:
                    self.grafo.add_weighted_edges_from([(v, u, self.peso_medio), (u, v, self.peso_medio)], weight="beta")
                else:
                    self.grafo.add_edge(v, u)
                    for vertice in [v, u]:          # atualizar betas com a nova aresta
                        self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)


                self.resetar_grafo()
                self.avançar_tempo_movimentacao_dinamica_otimizado(printar=True)
                arquivo_picos.write(f" {v}-{u} {self.pico_infectados} {self.tempo_pico} {self.t}\n")
                arquivo_log.write(f"{self.SIRxTdeVertices}\n")

                for indice in range(0, len(ciclo) - 1):    # loop arestas do ciclo (menos v, u)
                    self.resetar_grafo()
                    x = ciclo[indice]
                    y = ciclo[indice + 1]
                    #with self.terminal.location(y=self.terminal.height-4):
                    print(f"Removendo {x} -> {y}")

                    self.grafo.remove_edges_from([(x, y), (y, x)])

                    #! rodar modelo
                    self.avançar_tempo_movimentacao_dinamica_otimizado(printar=True)

                    arquivo_picos.write(f"  {x}-{y} {self.pico_infectados} {self.tempo_pico} {self.t}\n")
                    arquivo_log.write(f"{self.SIRxTdeVertices}\n")
                    #self.tempo_pico, self.pico_infectados, self.SIRxTdeVertices
                    # salvar SIRT (criar outro modelo com SIRT diferente pra cá e otimizado)
                    # salvar resultado (id, dia, pico)
                    
                    if self.floripa:
                        self.grafo.add_weighted_edges_from([(x, y, self.grafo_original.get_edge_data(x, y)["beta"]), (y, x, self.grafo_original.get_edge_data(y, x)["beta"])], weight="beta")
                    else:
                        self.grafo.add_edge(x, y)
                        for vertice in [x, y]:          # atualizar betas com nova aresta
                            self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)
                    # salvar graficos SIR (arvore com v, u sem x, y.png)

                self.grafo.remove_edges_from([(v, u), (u, v)])

            # self.grafo.remove_edges_from(list(self.grafo.edges()))
            # self.grafo.add_edges_from(g.edges())
            # print(len(self.grafo.edges()))
            # print(self.grafo.nodes(data=True))
            # self.resetar_grafo()
            # print(self.grafo.nodes(data=True))


            # salvar arvore original
            self.salvar_grafo_arvore(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\heuristica\arvores\{inicio}.png")
        self.grafo = self.grafo_original

    def avançar_tempo_movimentacao_dinamica_otimizado(self, printar=True):    # s = nome vertice de origem (no caso de utilizar um grafo arvore)
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))
        soma_SIR = [0,1,0]
        while True:
            # distribuiçao de pessoas
            #self.printar_estado_vertice("Grupo 1")
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                if self.t != 1:          # antes da distribuiçao de pessoas salva estado de cada vértice
                    self.SIRxTdeVertices[nome][self.t] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]
                else:            
                    self.SIRxTdeVertices[nome] = dict()
                    self.SIRxTdeVertices[nome][self.t] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]

                if self.floripa:
                    #print(self.grafo.edges(data=True))
                    for vizinho in self.grafo.edges(nome, data=True):
                        #print(vizinho)
                        S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * vizinho[2]["beta"])
                        I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * vizinho[2]["beta"])
                        R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * vizinho[2]["beta"])

                        atributos["SIRdd"]["S"] -= S_2pontos_saindo
                        atributos["SIRdd"]["I"] -= I_2pontos_saindo
                        atributos["SIRdd"]["R"] -= R_2pontos_saindo

                        self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}

                else:
                    S_2pontos_saindo = floor(atributos["SIRdd"]["S"] * atributos["beta"])
                    I_2pontos_saindo = floor(atributos["SIRdd"]["I"] * atributos["beta"])
                    R_2pontos_saindo = floor(atributos["SIRdd"]["R"] * atributos["beta"])

                    atributos["SIRdd"]["S"] -= S_2pontos_saindo * len(self.grafo.edges(nome))
                    atributos["SIRdd"]["I"] -= I_2pontos_saindo * len(self.grafo.edges(nome))
                    atributos["SIRdd"]["R"] -= R_2pontos_saindo * len(self.grafo.edges(nome))


                    for vizinho in self.grafo.edges(nome):
                        self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {"S": S_2pontos_saindo, "I": I_2pontos_saindo, "R": R_2pontos_saindo}
             
            if printar:
                #with self.terminal.location(y=self.terminal.height-2):
                print("tempo=", self.t)

            self.tempos.append(self.t)
            self.SIRs.append([])
            soma_SIR = [0, 0, 0]

            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                atributos["SIRdddantes"]["S"] = atributos["SIRdddantes"]["I"] = atributos["SIRdddantes"]["R"] = 0

                for vizinho in atributos["SIRddd"].values():
                    atributos["SIRdddantes"]["S"] += vizinho["S"]
                    atributos["SIRdddantes"]["I"] += vizinho["I"]
                    atributos["SIRdddantes"]["R"] += vizinho["R"]

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

                # X_2pontos = X_2pontos_ii + X_2pontos_ji (somatorio Y_2pontos_ii(=Y_ponto) + somatorio dos que vieram dos vizinhos e, portanto, usam a probabilidade Y deste vertice)
                for i in range(atributos["SIRdd"]["S"]):                    # calculo do numero de encontros na pop de suscetiveis que nao respeitam e restam no vertice
                    X_2pontos_ii += random() < Y_ponto                      # baseado na probabilidade de Y_ponto


                # Cada grupo do SIR (respeitam, nao respeitam, vizinhos) tem probabilidades unicas (2 loops acima)

                # prob de acontecer um encontro de Sddd (estrangeiros) com Infectados nesse vertice
                # (do ponto de vista do vertice vizinho = Ypi -> j = Ypj, ou seja, do ponto de vista desse vertice = Ypi)
                for vizinho in atributos["SIRddd"].values():                # SIR t+1 vizinhos
                    X_2pontos_ji = 0

                    for i in range(vizinho["S"]):       # calculo do numero de encontros na pop de suscetiveis que vem de outros vertices
                        X_2pontos_ji += random() < Y_ponto                                            # baseado na probabilidade de Y_2pontos
                        
                    recuperados_novos_ddd = ceil(self.e * vizinho["I"])

                    vizinho["S"] = vizinho["S"] - floor(self.v * X_2pontos_ji)
                    vizinho["I"] = vizinho["I"] - recuperados_novos_ddd + floor(self.v * X_2pontos_ji)
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
            if printar:
                #with self.terminal.location(y=self.terminal.height-3):
                print(soma_SIR)

            if not any(i[1] > soma_SIR[1] for i in self.SIRs):
                self.tempo_pico = self.t
                self.pico_infectados = soma_SIR[1]

            if soma_SIR[1] == 0:
                break

            for node in list(self.grafo.nodes(data=True)):          # voltar pessoas para o proprio vertice
                nome, atributos = node

                for vizinho in self.grafo.edges(nome):
                    node_vizinho = self.grafo.nodes[vizinho[1]]["SIRddd"][nome]
                    atributos["SIRdd"]["S"] += node_vizinho["S"]
                    atributos["SIRdd"]["I"] +=  node_vizinho["I"]
                    atributos["SIRdd"]["R"] += node_vizinho["R"]
                    self.grafo.nodes[vizinho[1]]["SIRddd"][nome] = {}

            self.t += 1

    def printar_tabela_arvores(self):
        #SIRxTdeVerticesTXT_largura = open("./Resultados/SIR_vertice_por_tempo_LARGURA.txt", "r", encoding="utf-8")
        #SIRxTdeVerticesTXT_profundidade = open("./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt", "r", encoding="utf-8")
        resultadosL = open("./Resultados/resultados_arvore_largura.txt", "r")
        resultadosP = open("./Resultados/resultados_arvore_profundidade.txt", "r")

        tabela_profundidade = "./Resultados/tabela_arvores_profundidade.xlsx"
        tabela_largura = "./Resultados/tabela_arvores_largura.xlsx"

        resultados_lista = [x for x in range(160)]
        wb = openpyxl.Workbook()
        wb.create_sheet("Largura")

        with pd.ExcelWriter(tabela_largura, engine="openpyxl") as writer:
            writer.book = wb
            lista_inicio = []
            lista_resultados = []
            for linha in resultadosL:
                linha = linha.strip()

                if linha == "":
                    break

                id, dia_pico, max_infect = linha.split(", ")
                resultados_lista[int(id)] = [int(float(dia_pico)), int(float(max_infect))]


            for linha in open("./Resultados/SIR_vertice_por_tempo_LARGURA.txt", "r", encoding="utf-8"):
                inicio, dicionario_dados = linha.split(", ", maxsplit=(1))
                SIRxTdeVerticesTXT_largura = ast.literal_eval(dicionario_dados)

                infectados_em_t = []
                dia_fim = 0

                largura_pico = 0
                pico_terminou = False
                pico = 0

                terminou = False
                print("Inicio:", inicio)
                for t in range(1, 401):
                    infectados_vertices = [int(SIRxT[t][1]) for SIRxT in SIRxTdeVerticesTXT_largura.values()]

                    soma_infectados = sum(infectados_vertices)
                    infectados_em_t.append(soma_infectados)

                    if t == resultados_lista[self.grafo.nodes[inicio]["id"]][0]:
                        print("a")
                        tempo = t
                        while abs(infectados_em_t[tempo-1] - infectados_em_t[tempo-2]) > 5000:
                            largura_pico += 1
                            tempo -= 1

                        tempo = t
                        infectados_em_t2 = []
                        soma_infectados = sum([int(SIRxT[tempo][1]) for SIRxT in SIRxTdeVerticesTXT_largura.values()])
                        infectados_em_t2.append(soma_infectados)

                        while abs(infectados_em_t2[tempo-t] - sum([int(SIRxT[tempo+1][1]) for SIRxT in SIRxTdeVerticesTXT_largura.values()])) > 5000:
                            largura_pico += 1
                            tempo += 1


                    print(f"{t}:", abs(infectados_em_t[t-1] - infectados_em_t[t-2]), "t-2:", infectados_em_t[t-2], "t-1:", infectados_em_t[t-1])
                    if soma_infectados == 0:
                        dia_fim = t
                        print("FIM: ", t)
                        break
                lista_inicio.append(inicio)
                lista_resultados.append([resultados_lista[self.grafo.nodes[inicio]["id"]][0], dia_fim if dia_fim else "200+", largura_pico])

                df = pd.DataFrame(lista_resultados,
                            index=lista_inicio, columns=["Dia Pico", "Dia fim do espalhamento", "Largura pico"])
                
                df.to_excel(writer, sheet_name="Largura")


            lista_inicio = []
            lista_resultados = []

            for linha in resultadosP:
                linha = linha.strip()

                if linha == "":
                    break

                id, dia_pico, max_infect = linha.split(", ")
                resultados_lista[int(id)] = [int(float(dia_pico)), int(float(max_infect))]

            for linha in open("./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt", "r", encoding="utf-8"):
                inicio, dicionario_dados = linha.split(", ", maxsplit=(1))
                SIRxTdeVerticesTXT_profundidade = ast.literal_eval(dicionario_dados)

                dia_fim = 0
                terminou = False
                print("Inicio:", inicio)
                for t in range(1, 401):
                    if all(int(SIRxT[t][1]) == 0 for SIRxT in SIRxTdeVerticesTXT_profundidade.values()):
                        dia_fim = t
                        print("FIM: ", t)
                        break
                lista_inicio.append(inicio)
                lista_resultados.append([resultados_lista[self.grafo.nodes[inicio]["id"]][0], dia_fim if dia_fim else "400+", 0])

                df = pd.DataFrame(lista_resultados,
                            index=lista_inicio, columns=["Dia Pico", "Dia fim do espalhamento", "Largura pico"])
                
                df.to_excel(writer, sheet_name="Profundidade")

    def montar_tabela_zona_sul_ciclos(self):
        #bairros_fora_ciclo = ("Cosme Velho", "Urca", "Lagoa", "Leme")
        ciclo1 = [('Flamengo', "Botafogo"), ('Flamengo', "Glória"), ("Glória", "Catete"), ("Catete", "Laranjeiras"), ("Botafogo", "Laranjeiras")]
        ciclo2 = [('Botafogo', 'Humaitá'), ('Botafogo', 'Copacabana'), ('Humaitá', 'Jardim Botânico'), ('Copacabana', 'Ipanema'), ('Jardim Botânico', 'Gávea'), ('Ipanema', 'Leblon'), ('Leblon', 'Vidigal'), ('Vidigal', 'São Conrado'), ('Gávea', 'Rocinha'), ('Rocinha', 'São Conrado')]

        inicios = []

        #txt_resultados = open("Resultados")
        g = self.grafo.copy()
        for inicio in self.grafo.nodes():
            picos_infectados = []
            arestas_removidas = []
            inicios.append(inicio)

            print("Inicio:", inicio)
            self.vertice_de_inicio = inicio

            for arestaA1, arestaB1 in ciclo1:
                print(f"{arestaA1}-{arestaB1}")
                self.grafo.remove_edge(arestaA1, arestaB1)
        
                for arestaA2, arestaB2 in ciclo2:
                    print(f"    {arestaA2}-{arestaB2}", end=" ")
                    self.grafo.remove_edge(arestaA2, arestaB2)

                    self.avançar_tempo_movimentacao_dinamica(200)
                    
                    picos_infectados.append(self.pico_infectados)
                    print(self.pico_infectados)
                    arestas_removidas.append(f"{arestaA1}-{arestaB1} {arestaA2}-{arestaB2}")

                    self.grafo.add_edge(arestaA2, arestaB2)
                
                self.grafo.add_edge(arestaA1, arestaB1)
        
        # with pd.ExcelWriter(tabela_ciclos_zs, engine="openpyxl") as writer:
        #     writer.book = wb

        #     df = pd.DataFrame(lista_resultados,
        #                 index=inicios, columns=colunas)
        
        # df.to_excel(writer, sheet_name="Profundidade")


    def printar_grafico_SIR_t0_VerticePizza(self, path=None, dia=1, v="Flamengo"):
        vertice = v


        sir_t0_antes = self.SIRxTdeVertices[vertice][dia][0]
        
        sir_t0_depois = self.SIRxTdeVertices[vertice][dia][1]
        print(sir_t0_depois)
        sir_t0_depois[0][0] = floor(self.lambda_S * sir_t0_depois[0][0])
        sir_t0_depois[1][0] = floor(self.lambda_I * sir_t0_depois[1][0])
        sir_t0_depois[2][0] = floor(self.lambda_R * sir_t0_depois[2][0])


        print(sir_t0_depois)


        #plt.style.use('_mpl-gallery-nogrid')
        #os.mkdir(fr"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ\Resultados\figs\{tipo}\inicio {inicio}")
        y_grafico = []
        
        y_grafico.append([0,0,0])
        #print(inicio, "imagem", t)
        #coluna = 0
        #linha = 0

        fig = plt.figure(1)
        fig.set_size_inches([15, 7.5])
        #ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(121)
        ax2.set_title(f"Vértice {vertice} no dia {dia}", fontdict={"size":18}, pad=60)
        ax3 = fig.add_subplot(122)

        #ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        # #ax1.set_title("Flamengo em t0 antes esplhamento")
        # colors = ["blue", "#0C46E8", "#33691E", "#39AB33", "#BF1900", "red"]
        # labels = [r"$\mathcal{\dot{S}}$", r"$\mathcal{\ddot{S}}$", r"$\mathcal{\dot{I}}$", r"$\mathcal{\ddot{I}}$", r"$\mathcal{\dot{R}}$", r"$\mathcal{\ddot{R}}$"]
        # x1 = [sir_t0_antes[0][0], sir_t0_antes[0][1], sir_t0_antes[1][0], sir_t0_antes[1][1], sir_t0_antes[2][0], sir_t0_antes[2][1]]
        # total1 = sum(x1)
        # ax1.pie(x1, colors=colors, radius=6, center=(4, 4), textprops={'fontsize': 18},
        #         autopct=lambda p: '{:.0f}'.format(p * total1 / 100), wedgeprops={"linewidth": 0, "edgecolor": "white"}, frame=True, labels=labels)
        

        Sddd = Iddd = Rddd = 0

        bottom = 1
        width = 0.2

        for j, (bairro, dici) in enumerate(reversed(sir_t0_depois[3][0].items())):
            Sddd += dici["S"]
            Iddd += dici["I"]
            Rddd += dici["R"]

            bottom -= dici["S"]
            bc = ax3.bar(0, dici["S"], width, bottom=bottom, label=bairro, color="#0D8CFF",
                    alpha=0.1 + (1/len(sir_t0_depois[3][0].items())) * j)
            ax3.bar_label(bc, labels=[f"{dici['S']}"], label_type='center')

        ax3.set_title('Bairros vizinhos', fontdict={"size":18})
        ax3.legend(fontsize=15)
        ax3.set_xlim(- 2.5 * width, 2.5 * width)



        #colors_depois = ["blue", "#0202a6", "#3b96ff", "#1aa103", "#0e6100", "#55eb3b", "red", "#8c0000", "#f73e3e"]
        colors_depois = ["blue", "#0C46E8", "#0D8CFF", "#33691E", "#39AB33", "#6AC230", "#BF1900", "red", "#f73e3e"]
        labels_depois = [r"$\dot{\mathcal{S}}$", r"$\ddot{\mathcal{S}}$", r"$\dddot{\mathcal{S}}$", r"$\dot{\mathcal{I}}$", r"$\ddot{\mathcal{I}}$", r"$\dddot{\mathcal{I}}$", r"$\dot{\mathcal{R}}$", r"$\ddot{\mathcal{R}}$", r"$\dddot{\mathcal{R}}$"]
        x2 = [sir_t0_depois[0][0], sir_t0_depois[0][1], Sddd, sir_t0_depois[1][0], sir_t0_depois[1][1], Iddd, sir_t0_depois[2][0], sir_t0_depois[2][1], Rddd]
        total2 = sum(x2)

        angle = -360 * ((sir_t0_depois[0][0] + sir_t0_depois[0][1] + Sddd/2) / total2)
        explode = [0, 0, 1.5, 0, 0, 0, 0, 0, 0]

        wedges, *_ = ax2.pie(x2, colors=colors_depois, radius=6, center=(4, 4), textprops={'fontsize': 18}, startangle=angle, explode=explode,
                autopct=lambda p: '{:.0f}'.format(p * total2 / 100), wedgeprops={"linewidth": 0, "edgecolor": "white"}, frame=True, labels=labels_depois)

        theta1, theta2 = wedges[2].theta1, wedges[2].theta2
        center, r = wedges[2].center, wedges[2].r
        bar_height = Sddd

        # draw top connecting line
        #(x, y - (2*np.pi*r)*(Sddd/2.5 / total2)),
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(- width / 2, -bar_height), xyB=(x, y),
                            coordsA=ax3.transData, coordsB=ax2.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(3)
        ax2.add_artist(con)

        # draw bottom connecting line
        #(x, y + (2*np.pi*r)*(Sddd/2.5 / total2))
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(- width / 2, 0), xyB=(x, y), coordsA=ax3.transData,
                            coordsB=ax2.transData)
        con.set_color([0, 0, 0])
        ax3.add_artist(con)
        con.set_linewidth(3)

        plt.savefig(path, format="png", dpi=300)
        plt.show()
        plt.close()

    def boxplot_zs_minimal_retirada_arestas(self, path):
        bairros_fora_ciclo = ("Cosme Velho", "Urca", "Lagoa", "Leme")

        ciclo1 = [('Flamengo', "Botafogo"), ('Flamengo', "Glória"), ("Glória", "Catete"), ("Catete", "Laranjeiras"), ("Botafogo", "Laranjeiras")]
        ciclo2 = [('Botafogo', 'Humaitá'), ('Botafogo', 'Copacabana'), ('Humaitá', 'Jardim Botânico'), ('Copacabana', 'Ipanema'), ('Jardim Botânico', 'Gávea'), ('Ipanema', 'Leblon'), ('Leblon', 'Vidigal'), ('Vidigal', 'São Conrado'), ('Gávea', 'Rocinha'), ('Rocinha', 'São Conrado')]

        inicios = []

        g = self.grafo.copy()
        arquivo_log = open(f"{path}/dados_boxplot.txt", "w", encoding="utf-8")
        for inicio in self.grafo.nodes():
            pico_infectados = []
            arestas_removidas = []
            inicios.append(inicio)
            
            print("Inicio:", inicio)
            self.vertice_de_inicio = inicio

            for arestaA1, arestaB1 in ciclo1:
                print(f"{arestaA1}-{arestaB1}")
                self.grafo.remove_edge(arestaA1, arestaB1)

                for arestaA2, arestaB2 in ciclo2:
                    print(f"    {arestaA2}-{arestaB2}", end=" ")
                    self.grafo.remove_edge(arestaA2, arestaB2)

                    for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                        vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)

                    self.avançar_tempo_movimentacao_dinamica(200, False)
                
                    pico_infectados.append(self.pico_infectados)
                    print(self.pico_infectados)
                    arestas_removidas.append(f"{arestaA1}-{arestaB1} {arestaA2}-{arestaB2}")


                    self.grafo.add_edge(arestaA2, arestaB2)
                    self.resetar_grafo()

                self.grafo.add_edge(arestaA1, arestaB1)

            arquivo_log.write(f"{inicio}: [{pico_infectados}, {arestas_removidas}]\n")
            df = pd.DataFrame(pico_infectados)
            sns.boxplot(data=df, width=0.5, fliersize=10, color="red")
            sns.swarmplot(data=df)
            plt.savefig(f"{path}/{inicio}.png", format="png", dpi=300)
            plt.close()

    def gerar_grafo_florianopolis(self):
        g = nx.DiGraph()
        self.peso_medio = 0

        for linha in self.arquivo:
            adj, propriedades = linha.strip().split("/")
            nome, *adj = adj.split(", ")
            numero, populaçao_s, populaçao_i, *beta = propriedades.split(", ")
            
            populaçao_s, populaçao_i = int(populaçao_s), int(populaçao_i)

            adj = [(nome, v, float(beta[i])) for i, v in enumerate(adj)]

            #if nome == self.vertice_de_inicio:
            I = populaçao_i
            #else:
                #I = 0
            #I = floor(self.frac_infect_inicial * populaçao)    # distribuir igualmente
            S = populaçao_s

            Sponto = floor(self.alpha * S)      # pessoas que respeitam o distanciamento social (ficam no vertice)
            Iponto = floor(self.alpha * I)

            S2pontos = S - Sponto               # pessoas que nao respeitam (podem sair do vertice)
            I2pontos = I - Iponto
            self.peso_medio += sum([float(beta[i]) for i, v in enumerate(adj)])

            g.add_node(nome, 
                id=int(numero), 
                #populaçao=populaçao_s + populaçao_i,
                populaçao_s = populaçao_s,
                populaçao_i = populaçao_i,
                SIR_t0={"S": S, "I": I, "R": 0},
                SIRd={"S": Sponto, "I": Iponto, "R": 0},
                SIRdd={"S": S2pontos, "I": I2pontos, "R": 0},
                SIRddd={},       # SIR ESTRANGEIRO (SIRdd de outros) #
                SIRdddantes={},
                #beta=float(beta),
                isolado=False)
            
            g.add_weighted_edges_from(adj, "beta")
            
        self.peso_medio = self.peso_medio / len(g.nodes())
        print(len(g.edges()))
        return g

    def boxplot_heuristica_floripa(self, path_picos):
        picos = open(path_picos, "r", encoding="utf-8")

        picos_por_arvore = dict()

        picos.readline();picos.readline();picos.readline();picos.readline()

        for linha in picos:
            if linha.strip():
                if linha[0] == " ":
                    if linha[1] == " ":             # remoção de arestas
                        linha = (linha.strip()).split("-")

                        verticeA = linha.pop(0)
                        linha = linha[0].split(" ")
                        verticeB = linha[0:2]

                        pico, dia_pico, dia_fim = linha[2::]

                        picos_por_arvore[arvore_atual].append(int(pico))
                    else:                       # adiçao de aresta
                        linha = (linha.strip()).split("-")

                        verticeA = linha.pop(0)
                        linha = linha[0].split(" ")
                        verticeB = linha[0:2]

                        pico, dia_pico, dia_fim = linha[2::]

                        picos_por_arvore[arvore_atual].append(int(pico))
                else:
                    arvore_atual = linha.strip().split(" ", maxsplit=1)[1].split(":")[0]
                    picos_por_arvore[arvore_atual] = []

        #print(picos_por_arvore)
        #fig, ax = plt.subplots(3, 6)
        #df = pd.DataFrame(picos_por_arvore)

        #print(picos_por_arvore)
        #picos_por_arvore = {"Grupos": [key for key in picos_por_arvore.keys()], "Picos": [picos_por_arvore[key] for key in picos_por_arvore.keys()]}
        #print(picos_por_arvore)
        picos_por_arvore = {key: picos_por_arvore[key] for key in sorted(picos_por_arvore.keys(), key=lambda x: int(x.split(" ")[1]))}
        df = pd.DataFrame.from_dict({'Início da Árvore': list(picos_por_arvore.keys()), 'Pico': list(picos_por_arvore.values())})#, columns=("Grupo", "Pico"), orient='index')
        df = df.explode(column='Pico').reset_index(drop=True)
        plt.figure(figsize=(14,7))

        print(df)
        #g = sns.FacetGrid(df, col="Grupos") #col_wrap=4,  height=2, ylim=(0, 10))
        sns.boxplot(data=df, x="Início da Árvore", y="Pico", width=0.7, color="red").set(title='Pico de Infectados da Heurística')
        sns.swarmplot(data=df, x="Início da Árvore", y="Pico", size=1)

        #g.map_dataframe(sns.boxplot, width=0.5, fliersize=10, color="red")
        #g.map_dataframe(sns.swarmplot, size=2)
            
            #plt.show()
        plt.savefig(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\heuristica/Boxplotteste.png", format="png", dpi=300, bbox_inches="tight")
            
        plt.close()

#? Escrever resultados etc
#? Salvar arquivos relevantes drive e separado

#os.chdir(r"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ")
os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

# "./txts/normal (real)/adjacencias.txt"
# "./txts/zona sul/arquivo_final.txt"
# "./txts/zona sul/adjacencias_zona_sul.txt"
# "./txts/otimizado/adjacencias.txt"
# "./txts/outros/zona sul/arquivo_final_otimizado_circulo.txt"
# "./txts/zona sul modificada menor/adjacencias_zona_sul_sem_botafogo.txt"
arquivo_adjacencias = "./Txts\outros\zona sul modificada ciclos/adjacencias_zona_sul.txt"
arquivo_final =  "./txts/normal (real)/arquivo_final.txt" #"./Txts/outros\zona sul/arquivo_final.txt"
arquivo_final_flo = "../Instancia Florianopolis/arquivo_final.txt"
arquivo_ID_nomes = "./txts/nova relaçao ID - bairros.txt"
tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"


resultados_arvore_profundidade = "./Resultados/resultados_arvore_profundidade.txt"
SIRxTdeVerticesTXT_profundidade = "./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt"
resultados_arvore_largura = "./Resultados/resultados_arvore_largura.txt"
SIRxTdeVerticesTXT_largura = "./Resultados/SIR_vertice_por_tempo_LARGURA.txt"


#txt = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_final, tabela_populaçao)
#txt.gerar_arquivo_destino()

#? RODAR HEURISTICA NA ZONA SUL
# MUDAR GERAÇÃO DOS VALORES INICIAIS
m = Modelo(arquivo_final_flo, flo=True)
#print(m.grafo.edges(data=True))
#m.vertice_de_inicio = "Flamengo"
#m.resetar_grafo()
#m.avançar_tempo_movimentacao_dinamica_otimizado()
#m.printar_grafo()
#m.boxplot_heuristica_floripa(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\heuristica\arvores\Infectados em todos\picos_por_arvores_e_arestas_profundidade.txt")
path_log = "./Resultados/heuristica/SIR_vertice_por_tempo_heuristica"
path_picos = "./Resultados/heuristica/picos_por_arvores_e_arestas"
m.heuristica_arvores_vizinhas("profundidade", path_log, path_picos)

#?visitados = set()
# anterior = {}
# niveis = {}
# m.grafo_original = m.grafo.copy()
# m.busca_em_profundidade("Grupo 1", anterior, visitados, niveis, 0)
# ciclo = []
# v, u = "Grupo 10", "Grupo 6"
# w, z = (v, u) if (max(niveis[v], niveis[u]), min(niveis[v], niveis[u])) == (niveis[v], niveis[u]) else (u, v)
# nivel = niveis[w]
# print(niveis, "\n\n", anterior)
# m.grafo.remove_edges_from(list(m.grafo.edges()))
# for vertice, ant in anterior.items():
#     m.grafo.add_weighted_edges_from([(ant, vertice, m.grafo_original.get_edge_data(ant, vertice)["beta"]), (vertice, ant, m.grafo_original.get_edge_data(vertice, ant)["beta"])], weight="beta")
# m.printar_grafo()
# while nivel > niveis[z]:
#     ciclo.append(w)
#     w = anterior[w]
#     nivel -= 1
# print("W e Z mesmo nivel:", w, z)
# ciclo2 = []
# while w != z:
#     ciclo.append(w)
#     w = anterior[w]
#     ciclo2.append(z)
#     z = anterior[z]
# ciclo.append(w)     # raiz
# if ciclo2:
#     ciclo.extend(list(reversed(ciclo2)))  # append outro lado do caminho na arvore
# print("W e Z dps de achar ciclo:", w, z)
#?print("Criando", v, "-->", u, "/ ciclo:", ciclo, "\n")


# print(len(m.grafo.nodes()))
# for bairro in m.grafo.nodes():
#     m.avançar_tempo_movimentacao_dinamica(200, False)
#     print(f"Inicio {bairro}: {m.pico_infectados}")

#m.printar_tabela_arvores()
#m.montar_tabela_zona_sul_ciclos()

#print(m.grafo.edges())

#m.printar_grafo("zonasul")
#m.avançar_tempo_movimentacao_dinamica(30)
#m.printar_grafico_SIR_t0_VerticePizza(r"C:\Users\rasen\Desktop\pizza1.png", dia=30, v="Flamengo")
#m.boxplot_zs_minimal_retirada_arestas(r"C:\Users\rasen\Desktop")

# print(m.tempo_pico)
#print(m.pico_infectados)
#print(m.grafo.edges())
#m.printar_grafico_SIRxT(path=r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Grafico SIRxT zs original.png") 


## ! teste isolamento
# pico_antigo = 73
# duraçao_ciclo = pico_antigo // 10
# dia_aumento_significativo = 18

# vezes = (pico_antigo - dia_aumento_significativo) // duraçao_ciclo
# dias_sobrando = (pico_antigo - dia_aumento_significativo) % duraçao_ciclo

# m.avançar_tempo_movimentacao_dinamica(dia_aumento_significativo)

# for dia in range(vezes):
#     m.isolar_vertices_mais_populosos()
#     m.avançar_tempo_movimentacao_dinamica(duraçao_ciclo)

# m.avançar_tempo_movimentacao_dinamica(dias_sobrando)
# print("tejnpo :",m.t)
# if m.isolados == True:
#     m.isolar_vertices_mais_populosos()

# m.avançar_tempo_movimentacao_dinamica(200 - m.t + 1)

# print(m.tempo_pico)
# print(m.pico_infectados)
## ! fim teste isolamento


# m.arvores_vizinhas("largura")

# m.printar_grafico_ID_MAXINFECT_arvore(tipo_arvore="largura")
# m.printar_grafico_ID_MAXINFECT_arvore(tipo_arvore="profundidade")
# m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()

#m.printar_tabela_arvores()

#m.printar_grafico_SIRxTdeVerticesPizza()
# m.avançar_tempo_movimentacao_dinamica_nao_discreto(0.5, 200)

# print(m.pico_infectados)
# m.printar_grafico_SIRxT()
#m.arvores_vizinhas("profundidade")

#m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()
#m.printar_grafico_ID_MAXINFECT_arvore("profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_profundidade_antes_depois()

