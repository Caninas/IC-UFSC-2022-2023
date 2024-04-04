import os
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
from numpy import array
from math import floor, ceil, sqrt
from random import random
import ast
import time
import pandas as pd
import openpyxl
import seaborn as sns

from scipy.interpolate import interp1d

# from Txt import Txt

# usar latex avançado na geraçao de grafico (altera levemente as fontes também)
# matplotlib.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{stackengine} \usepackage{amsmath} \usepackage{calc} \usepackage[utf8]{inputenc} \stackMath \newcommand\tsup[2][2]{ \def\useanchorwidth{T} \ifnum#1>1 \stackon[-.5pt]{\tsup[\numexpr#1-1\relax]{#2}}{\scriptscriptstyle\sim} \else \stackon[.5pt]{#2}{\scriptscriptstyle\sim} \fi}")

class Modelo:
    # arquivo no formato | int(id), adjs(sep=", ")/prop1(name), prop2(populaçao), prop3(beta)... |
    def __init__(self, arq_final, flo=False):
        self.arquivo = open(arq_final, "r", encoding="utf-8")

        self.t = 1

        # variaveis para a construçao de graficos posteriormente
        self.tempos = []
        self.SIRs = []
        self.pico_infectados = 0
        self.vertice_de_inicio = ""
        self.SIRxTdeVertices = dict()

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

        # grafo de floripa precisa ser gerado e rodade de forma diferente
        self.floripa = flo
        self.grafo = (self.gerar_grafo_florianopolis if self.floripa else self.gerar_grafo)()    # grafo utilizado atualmente
        self.grafo_original = 0             # copia do grafo original, usado na criaçao de arvores

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
                beta=float(beta),
                isolado=False)
            g.add_edges_from(adj)

        return g
    
    def gerar_grafo_florianopolis(self):
        g = nx.DiGraph()
        self.peso_medio = 0

        for linha in self.arquivo:
            adj, propriedades = linha.strip().split("/")
            nome, *adj = adj.split(", ")
            numero, populaçao_s, populaçao_i, *beta = propriedades.split(", ")
            
            populaçao_s, populaçao_i = int(populaçao_s), int(populaçao_i)

            adj = [(nome, v, float(beta[i])) for i, v in enumerate(adj)]


            S = populaçao_s
            I = populaçao_i

            Sponto = floor(self.alpha * S)      # pessoas que respeitam o distanciamento social (ficam no vertice)
            Iponto = floor(self.alpha * I)

            S2pontos = S - Sponto               # pessoas que nao respeitam (podem sair do vertice)
            I2pontos = I - Iponto
            self.peso_medio += sum([float(beta[i]) for i, v in enumerate(adj)])

            g.add_node(nome, 
                id=int(numero), 
                populaçao_s = populaçao_s,
                populaçao_i = populaçao_i,
                SIR_t0={"S": S, "I": I, "R": 0},
                SIRd={"S": Sponto, "I": Iponto, "R": 0},
                SIRdd={"S": S2pontos, "I": I2pontos, "R": 0},
                SIRddd={},       # SIR ESTRANGEIRO (SIRdd de outros) #
                SIRdddantes={},
                isolado=False)
            
            g.add_weighted_edges_from(adj, "beta")
            
        self.peso_medio = self.peso_medio / len(g.nodes())
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
            self.SIRxTdeVertices = {}
            self.tempos = []
            self.t = 1
            self.pico_infectados = 0

    def estimar_tempo_restante(self, percorrido, total, texto_restante="arvores restantes"):
        if percorrido:
            estimativa = ((time.perf_counter() - self.estimativa_tempo_inicial) / percorrido) * (total - percorrido)
        else:
            self.estimativa_tempo_inicial = time.perf_counter()
            estimativa = 0

        print(f"Estimativa {estimativa/60:.2f} minutos | {total - percorrido} {texto_restante}")


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
        # pos = nx.circular_layout(self.grafo.subgraph(("Ipanema"...)))
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

        nx.draw(g, pos, with_labels=True, font_weight='bold', font_size=11, node_size=300) #fonte 6 nodesize 200
        print(pos)
        plt.savefig(fr"Exemplo grafo zs novo.png", format="png", dpi=300)

        plt.show()  

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
        fig.set_size_inches([9, 6])

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=15, labelsize=15)
        
        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
       
        # for i in range(len(self.tempos), 201):    # igualar a 200 dias
        #     self.tempos.append(i)
        #     self.SIRs.append(self.SIRs[-1])

        if x:
            plt.xlim(left=1, right=len(x))
            plt.plot(x, y)
        else:
            plt.xlim(left=self.tempos[0], right=self.tempos[-1])
            plt.plot(self.tempos, self.SIRs)


        ax.legend(["S", "I", "R"], loc='upper right')
        ax.set_xlabel('Tempo', fontsize=14)
        ax.set_ylabel('Pessoas', fontsize=14)


        [tick.set_fontsize(13) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]

        if path and not x:
            plt.savefig(path, format="png", dpi=300, bbox_inches='tight')
        elif x:
            #inicio = path.split("/")[-1]
            #plt.title(f'Início: {inicio.split(".png")[0]}')
            
            plt.title(path)
            plt.savefig("Grafico SIRxT.png", format="png", dpi=300, bbox_inches='tight')
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

    def busca_em_largura(self, inicio, niveis=dict()):
        fila = []
        visitados = {inicio}
        anterior = {}
        fila.append(inicio)

        niveis[inicio] = 0

        while len(fila):
            v = fila.pop(0)

            for vizinho in self.grafo_original.edges(v):
                vizinho = vizinho[1]
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
                    anterior[vizinho] = v
                    niveis[vizinho] = niveis[v] + 1

        return anterior

    def busca_em_profundidade(self, v, anterior, visitados, niveis=dict(), nivel=0):
        visitados.add(v)
        niveis[v] = nivel

        for vizinho in self.grafo_original.edges(v):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                anterior[vizinho] = v
                self.busca_em_profundidade(vizinho, anterior, visitados, niveis, nivel+1)

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
                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao(tempo)
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
                print("\nInicio:", inicio, "| Iteração:", i+1)

                self.avançar_tempo_movimentacao(tempo)
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


    def printar_grafico_ID_MAXINFECT_arvore(self, tipo_arvore):
        resultados_grafo_original = open("./Resultados/picos_inicios_grafo_original.txt", "r", encoding="utf-8")

        if tipo_arvore == "largura":
            resultados_arvore = open("./Resultados/resultados_arvore_largura.txt", "r")
        else:
            resultados_arvore = open("./Resultados/resultados_arvore_profundidade.txt", "r")

        titulo = f'Picos de Infectados das Árvores de Busca em {tipo_arvore.title()} e Grafo Original'
        resultados_lista = [x for x in range(159)]

        for linha in resultados_arvore:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)-1] = [int(float(max_infect))]


        resultados_grafo_original.readline();resultados_grafo_original.readline();resultados_grafo_original.readline();
        
        for linha in resultados_grafo_original:
            linha = linha.strip()
            linha = linha.split(" ")
            if linha == "":
                break

            nome_bairro = " ".join(linha[0:len(linha)-3])
            pico, dia_pico, dia_fim = linha[len(linha)-3:len(linha)]
            resultados_lista[self.grafo.nodes[nome_bairro]["id"] - 1].append(int(pico))
        
        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([16, 9])

        [tick.set_fontsize(13) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]


        plt.gca().set_prop_cycle('color', ['green', '0d66a3', "red"])

        id_bairros = {x[1]["id"]:x[0] for x in self.grafo.nodes(data=True)}

        #bairros_selecionados = set(("Saúde", "Cidade Nova", "Barra de Guaratiba", "Jacaré", "Vaz Lobo", "Vista Alegre", "Cocotá", "Deodoro", "Padre Miguel"))

        x = [id_bairros[id+1] for id in range(len(id_bairros))]
        
        ticks = []
        for i, bairro in enumerate(x):
            # if id_bairros[i+1] in bairros_selecionados:
            #     print(i)
            if i % 3 == 0:
                ticks.append(i)

        plt.xticks(ticks, rotation=90, fontsize=10)
        plt.xlim(left=-2, right=160)

        cor_arvore = '0d66a3' if tipo_arvore == "profundidade" else 'green'
        plt.gca().set_prop_cycle('color', [cor_arvore, "red"])

        plt.plot(x, resultados_lista, "o")    # valores arvores
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend([tipo_arvore.title(), "Grafo Original"], loc='center right', bbox_to_anchor=(1.130, 0.5))
        ax.legend([tipo_arvore.title(), "Grafo Original"], loc='upper right')

        plt.title(titulo)
        ax.set_xlabel('Nome do Bairro de Início')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(bottom=0.2)
        #plt.show()
        plt.savefig(fr"C:\Users\rasen\Desktop\Pico Infectados Arvores {tipo_arvore.title()} novo.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

    def printar_grafico_ID_MAXINFECT_arvores_largura_profundidade(self):
        #./Resultados/picos_inicios_grafo_original.txt
        resultados_grafo_original = open(r"C:\Users\rasen\Documents\Programacao\IC Iniciação Científica\Instancia RJ\Resultados\picos_inicios_grafo_original.txt", "r", encoding="utf-8")
        resultadosL = open("./Resultados/resultados_arvore_largura.txt", "r")
        resultadosP = open("./Resultados/resultados_arvore_profundidade.txt", "r")
        
        titulo = f'Picos de Infectados do Grafo Original e das Árvores de Busca em Largura e Profundidade'

        resultados_lista = [x for x in range(159)]

        for linha in resultadosL:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)-1] = [int(float(max_infect))]


        for linha in resultadosP:
            linha = linha.strip()

            if linha == "":
                break

            id, dia_pico, max_infect = linha.split(", ")

            resultados_lista[int(id)-1].append(int(float(max_infect)))

        resultados_grafo_original.readline();resultados_grafo_original.readline();resultados_grafo_original.readline();
        
        for linha in resultados_grafo_original:
            linha = linha.strip()
            linha = linha.split(" ")
            if linha == "":
                break

            nome_bairro = " ".join(linha[0:len(linha)-3])
            pico, dia_pico, dia_fim = linha[len(linha)-3:len(linha)]
            
            resultados_lista[self.grafo.nodes[nome_bairro]["id"] - 1].append(int(pico))

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([16, 9])

        [tick.set_fontsize(13) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]


        plt.gca().set_prop_cycle('color', ['green', '0d66a3', "red"])

        id_bairros = {x[1]["id"]:x[0] for x in self.grafo.nodes(data=True)}

        #bairros_selecionados = set(("Saúde", "Cidade Nova", "Barra de Guaratiba", "Jacaré", "Vaz Lobo", "Vista Alegre", "Cocotá", "Deodoro", "Padre Miguel"))

        x = [id_bairros[id+1] for id in range(len(id_bairros))]
        
        ticks = []
        for i, bairro in enumerate(x):
            # if id_bairros[i+1] in bairros_selecionados:
            #     print(i)
            if i % 3 == 0:
                ticks.append(i)
        # print(ticks)

        # ticks = [0, 3, 7, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 49,
        #         51, 54, 57, 60, 63, 66, 69, 73, 75, 78, 82, 84, 87, 90, 94, 96,
        #         99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 
        #         135, 138, 141, 144, 147, 150, 153, 156]

        # ticks = [0, 4, 7, 12, 16, 20, 24, 28, 32, 36, 40, 44, 49, 52, 56, 60, 
        #         64, 68, 73, 76, 80, 82, 84, 88, 92, 94, 96, 100, 104, 108, 112, 116,
        #         120, 124, 128, 132, 136, 138, 140, 144, 148, 150, 152, 156]
        
        #fig.autofmt_xdate()
        plt.xticks(ticks, rotation=90, fontsize=10)
        plt.xlim(left=-2, right=160)
        
        plt.plot(x, resultados_lista, "o")    # valores arvores

        # for bairro in bairros_selecionados:
        #     for tick in range(len(ax.get_xticklabels())):
        #         if ax.get_xticklabels()[tick].get_text() == bairro:
        #             ax.get_xticklabels()[tick].set_weight(600)

        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend(["Largura", "Profundidade", "Grafo Original"], loc='upper right')
        #plt.grid()
        plt.title(titulo)
        ax.set_xlabel('Nome do Bairro de Início')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(bottom=0.2)
        #plt.show()
        plt.savefig(fr"C:\Users\rasen\Desktop\Pico Infectados Arvores Largura e Profundidade FINAL.png", format="png", dpi=300, bbox_inches="tight")
    
    def printar_grafico_ID_MAXINFECT_arvores_original_florianopolis(self):
        resultadosL = open(r"C:\Users\rasen\Desktop\Resultados\Resultados Arvores Florianopolis/resultados_arvore_largura.txt", "r")
        resultadosP = open(r"C:\Users\rasen\Desktop\Resultados\Resultados Arvores Florianopolis/resultados_arvore_profundidade.txt", "r")
        
        titulo = f'Picos de Infectados do Grafo Original e das Árvores de Busca em Largura e Profundidade'

        resultados_lista = [x for x in range(16)]

        for linha in resultadosL:
            linha = linha.strip()

            if linha == "":
                break

            inicio, pico, dia_pico, dia_fim = linha.split(", ")

            resultados_lista[int(inicio.split(" ")[1])-1] = [int(pico)]


        for linha in resultadosP:
            linha = linha.strip()

            if linha == "":
                break

            inicio, pico, dia_pico, dia_fim = linha.split(", ")

            resultados_lista[int(inicio.split(" ")[1])-1].append(int(pico))

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([16, 9])

        [tick.set_fontsize(13) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]
        
        plt.xlim(left=0, right=17)
        plt.xticks([x for x in range(1, 17)])

        plt.gca().set_prop_cycle('color', ['green', '0d66a3', "red"])

        plt.plot([x for x in range(1, 17)], resultados_lista, "o", markersize=10)    # valores arvores
        left, right = plt.xlim()
        plt.plot([left, right], [83271, 83271], linewidth=2, label="Pico Original")

        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.legend(["Largura", "Profundidade", "Grafo Original"], loc='upper right')

        plt.title(titulo)
        ax.set_xlabel('Grupo de Início da Árvore')
        ax.set_ylabel('Pico de Infectados')

        #plt.show()
        plt.savefig(fr"C:\Users\rasen\Desktop\Pico Infectados Floripa Arvores Largura e Profundidade.png", format="png", dpi=300, bbox_inches='tight')
    
    def printar_grafico_ID_MAXINFECT_arvores_profundidade_antes_depois(self):
        # 200 dias rodados vs 400 dias
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
        plt.close()

    def avançar_tempo_movimentacao(self, t, printar=True):
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

    # MODELO TERMINA QUANDO NAO HA INFECTADOS
    def avançar_tempo_movimentacao_otimizado(self, printar=0):
        # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))
       # soma_SIR = [0,1,0]
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
                    for vizinho in self.grafo.edges(nome, data=True):
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
                print("tempo=", self.t, end=' ') if printar==2 else print("tempo=", self.t, end=' | ')

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
                print(soma_SIR) if printar==2 else print(f"SIR: {soma_SIR}", end='\r')

            if not any(i[1] > soma_SIR[1] for i in self.SIRs):
                self.tempo_pico = self.t
                self.pico_infectados = soma_SIR[1]

            if soma_SIR[1] == 0:
                print("")
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

    def avançar_tempo_movimentacao_deltaT(self, deltaT: float, t: int):
          # prob y** pi -> i = prob y* pi (nao respeitam e ficam é igual ao respeitam (realizada sobre lambda_S*))

        for tempoAtual in range(1, int(t/deltaT) + 1):
            print(deltaT * tempoAtual)
            # distribuiçao de pessoas
            for node in list(self.grafo.nodes(data=True)):
                nome, atributos = node

                if tempoAtual != 1:
                    self.SIRxTdeVertices[nome][tempoAtual] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]
                else:
                    self.SIRxTdeVertices[nome] = dict()
                    self.SIRxTdeVertices[nome][tempoAtual] = [
                        atributos["SIRd"]["S"] + atributos["SIRdd"]["S"],
                        atributos["SIRd"]["I"] + atributos["SIRdd"]["I"],
                        atributos["SIRd"]["R"] + atributos["SIRdd"]["R"]
                    ]
                
                if self.floripa:
                    for vizinho in self.grafo.edges(nome, data=True):
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


            self.tempos.append(deltaT * tempoAtual)
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
                
                plt.savefig(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Grafico pizza grafo dinamico flamengo\tempo {t}.png", format="png", bbox_inches="tight")
                plt.close(fig="all")

    def printar_grafico_SIRxTdeVerticesPizzaTXT(self, path, tipo):
        self.SIRxTdeVerticesTXT = open(path, "r", encoding="utf-8")
        for linha in self.SIRxTdeVerticesTXT:
            inicio, dicionario_dados = linha.split(", ", maxsplit=(1))
            self.SIRxTdeVertices = ast.literal_eval(dicionario_dados)
            
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
                anterior = self.busca_em_largura(inicio, niveis)
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


            grafo_complemento = self.grafo_original.copy()

            grafo_complemento.remove_edges_from(self.grafo.edges)       # complemento do arvore levando em conta o original

            #print(grafo_complemento.edges(data=True), len(grafo_complemento.edges()))

            for v, u in grafo_complemento.edges():      # adiçao da aresta que cria ciclo
                print("ADIÇÃO ARESTA: ", end="")
                self.estimar_tempo_restante(indice_arvore, len(self.grafo_original.nodes)*len(grafo_complemento.edges()))
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

                def achar_ciclo_a(anterior):
                    # w vertice de maior nivel, z menor
                    w, z = (v, u) if (max(niveis[v], niveis[u]), min(niveis[v], niveis[u])) == (niveis[v], niveis[u]) else (u, v)

                    nivel = niveis[w]

                    while nivel > niveis[z]:
                        ciclo.append(w)
                        w = anterior[w]
                        nivel -= 1
                    
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


                    print("Criando", v, "-->", u, "/ ciclo:", ciclo)

                    # salvar niveis dos vertices na arvore (pair<nivel, vertice>)
                    # busca em largura e profundidade linha 1170 - 1173
                    # igualar niveis dos vertices v, u atraves dos anteriores 
                    # depois ir voltando com os dois ate chegarem em um antecessor comum
                    # salvar caminho de v em um array normal e de u em um array de forma inversa,
                    # depois juntar colocando vertice igual no final do primeiro
                
                achar_ciclo_a(anterior)

                if self.floripa:
                    self.grafo.add_weighted_edges_from([(v, u, self.grafo_original.get_edge_data(v, u)["beta"]), (u, v, self.grafo_original.get_edge_data(u, v)["beta"])], weight="beta")
                else:
                    self.grafo.add_edge(v, u)
                    for vertice in [v, u]:          # atualizar betas com a nova aresta
                        self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)

                pico = [0, 0, 0]
                for iteraçao in range(3):
                    print("Iteração", iteraçao+1)
                    self.resetar_grafo()
                    self.avançar_tempo_movimentacao_otimizado(printar=True)
                    pico = [x+y for x, y in zip(pico, [self.pico_infectados, self.tempo_pico, self.t])]
                pico = [int(x/3) for x in pico]

                
                arquivo_picos.write(f" {v}-{u} {self.pico_infectados} {self.tempo_pico} {self.t}\n")
                arquivo_log.write(f"{self.SIRxTdeVertices}\n")

                for indice in range(0, len(ciclo) - 1):    # loop arestas do ciclo (menos v, u)
                    x = ciclo[indice]
                    y = ciclo[indice + 1]
                    #self.resetar_grafo()
                    #with self.terminal.location(y=self.terminal.height-4):
                    print(f"Removendo {x} -> {y}")

                    self.grafo.remove_edges_from([(x, y), (y, x)])

                    #! rodar modelo

                    pico = [0, 0, 0]

                    for iteraçao in range(3):
                        print("Iteração", iteraçao+1)
                        self.resetar_grafo()
                        self.avançar_tempo_movimentacao_otimizado(printar=True)
                        pico = [x+y for x, y in zip(pico, [self.pico_infectados, self.tempo_pico, self.t])]
                    pico = [int(x/3) for x in pico]

                    arquivo_picos.write(f"  {x}-{y} {self.pico_infectados} {self.tempo_pico} {self.t}\n")
                    arquivo_log.write(f"{self.SIRxTdeVertices}\n")
                    
                    if self.floripa:
                        self.grafo.add_weighted_edges_from([(x, y, self.grafo_original.get_edge_data(x, y)["beta"]), (y, x, self.grafo_original.get_edge_data(y, x)["beta"])], weight="beta")
                    else:
                        self.grafo.add_edge(x, y)
                        for vertice in [x, y]:          # atualizar betas com nova aresta
                            self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)

                self.grafo.remove_edges_from([(v, u), (u, v)])

            # salvar arvore original
            self.salvar_grafo_arvore(fr"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\heuristica\arvores\{inicio}.png")
        self.grafo = self.grafo_original


    def printar_tabela_arvores(self):
        # incompleto?
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

                    self.avançar_tempo_movimentacao(200)
                    
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

        #sir_t0_antes = self.SIRxTdeVertices[vertice][dia][0]
        
        sir_t0_depois = self.SIRxTdeVertices[vertice][dia][1]
        
        sir_t0_depois[0][0] = floor(self.lambda_S * sir_t0_depois[0][0])
        sir_t0_depois[1][0] = floor(self.lambda_I * sir_t0_depois[1][0])
        sir_t0_depois[2][0] = floor(self.lambda_R * sir_t0_depois[2][0])



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

        Sddd = Iddd = Rddd = 0

        bottom = 1
        width = 0.2

        for j, (bairro, dici) in enumerate((sir_t0_depois[3][0].items())):
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
        labels_depois = [r"$\mathcal{\dot{S}}$", r"$\mathcal{\ddot{S}}$", r"$\mathcal{\dddot{S}}$", r"$\mathcal{\dot{I}}$", r"$\mathcal{\ddot{I}}$", r"$\mathcal{\dddot{I}}$", r"$\dot{\mathcal{R}}$", r"$\mathcal{\ddot{R}}$", r"$\mathcal{\dddot{R}}$"]
        #labels_depois = [r"$\bar{\mathcal{S}}$", r"$\tsup[1]{\mathcal{S}}$", r"$\tsup{\mathcal{S}}$", r"$\bar{\mathcal{I}}$", r"$\tsup[1]{\mathcal{I}}$", r"$\tsup{\mathcal{I}}$", r"$\bar{\mathcal{R}}$", r"$\tsup[1]{\mathcal{R}}$", r"$\tsup{\mathcal{R}}$"]
        x2 = [sir_t0_depois[0][0], sir_t0_depois[0][1], Sddd, sir_t0_depois[1][0], sir_t0_depois[1][1], Iddd, sir_t0_depois[2][0], sir_t0_depois[2][1], Rddd]
        total2 = sum(x2)

        angle = -360 * ((sir_t0_depois[0][0] + sir_t0_depois[0][1] + Sddd/2) / total2)
        explode = [0, 0, 1.5, 0, 0, 0, 0, 0, 0]

        wedges, *_ = ax2.pie(x2, colors=colors_depois, radius=6, center=(4, 4), textprops={'fontsize': 20}, startangle=angle, explode=explode,
                autopct=lambda p: '{:.0f}'.format(p * total2 / 100), wedgeprops={"linewidth": 0, "edgecolor": "white"}, frame=True, labels=labels_depois)

        theta1, theta2 = wedges[2].theta1, wedges[2].theta2
        center, r = wedges[2].center, wedges[2].r
        bar_height = Sddd

        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(- width / 2, -bar_height), xyB=(x, y),
                            coordsA=ax3.transData, coordsB=ax2.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(3)
        ax2.add_artist(con)


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
        arquivo_log = open(f"{path}/dados_boxplot_zs_minimal.txt", "w", encoding="utf-8")
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

                    self.avançar_tempo_movimentacao(200, False)
                
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

    def boxplot_heuristica_floripa(self, path_picos, tipo_arvore):
        picos = open(path_picos, "r", encoding="utf-8")

        picos_por_arvore = dict()
        picos_por_arvore_ad = dict()

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
                        picos_por_arvore_ad[arvore_atual].append(int(pico))
                else:
                    arvore_atual = linha.strip().split(" ", maxsplit=1)[1].split(":")[0]
                    picos_por_arvore[arvore_atual] = []
                    picos_por_arvore_ad[arvore_atual] = []

        picos_por_arvore = {key: picos_por_arvore[key] for key in sorted(picos_por_arvore.keys(), key=lambda x: int(x.split(" ")[1]))}
        df = pd.DataFrame.from_dict({f'Início da Árvore de Busca em {tipo_arvore.title()}': list(picos_por_arvore.keys()), 'Pico': list(picos_por_arvore.values()), "Tipo": "Remoção"})
        df = df.explode(column='Pico').reset_index(drop=True)

        picos_por_arvore_ad = {key: picos_por_arvore_ad[key] for key in sorted(picos_por_arvore_ad.keys(), key=lambda x: int(x.split(" ")[1]))}
        df2 = pd.DataFrame.from_dict({f'Início da Árvore de Busca em {tipo_arvore.title()}': list(picos_por_arvore_ad.keys()), 'Pico': list(picos_por_arvore_ad.values()), "Tipo": "Adição"})
        df2 = df2.explode(column='Pico').reset_index(drop=True)

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)
        fig = plt.figure(1)

        fig.set_size_inches([16, 9])
        plt.yticks([x for x in range(70000, 89000, 2000)])

        sns.boxplot(data=df, x=f'Início da Árvore de Busca em {tipo_arvore.title()}', y="Pico", width=0.7, color="red").set(title=f'Pico de Infectados da Heurística em {tipo_arvore.title()}')
        sns.swarmplot(data=df, x=f'Início da Árvore de Busca em {tipo_arvore.title()}', y="Pico", size=4) #hue="Tipo"
        left, right = plt.xlim()
        plt.plot([left, right], [83271, 83271], color="green", linewidth=2, label="Pico Original")

        plt.legend(loc="upper left")
        plt.show()
        #plt.savefig(fr"C:\Users\rasen\Documents\Programacao\IC Iniciação Científica\Instancia RJ\Resultados\heuristica/Boxplot_heuristica_{tipo_arvore}_sem_adiçoes.png", format="png", dpi=300, bbox_inches="tight")

        plt.close()

    def rodar_modelo_inicio_vertices(self, path_resultados, path_sirt):
        arquivo_picos = open(path_resultados, "w", encoding="utf-8", buffering=1)
        SIRxTdeVerticesTXT = open(path_sirt, "w", encoding="utf-8", buffering=1)

        arquivo_picos.write("Picos por Inicio Grafo Original RJ\nInicio Pico Dia_do_Pico Fim_Espalhamento\n\n")
        
        for inicio in self.grafo.nodes():
            self.vertice_de_inicio = inicio
            pico = [0, 0, 0]
            print("\nInicio:", inicio)
            self.estimar_tempo_restante(list(self.grafo.nodes()).index(inicio), len(self.grafo.nodes()), "vértices de início restante")
            for iteraçao in range(3):
                print("Iteração:", iteraçao+1)

                self.resetar_grafo()
                self.avançar_tempo_movimentacao_otimizado(1)

                pico = [x+y for x, y in zip(pico, [self.pico_infectados, self.tempo_pico, self.t])]

            pico = [int(x/3) for x in pico]
            
            arquivo_picos.write(f"{inicio} {pico[0]} {pico[1]} {pico[2]}\n")
                        
            SIRxTdeVerticesTXT.write(f"{inicio}, {self.SIRxTdeVertices}\n")

    def gerar_SIRxT_inicios_rj(self, path_sirt):
        SIRxT = open(path_sirt, "r", encoding="utf-8")

        for linha in SIRxT:
            if linha:
                inicio, dicionario_dados = linha.split(", ", maxsplit=1)
                dicionario_dados = ast.literal_eval(dicionario_dados)
                #total_SIR = [0, 0, 0]
                x = [i for i in range(1, 201)]
                y = [[0, 0, 0] for i in range(200)]
                path = f"./Resultados/SIRxT Grafo Original/{inicio}.png"
                
                for vertice, valores in dicionario_dados.items():
                    for tempo, sir in valores.items():
                        if tempo < 201:     # 2 bairros acima de 200 com 205, 209
                            y[tempo-1] = [s+k for s, k in zip(sir, y[tempo-1])]
                    for i in range(200 - tempo):
                        y[tempo + i] = [s+k for s, k in zip(sir, y[tempo + i])]
                    
                self.printar_grafico_SIRxT(x, y, path)


    def gerar_grafos_arvores_florianopolis(self, tipo_arvore):
        resultados_arvore = open(f"./Resultados/Florianopolis arvores/resultados_arvore_{tipo_arvore}.txt", "w", encoding="utf-8")
        SIRxTdeVerticesTXT = open(f"./Resultados/Florianopolis arvores/SIRxTdeVerticesTXT_{tipo_arvore}.txt", "w", encoding="utf-8")

        self.grafo_original = self.grafo.copy()
        
        arvore = 0
        quant_arvores = len(self.grafo_original.nodes)
        
        for inicio in self.grafo_original.nodes:
            self.estimar_tempo_restante(arvore, quant_arvores)
            arvore += 1
            soma_pico = 0
            tempo_pico = 0
            fim_pico = 0

            anterior = dict()

            if tipo_arvore == "largura":
                anterior = self.busca_em_largura(inicio)
            else:
                visitados = set()
                self.busca_em_profundidade(inicio, anterior, visitados)

            self.grafo.remove_edges_from(list(self.grafo.edges()))

            for vertice, ant in anterior.items():
                self.grafo.add_weighted_edges_from([(ant, vertice, self.grafo_original.get_edge_data(ant, vertice)["beta"]), (vertice, ant, self.grafo_original.get_edge_data(vertice, ant)["beta"])], weight="beta")

            for i in range(3):  
                self.resetar_grafo()
                # testa semelhança entre arvores
                # qtd_arestas_iguais = 0            # qtd arestas iguais a saude
                # if inicio == "Saúde":
                #     self.arestas_primeira_arvore = (self.grafo.copy()).edges()
                # else:
                #     for aresta in self.grafo.edges():
                #         if aresta in self.arestas_primeira_arvore:
                #             qtd_arestas_iguais += 1
                #print(self.grafo.nodes[inicio]["id"], "| Arestas iguais à Saúde:", qtd_arestas_iguais, "/158")

                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao_otimizado(1)
                print("Pico:", self.pico_infectados)

                soma_pico += self.pico_infectados
                tempo_pico += self.tempo_pico
                fim_pico += self.t

            
            tempo_pico = int(tempo_pico / 3)
            media_pico = int(soma_pico / 3)
            fim_pico = int(fim_pico / 3)

            resultados_arvore.write(f"{inicio}, {media_pico}, {tempo_pico}, {fim_pico}\n")   
            SIRxTdeVerticesTXT.write(f"{inicio}, {self.SIRxTdeVertices}\n")
                
            self.salvar_grafo_arvore(f"./Resultados/Florianopolis arvores/arvores/{inicio}.png")

        resultados_arvore.close()
        SIRxTdeVerticesTXT.close()
        self.grafo = self.grafo_original

    def printar_grafico_convergencia(self):
        picos_delta = open("picos_discreto_final.txt", "r", encoding="utf-8")
        picos = []
        deltas = []
        
        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=13)

        for linha in picos_delta:
            delta, pico = linha.split(" ")

            delta = float(delta.split("=")[1])
            pico = float(pico)

            picos.append(pico)     
            deltas.append(delta)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig.set_size_inches([17, 7.5])

        plt.xticks([i for i in range(0, len(deltas))], deltas)

        deltas = [str(i) for i in range(len(deltas))]
        plt.plot(deltas, picos, "o", ms=9, color="blue")    # valores arvores

        for i in range(len(deltas)):
            if i:
                plt.annotate(f"{'+' if picos[i] > picos[i-1] else ''}{((picos[i]/picos[i-1]) * 100) - 100:.2f}%", (float(deltas[i]), picos[i]+300), fontsize=12)


        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        plt.title("Convergência do Pico de Infectados")
        
        ax.set_xlabel(r'$\Delta{t}$')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(fr"C:\Users\rasen\Desktop\Convergencia Pico Infectados2.png", format="png", dpi=300, bbox_inches='tight')

    def printar_grafico_SIRxT_TXT(self, path):
        SIRxT = open(path, "r", encoding="utf-8")
        x = []
        y = []
                            
        path = fr"C:\Users/rasen\Desktop\EAMat/picos_mat.png"
        print(y)
        fig = plt.figure(1)#.set_fig
        ax = fig.add_subplot(111)
        fig.set_size_inches([20, 5])

        #plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        #plt.xlim(left=1, right=len(x))
        

        ax.legend(["S", "I", "R"], loc='center right', bbox_to_anchor=(1.1, 0.5))
        plt.title("Infectados ao longo do tempo")
        ax.set_xlabel('Tempo (dias)')
        ax.set_ylabel('Infectados')


        for linha in SIRxT:
            if linha:
                deltaT, lista_dados = linha.split(" ", maxsplit=1)
                deltaT = float(deltaT.split("=")[1])
                lista_dados = ast.literal_eval(lista_dados)
                

                #xx = np.linspace(1, 200, int(200/deltaT))
                xx = [i for i in range(1, 121)]
                # y = [sir[1] for sir in lista_dados]
                # y = np.linspace(y[0], y[-1], 200)
                yy = [lista_dados[int(i/deltaT)][1] for i in range(120)]

                x.append(xx)
                y.append(yy)
                plt.plot(xx, yy, label=deltaT)
                # for vertice, valores in dicionario_dados.items():
                #     for tempo, sir in valores.items():
                #         if tempo < 201:     # 2 bairros acima de 200 com 205, 209
                #             y[tempo-1] = [s+k for s, k in zip(sir, y[tempo-1])]
                #     for i in range(200 - tempo):
                #         y[tempo + i] = [s+k for s, k in zip(sir, y[tempo + i])]


        #inicio = path.split("/")[-1]
        #plt.title(f'Início: {inicio.split(".png")[0]}')
                
        plt.legend(title="$\Delta{t}$")
        plt.savefig(path, format="png", dpi=1000, bbox_inches='tight')

        plt.close()

    def printar_grafico_delta_distanciamento(self, path):
        x, y = [], []
        delta = 0.0625
        #picos_dist = open("picos_distanciamento.txt", "w", encoding="utf-8", buffering=1)
        y = [1442315, 1203502, 924320, 581158, 168176, 2502]
        x = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=13)

        # for i in range(6):
        #     self.alpha = i/5        # fraçao de pessoas que respeitam o distanciamento, fator de distanciamento
        #     self.resetar_grafo()
            
        #     self.avançar_tempo_movimentacao_deltaT(delta, 200)       # mudar delta t 0.0625
        #     x.append(str(self.alpha))
        #     y.append(self.pico_infectados)
        #     picos_dist.writelines(f"{self.alpha} - ")

        fig = plt.figure(1)#.set_fig
        ax = fig.add_subplot(111)
        fig.set_size_inches([12, 10])

        #plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        #plt.xlim(left=1, right=len(x))

        plt.title("Pico de Infectados com diferentes fatores de distanciamento")
        ax.set_xlabel(r'Fator de distanciamento ($\alpha$)')
        ax.set_ylabel('Infectados')

        b = plt.bar(x, y, color=["#020024", "#070764", "#090979", "#055fb2", "#0045ff", "#00d4ff"])
        ax.bar_label(b, fmt='%.1i', fontsize=14)
        
        plt.savefig(f"{path}/Pico infectados distanciamento delta {delta}.png", format="png", dpi=300, bbox_inches='tight') 
        plt.show()

    def printar_grafico_SIRxT_TXT_sobreposto(self, path_arvore, path_original=None):

        SIRxT_arvore = open(path_arvore, "r", encoding="utf-8")

        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)

        fig = plt.figure(1)#.set_fig
        ax = fig.add_subplot(111)
        fig.set_size_inches([16, 9])

        [tick.set_fontsize(13) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]

        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue', ""])

        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        ax.set_xlabel('Tempo')
        ax.set_ylabel('Número de pessoas (normalizado)')

       
        plt.yticks([i/10 for i in range(0, 11)])


        # y modelo original
        y = [[6314583, 2502, 0], [6313868, 2852, 365], [6313068, 3208, 809], [6312074, 3693, 1318], [6310893, 4293, 1899], [6309526, 4973, 2586], [6307899, 5807, 3379], [6306044, 6743, 4298], [6303761, 7974, 5350], [6301066, 9434, 6585], [6297826, 11224, 8035], [6294077, 13250, 9758], [6289590, 15714, 11781], [6284256, 18662, 14167], [6278001, 22095, 16989], [6270603, 26163, 20319], [6261822, 31018, 24245], [6251608, 36596, 28881], [6239625, 43166, 34294], [6225921, 50447, 40717], [6209909, 58990, 48186], [6191657, 68513, 56915], [6170954, 79092, 67039], [6147781, 90607, 78697], [6122055, 102991, 92039], [6093583, 116311, 107191], [6062707, 130095, 124283], [6029500, 144200, 143385], [5994140, 158418, 164527], [5957025, 172284, 187776], [5918251, 185756, 213078], [5877893, 198857, 240335], [5836628, 210908, 269549], [5794514, 222124, 300447], [5751803, 232259, 333023], [5708381, 241629, 367075], [5664540, 250056, 402489], [5620280, 257659, 439146], [5575825, 264363, 476897], [5530534, 270890, 515661], [5484228, 277504, 555353], [5436857, 284230, 595998], [5388143, 291274, 637668], [5337583, 299192, 680310], [5284422, 308528, 724135], [5228995, 318735, 769355], [5170841, 330201, 816043], [5109538, 343123, 864424], [5044649, 357778, 914658], [4976018, 374032, 967035], [4903451, 391826, 1021808], [4826833, 411095, 1079157], [4746664, 431143, 1139278], [4661805, 452960, 1202320], [4573348, 475218, 1268519], [4481468, 497637, 1337980], [4386020, 520391, 1410674], [4287607, 542839, 1486639], [4185503, 565728, 1565854], [4080546, 588131, 1648408], [3972604, 610257, 1734224], [3861736, 632118, 1823231], [3747958, 653735, 1915392], [3631149, 675229, 2010707], [3511320, 696624, 2109141], [3389278, 717140, 2210667], [3264437, 737453, 2315195], [3137743, 756688, 2422654], [3009864, 774369, 2532852], [2881871, 789539, 2645675], [2755549, 800846, 2760690], [2631260, 808462, 2877363], [2511758, 810202, 2995125], [2397498, 806440, 
                    3113147], [2290151, 796342, 3230592], [2190832, 779638, 3346615], [2099552, 757344, 3460189], [2016788, 729783, 3570514], [1942298, 697922, 3676865], [1875955, 662531, 3778599], [1817194, 624700, 3875191], [1765582, 585231, 3966272], [1720183, 545257, 4051645], [1680297, 505600, 4131188], [1645546, 466550, 4204989], [1615171, 428830, 4273084], [1588587, 392786, 4335712], [1565560, 358417, 4393108], [1545238, 326335, 4445512], [1527606, 296194, 4493285], [1512207, 268220, 4536658], [1498630, 242467, 4575988], [1486854, 218642, 4611589], [1476547, 196852, 4643686], [1467676, 176780, 4672629], [1459787, 158626, 4698672], [1452907, 142120, 4722058], [1446874, 127158, 4743053], [1441638, 113575, 4761872], [1437146, 101226, 4778713], [1433126, 90184, 4793775], [1429661, 80230, 4807194], [1426716, 71168, 4819201], [1424099, 63118, 4829868], [1421878, 55839, 4839368], [1419995, 49288, 4847802], [1418315, 43480, 4855290], [1416896, 38287, 
                    4861902], [1415662, 33677, 4867746], [1414628, 29539, 4872918], [1413719, 25884, 4877482], [1412935, 22658, 4881492], [1412261, 19797, 4885027], [1411705, 17238, 4888142], [1411234, 14992, 4890859], [1410833, 13010, 4893242], [1410476, 
                    11283, 4895326], [1410175, 9767, 4897143], [1409936, 8419, 4898730], [1409762, 7216, 4900107], [1409600, 6205, 4901280], [1409457, 5334, 4902294], [1409349, 4574, 4903162], [1409271, 3902, 4903912], [1409197, 3320, 4904568], [1409134, 2826, 4905125], [1409091, 2380, 4905614], [1409060, 1981, 4906044], [1409035, 1663, 4906387], [1409014, 1389, 4906682], [1408996, 1162, 4906927], [1408986, 959, 4907140], [1408982, 775, 4907328], [1408976, 620, 4907489], [1408974, 502, 4907609], [1408974, 402, 4907709], [1408974, 320, 4907791], [1408974, 251, 4907860], [1408974, 191, 4907920], [1408974, 153, 4907958], [1408974, 118, 4907993], [1408974, 94, 4908017], [1408974, 76, 4908035], [1408974, 59, 4908052], [1408974, 44, 4908067], [1408974, 32, 4908079], [1408974, 25, 4908086], [1408974, 19, 4908092], [1408974, 14, 4908097], [1408974, 9, 4908102], [1408974, 6, 4908105], [1408974, 4, 4908107], [1408974, 2, 4908109], [1408974, 1, 4908110], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111], [1408974, 0, 4908111]]
        
        pop_total = 6317085
        for i in range(len(y)):
            y[i][0] = y[i][0] / pop_total
            y[i][1] = y[i][1] / pop_total
            y[i][2] = y[i][2] / pop_total

        yy = [[sqrt(sir[0]), sqrt(sir[1]), sqrt(sir[2])] for sir in y]
        plt.xlim(left=1, right=200)
        plot1 = plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 
                    164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200], yy)

        x = [i for i in range(1, 201)]
        y = [[0, 0, 0] for i in range(200)]

        for linha in SIRxT_arvore:
            if linha:
                inicio, dicionario_dados = linha.split(", ", maxsplit=(1))
                if inicio == "Flamengo":
                    dicionario_dados = ast.literal_eval(dicionario_dados)
                    #print(y)
                    for vertice, valores in dicionario_dados.items():
                        for tempo, sir in valores.items():
                            y[tempo-1] = [s+k for s, k in zip(sir, y[tempo-1])]

        for i in range(len(y)):
            y[i][0] = y[i][0] / pop_total
            y[i][1] = y[i][1] / pop_total
            y[i][2] = y[i][2] / pop_total
            
        y = [[sqrt(sir[0]), sqrt(sir[1]), sqrt(sir[2])] for sir in y]

        plot2 = plt.plot(x, y, marker="^", ms=3, linestyle='None')
        p5, = plt.plot([0], marker='None',
           linestyle='None', label='dummy-tophead')
        
        legenda = plt.legend([p5, *plot1, p5, *plot2], ["Original"] + ["S", "I", "R"] + ["Árvore"] + ["S", "I", "R"], loc='upper right', ncol=2, )
        plt.gca().add_artist(legenda)
        plt.grid()

        plt.show()
        plt.close()

    def salvar_plt_formataçao_padrao(self, fig, ax, path):
        matplotlib.rc('font', size=11)
        matplotlib.rc('axes', titlesize=14, labelsize=15)
        fig.set_size_inches([16, 9])

        [tick.set_fontsize(12) for tick in ax.get_xticklabels()]
        [tick.set_fontsize(13) for tick in ax.get_yticklabels()]
        plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

    def printar_grafico_SIRxT_formataçao_final(self, arvore):  
        SIRxT = open(f"..\Resultados\SIR_vertice_por_tempo_{arvore.upper()}.txt", "r", encoding="utf-8")
        x = []
        y = [[0,0,0] for x in range(0,200)]

        for linha in SIRxT:
            if linha:
                inicio, dict_dados = linha.split(", ", maxsplit=1)
                if inicio == "Flamengo":
                    dict_dados = ast.literal_eval(dict_dados)
                    for vertice, valores in dict_dados.items():
                        for tempo, sir in valores.items():
                            if tempo < 201:     # 2 bairros acima de 200 com 205, 209
                                y[tempo-1] = [s+k for s, k in zip(sir, y[tempo-1])]


        m.printar_grafico_SIRxT([x for x in range(1, 201)], y, f"Raiz: Flamengo ({arvore.title()})")
#? Escrever resultados etc
#? Salvar arquivos relevantes drive e separado

try:
    os.chdir(r"C:\Users\rasen\Documents\GitHub\IC-UFSC-2022-2023\Instancia RJ")
except FileNotFoundError:
    os.chdir(r"C:\Users\Pedro\Documents\Programacao\IC-UFSC-2022-2023\Instancia RJ")

# "./Instancias/normal (real)/adjacencias.txt"
# "./Instancias/zona sul/arquivo_final.txt"
# "./Instancias/zona sul/adjacencias_zona_sul.txt"
# "./Instancias/otimizado/adjacencias.txt"
# "./Instancias/outros/zona sul/arquivo_final_otimizado_circulo.txt"
# "./Instancias/zona sul modificada menor/adjacencias_zona_sul_sem_botafogo.txt"
arquivo_adjacencias = "./Instancias\outros\zona sul modificada ciclos/adjacencias_zona_sul.txt"
arquivo_final =  "./Instancias/normal (real)/arquivo_final.txt" #"./Instancias/outros\zona sul/arquivo_final.txt"
arquivo_final_flo = "../Instancia Florianopolis/Instancias/normal/arquivo_final.txt"
arquivo_ID_nomes = "./Instancias/nova relaçao ID - bairros.txt"
tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"


resultados_arvore_profundidade = "./Resultados/resultados_arvore_profundidade.txt"
SIRxTdeVerticesTXT_profundidade = "./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt"
resultados_arvore_largura = "./Resultados/resultados_arvore_largura.txt"
SIRxTdeVerticesTXT_largura = "./Resultados/SIR_vertice_por_tempo_LARGURA.txt"


#txt = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_final, tabela_populaçao)
#txt.gerar_arquivo_destino()


m = Modelo(arquivo_final)
m.vertice_de_inicio = "Flamengo"
m.resetar_grafo()

#m.printar_grafico_SIRxT_formataçao_final("largura")

#m.boxplot_heuristica_floripa(r"C:\Users\rasen\Desktop\Resultados\Heuristica Florianopolis\com teste nos ciclos (adiçao de arestas)\picos_por_arvores_e_arestas_profundidade.txt", "profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_original_florianopolis()
#m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()
#m.printar_grafico_ID_MAXINFECT_arvore("profundidade")


#m.printar_grafico_SIRxT_TXT_sobreposto(r"C:\Users\rasen\Desktop\Resultados\com betas\Graficos SIRxT arvores largura\SIR_vertice_por_tempo_LARGURA.txt")
#m.printar_grafico_SIR_t0_VerticePizza(r"C:\Users\rasen\Desktop\pizza_composiçao_notaçao_antiga.png", dia=30, v="Flamengo")

#m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()

#m.printar_grafico_delta_distanciamento(r"C:\Users\rasen\Desktop")

# m.avançar_tempo_movimentacao_otimizado(printar=1)
# print(m.pico_infectados)
# m.resetar_grafo()
#m.printar_grafico_SIRxT_TXT(r"C:\Users\rasen\Desktop\EAMat/sirs_discreto_final.txt")
#m.avançar_tempo_movimentacao_deltaT(1, 200)
#m.printar_grafico_convergencia()
# m.avançar_tempo_movimentacao(20)
# print(m.SIRxTdeVertices)
# m.printar_estados_vertices()



# picos_discreto = open("picos_discreto.txt", "w", buffering=1)
# sirs = open("sirs_discreto.txt", "a", buffering=1)

# m.avançar_tempo_movimentacao_deltaT(1, 200)
# picos_discreto.write(f"dt=1 {m.pico_infectados}\n")
# sirs.write(f"dt=1 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.5, 200)
# picos_discreto.write(f"dt=0.5 {m.pico_infectados}\n")
# sirs.write(f"dt=0.5 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.25, 200)
# picos_discreto.write(f"dt=0.25 {m.pico_infectados}\n")
# sirs.write(f"dt=0.25 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.125, 200)
# picos_discreto.write(f"dt=0.125 {m.pico_infectados}\n")
# sirs.write(f"dt=0.125 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.0625, 200)
# picos_discreto.write(f"dt=0.0625 {m.pico_infectados}\n")
# sirs.write(f"dt=0.0625 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.03125, 200)
# picos_discreto.write(f"dt=0.03125 {m.pico_infectados}\n")
# sirs.write(f"dt=0.03125 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.015625, 200)
# picos_discreto.write(f"dt=0.015625 {m.pico_infectados}\n")
# sirs.write(f"dt=0.015625 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.0078125, 200)
# picos_discreto.write(f"dt=0.0078125 {m.pico_infectados}\n")
# sirs.write(f"dt=0.0078125 {m.SIRs}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.00390625, 200)
# picos_discreto.write(f"dt=0.00390625 {m.pico_infectados}\n")
# sirs.write(f"dt=0.00390625 {m.SIRs}\n")

# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.001953125, 200)
# picos_discreto.write(f"dt=0.001953125 {m.pico_infectados}\n")
# m.resetar_grafo()
# m.avançar_tempo_movimentacao_deltaT(0.0009765625, 200)
# picos_discreto.write(f"dt=0.0009765625 {m.pico_infectados}\n")
# m.resetar_grafo()


#print(m.grafo.edges(data=True))
#m.vertice_de_inicio = "Flamengo"
#m.resetar_grafo()
#m.avançar_tempo_movimentacao_otimizado()
#m.printar_grafo()
#path_log = "./Resultados/heuristica/SIR_vertice_por_tempo_heuristica"
#path_picos = "./Resultados/heuristica/picos_por_arvores_e_arestas"
#m.heuristica_arvores_vizinhas("largura", path_log, path_picos)
#m.heuristica_arvores_vizinhas("profundidade", path_log, path_picos)
#m.rodar_modelo_inicio_vertices("./Resultados/picos_inicios_grafo_original.txt", "./Resultados/SIRxT_inicios_grafo_original.txt")

#m.gerar_SIRxT_inicios_rj(".\Resultados\SIRxT_inicios_grafo_originall.txt")

#m.avançar_tempo_movimentacao_otimizado(1)
#print(m.pico_infectados)
#m.boxplot_heuristica_floripa(r"./Resultados/heuristica/com teste nos ciclos/picos_por_arvores_e_arestas_profundidade.txt", "profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_original_florianopolis()
#m.printar_grafico_SIRxT(path="./Resultados/Grafico SIRxT flo.png")

#m.gerar_grafos_arvores_florianopolis("profundidade")

#m.printar_grafico_ID_MAXINFECT_arvores_original_florianopolis()


# print(len(m.grafo.nodes()))
# for bairro in m.grafo.nodes():
#     m.avançar_tempo_movimentacao(200, False)
#     print(f"Inicio {bairro}: {m.pico_infectados}")

#m.printar_tabela_arvores()
#m.montar_tabela_zona_sul_ciclos()

#print(m.grafo.edges())

#m.printar_grafo("zonasul")
#m.avançar_tempo_movimentacao(30)
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

# m.avançar_tempo_movimentacao(dia_aumento_significativo)

# for dia in range(vezes):
#     m.isolar_vertices_mais_populosos()
#     m.avançar_tempo_movimentacao(duraçao_ciclo)

# m.avançar_tempo_movimentacao(dias_sobrando)
# print("tejnpo :",m.t)
# if m.isolados == True:
#     m.isolar_vertices_mais_populosos()

# m.avançar_tempo_movimentacao(200 - m.t + 1)

# print(m.tempo_pico)
# print(m.pico_infectados)
## ! fim teste isolamento


# m.arvores_vizinhas("largura")

# m.printar_grafico_ID_MAXINFECT_arvore(tipo_arvore="largura")
# m.printar_grafico_ID_MAXINFECT_arvore(tipo_arvore="profundidade")
# m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()

#m.printar_tabela_arvores()

#m.printar_grafico_SIRxTdeVerticesPizza()
# m.avançar_tempo_movimentacao_deltaT(0.5, 200)

# print(m.pico_infectados)
# m.printar_grafico_SIRxT()
#m.arvores_vizinhas("profundidade")

#m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()
#m.printar_grafico_ID_MAXINFECT_arvore("profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_profundidade_antes_depois()

