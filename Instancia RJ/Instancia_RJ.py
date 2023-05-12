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
import time

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

        self.grafo = self.gerar_grafo()     # grafo utilizado atualmente
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
                beta=float(beta))
            g.add_edges_from(adj)

        return g


    def resetar_grafo(self):
        for vertice in self.grafo:
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

    def estimar_tempo_restante(self, percorrido, total):
        if percorrido:
            estimativa = ((time.perf_counter() - self.estimativa_tempo_inicial) / percorrido) * (total - percorrido)
        else:
            self.estimativa_tempo_inicial = time.perf_counter()
            estimativa = 0
        print(f"Estimativa {estimativa/60:.2f} minutos | {total - percorrido} arvores restantes")


    def salvar_grafo_arvore(self, path):
        mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}            
        g = nx.relabel_nodes(self.grafo, mapping)

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
        fig.set_size_inches(30, 10)
        plt.savefig(path, format="png", dpi=300)

    def printar_grafo(self, tipo=None):
        # #pos = nx.circular_layout(self.grafo.subgraph(("Ipanema"...)))
        # pos = {'Ipanema': array([1.0000000e+00, 1.4702742e-08]), 'Glória': array([0.809017  , 0.58778526]), 'Catete': array([0.30901698, 0.95105655]),
        # 'Laranjeiras': array([-0.30901702,  0.95105649]), 'Cosme Velho': array([-0.80901699,  0.58778526]), 'Urca': array([-9.99999988e-01, -7.27200340e-08]),
        # 'Leme': array([-0.80901693, -0.58778529]), 'São Conrado': array([-0.30901711, -0.95105646]), 'Vidigal': array([ 0.30901713, -0.95105646]),
        # 'Leblon': array([ 0.80901694, -0.58778529]),
        # 'Gávea': (0, -0.2), 'Flamengo': (0, 0.4), 'Botafogo': (-0.5, 0.2), 'Humaitá': (0.25, 0.1), 'Copacabana': (-0.2, -0.5),
        # 'Lagoa': (0.4, -0.25), 'Jardim Botânico': (0.6, 0.6), 'Rocinha': (0.7, 0.1)}

        #T = nx.balanced_tree(2, 5)
        #nx.spring_layout(self.grafo)
        
        #pos = {'Flamengo': array([0.6043461, 0.4442784]), 'Laranjeiras': array([0.45074005, 0.55273503]), 'Glória': array([0.8534418 , 0.58982338]), 'Botafogo': array([0.31947341, 0.18126152]), 'Catete': array([0.68333495, 0.64406827]), 'Cosme Velho': array([0.42134842, 0.85813163]), 'Humaitá': array([-0.04907372,  0.02847084]), 'Copacabana': array([ 0.11292418, -0.20412333]), 'Urca': array([0.57723837, 0.07557802]), 'Jardim Botânico': array([-0.34296672, -0.06957464]), 'Lagoa': array([-0.2287956 , -0.22462745]), 'Leme': array([ 0.30783336, -0.43993586]), 'Ipanema': array([-0.13604816, -0.39443646]), 'Leblon': array([-0.42540793, -0.42112642]), 'Gávea': array([-0.55692957, -0.28087638]), 'Vidigal': array([-0.72900454, -0.46273238]), 'Rocinha': array([-0.8624544 , -0.36491218]), 'São Conrado': array([-1.        , -0.51200198])}
        #print(pos)
        if tipo:
            mapping = {old_label:new_label["id"] for old_label, new_label in self.grafo.nodes(data=True)}
            
            g = nx.relabel_nodes(self.grafo, mapping)

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
        fig.set_size_inches([9, 6])


        plt.gca().set_prop_cycle('color', ['red', '#55eb3b', 'blue'])
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        if path:
            plt.xlim(left=1, right=len(x))
            plt.plot(x, y)
        else:
            plt.xlim(left=self.tempos[0], right=self.tempos[-1])
            plt.plot(self.tempos, self.SIRs)

        ax.legend(["S", "I", "R"], loc='center right', bbox_to_anchor=(1.1, 0.5))
        ax.set_xlabel('Tempo')
        ax.set_ylabel('Pessoas')


        if not path:
            plt.show()
            #plt.close()
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

    def busca_em_profundidade(self, v, anterior, visitados):
        visitados.add(v)
        for vizinho in self.grafo_original.edges(v):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                anterior[vizinho] = v
                self.busca_em_profundidade(vizinho, anterior, visitados)

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
                print("Inicio:", inicio, "/ Iteração:", i+1)

                self.avançar_tempo_movimentacao_dinamica(tempo)
                print("Pico:", self.pico_infectados)

                soma_pico += self.pico_infectados
                tempo_pico += self.tempo_pico

                self.resetar_grafo()
            
            tempo_pico = tempo_pico / (i + 1)
            media = soma_pico / (i + 1)
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

        #plt.gca().set_prop_cycle('color', ['green', '0d66a3'])
        plt.gca().set_prop_cycle('color', ['0d66a3'])
        plt.plot([x for x in range(1, 160)], resultados_lista, "o")    # valores arvores
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

        #ax.legend(["Grafo Normal", "Largura", "Profundidade"], loc='center right', bbox_to_anchor=(1.127, 0.5))
        ax.legend(["Grafo Normal", tipo_arvore.title()], loc='center right', bbox_to_anchor=(1.127, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Real)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(fr"C:\Users\rasen\Desktop\Resultados\com betas\Pico Infectados Arvores {tipo_arvore.title()} FINAL.png", format="png", dpi=300)

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

        ax.legend(["Grafo Normal", "Largura", "Profundidade"], loc='center right', bbox_to_anchor=(1.127, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Real)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(fr"C:\Users\rasen\Desktop\Resultados\com betas\Pico Infectados Arvores Largura e Profundidade FINAL.png", format="png", dpi=300)

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

        ax.legend(["Grafo Normal", "Antes", "Depois"], loc='center right', bbox_to_anchor=(1.127, 0.5))
        #ax.legend(["Grafo Normal", tipo_arvore.title()], loc='center right', bbox_to_anchor=(1.127, 0.5))

        plt.title(titulo)
        ax.set_xlabel('ID de Início da Árvore (0 = Grafo Real)')
        ax.set_ylabel('Pico de Infectados')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        #C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ\Resultados\Pico Infectados Arvores {tipo_arvore.title()}.png
        plt.savefig(fr"C:\Users\rasen\Desktop\Resultados\com betas\Pico Infectados Arvores Profundidade 200x400 dias.png", format="png", dpi=300)
        plt.close

    def arvores_vizinhas(self, tipo_arvore):
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
            if tipo_arvore == "largura":
                anterior = self.busca_em_largura(inicio)
            else:
                visitados = set()
                self.busca_em_profundidade(inicio, anterior, visitados)

            self.grafo.remove_edges_from(list(self.grafo.edges()))
            for vertice, ant in anterior.items():
                self.grafo.add_edge(ant, vertice)

            for vertice in self.grafo.nodes(data=True):                         # setar betas novamente
                vertice[1]["beta"] = 1 / (len(self.grafo.edges(vertice[0])) + 1)
        

            #print(g.nodes(data=True))

            grafo_complemento = nx.complement(self.grafo).edges()

            for v, u in grafo_complemento:      # adiçao da aresta que cria ciclo
                print("Criando", v, "-->", u)
                anterior = {}
                visitados = set()
                self.encontrou_ciclo = False
                ciclo = []

                def busca_em_profundidade(k, anterior, visitados):          # achar ciclo
                    visitados.add(k)
                    for vizinho in self.grafo.edges(k):
                        if not self.encontrou_ciclo:
                            vizinho = vizinho[1]

                            if vizinho not in visitados:
                                anterior[vizinho] = k
                                if vizinho == v:
                                    print("Encontrou ciclo:", end="")
                                    self.encontrou_ciclo = True
                                    break
                                busca_em_profundidade(vizinho, anterior, visitados)

                busca_em_profundidade(u, anterior, visitados)

                ciclo.append(v)
                while True:                 # montar ciclo
                    try:
                        if anterior[ciclo[-1]] != v:
                            ciclo.append(anterior[ciclo[-1]])
                        else:
                            raise Exception
                    except:
                        break

                # achar ciclo
                # busca em profundidade com comparaçao atual == v
                # ir salvando caminho até chegar no final, quando começar a voltar
                # remover quando ele parar e entrar no if nao visitado
                # a partir de onde parou +1 e começar montando dnv

                print(ciclo)
                self.printar_grafo("arvore")
                #continue
                self.grafo.add_edge(v, u)

                for vertice in [v, u]:          # atualizar betas com nova aresta
                    self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)

                #? rodar modelo com ciclo?

                for indice in range(0, len(ciclo) - 1):    # loop arestas do ciclo (tirando v, u)
                    x = ciclo[indice]
                    y = ciclo[indice + 1]

                    self.grafo.remove_edge(x, y)

                    for vertice in [x, y]:          # atualizar betas com nova aresta
                        self.grafo.nodes[vertice]["beta"] = 1 / (len(self.grafo.edges(vertice)) + 1)
            
                    # rodar modelo
                    self.avançar_tempo_movimentacao_dinamica_otimizado()
                    # salvar SIRT (criar outro modelo com SIRT diferente pra cá e otimizado)
                    # salvar resultado (id, dia, pico)
                    
                    # salvar grafico SIR (arvore com v, u sem x, y.png)

                    self.grafo.add_edge(x, y)

            print(type(grafo_complemento["Copacabana", "Flamengo"]))
            # self.grafo.remove_edges_from(list(self.grafo.edges()))
            # self.grafo.add_edges_from(g.edges())
            # print(len(self.grafo.edges()))
            # print(self.grafo.nodes(data=True))
            # self.resetar_grafo()
            # print(self.grafo.nodes(data=True))


            # salvar arvore original
            self.salvar_grafo_arvore(fr"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ\Resultados\dsdsaddas\{inicio}.png")
        self.grafo = self.grafo_original 

    def avançar_tempo_movimentacao_dinamica_otimizado(self, t):    # s = nome vertice de origem (no caso de utilizar um grafo arvore)
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

#? Escrever resultados etc
#? Salvar arquivos relevantes drive e separado

#os.chdir(r"C:\Users\rasen\Documents\GitHub\IC Iniciação Científica\Instancia RJ")
os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

# "./txts/normal (real)/adjacencias.txt"
# "./txts/zona sul/arquivo_final.txt"
# "./txts/zona sul/adjacencias_zona_sul.txt"
# "./txts/otimizado/adjacencias.txt"
# "./txts/zona sul modificada menor/adjacencias_zona_sul_sem_botafogo.txt"
arquivo_adjacencias = "./txts/zona sul modificada menor/adjacencias_zona_sul_sem_botafogo.txt"
arquivo_final = "./txts/normal (real)/arquivo_final.txt"
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
m = Modelo(arquivo_final)

#m.avançar_tempo_movimentacao_dinamica(200)
#m.avançar_tempo_movimentacao_dinamica_otimizado(200)
# m.arvores_vizinhas("largura")
#m.printar_grafo()

# m.gerar_grafos_arvore_largura(200, 1) # FEITO
m.printar_grafico_SIRxTdeVerticesPizzaTXT(SIRxTdeVerticesTXT_largura, "largura") # FEITO
# os.rename(SIRxTdeVerticesTXT_largura, "./Resultados/Graficos SIRxT arvores largura/SIR_vertice_por_tempo_LARGURA.txt")
# os.rename(resultados_arvore_largura, "./Resultados/Graficos SIRxT arvores largura/resultados_arvore_largura.txt")

# m.gerar_grafos_arvore_profundidade(200, 1) # FEITO
# m.printar_grafico_SIRxTdeVerticesPizzaTXT(SIRxTdeVerticesTXT_profundidade, "profundidade 200") # FEITO
# os.rename(SIRxTdeVerticesTXT_profundidade, "./Resultados/Graficos SIRxT arvores profundidade 200/SIR_vertice_por_tempo_PROFUNDIDADE.txt")
# os.rename(resultados_arvore_profundidade, "./Resultados/Graficos SIRxT arvores profundidade 200/resultados_arvore_profundidade.txt")

# m.gerar_grafos_arvore_profundidade(400, 1) # FEITO
#m.printar_grafico_SIRxTdeVerticesPizzaTXT(SIRxTdeVerticesTXT_profundidade, "profundidade 400") # FEITO
# os.rename(SIRxTdeVerticesTXT_profundidade, "./Resultados/Graficos SIRxT arvores profundidade 400/SIR_vertice_por_tempo_PROFUNDIDADE.txt")
# os.rename(resultados_arvore_profundidade, "./Resultados/Graficos SIRxT arvores profundidade 400/resultados_arvore_profundidade.txt")

# print(m.pico_infectados)
#m.printar_grafico_SIRxTdeVerticesPizza()
#m.printar_grafico_SIRxT()
# m.avançar_tempo_movimentacao_dinamica_nao_discreto(0.5, 200)

# print(m.pico_infectados)
# m.printar_grafico_SIRxT()
#m.arvores_vizinhas("profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_largura_profundidade()
#m.printar_grafico_ID_MAXINFECT_arvore("profundidade")
#m.printar_grafico_ID_MAXINFECT_arvores_profundidade_antes_depois()

