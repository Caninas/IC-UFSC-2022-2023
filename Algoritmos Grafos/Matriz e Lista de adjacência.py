from random import randint
import math

class Grafo:
    def __init__(self, arquivo):
        self.vertices = []
        self.grafo = dict()
        self.arquivo = open(arquivo)
        self.visitados = []
        self.caminho = []
        self.tem = False
        
        self.montar_grafo()

        self.t = 0

        self.S = 1000
        self.I = 10
        self.R = 0
        
        self.X = randint(1, 30)

        self.taxa_virulencia = 0.55
        self.taxa_recuperaçao = 0.1
        #self.taxa_distanciamento = 0

    def montar_grafo(self):
        linhas = self.arquivo.readlines()
        
        for linha in linhas:
            linha = linha.split(",")
            vertice_a = linha[0]
            vertice_b = linha[1].strip("\n")
            if vertice_a in self.grafo.keys():
                self.grafo[vertice_a].append(vertice_b)
            else:
                self.vertices.append(vertice_a)
                self.grafo[vertice_a] = [vertice_b]
                if vertice_b not in self.grafo.keys():
                    self.vertices.append(vertice_b)
                    self.grafo[vertice_b] = []

        self.vertices.sort()

    def printar_matriz(self):  
        print("   ", end="")
        for vertice in self.vertices:  # primeira linha
            print(vertice, end="  ")
        print("")

        for linha in self.vertices:    #resto das linhas
            print(linha, end="  ")
            for coluna in self.vertices:
                print(f"{self.grafo[linha].count(coluna)}  ", end="")
            print("")

    def tem_caminho(self, g_inicio, g_final):
        vertice_a = g_inicio        # vertice principal, do qual sai o loop
        for vertice in self.grafo[g_inicio]:
            if self.tem == True:
                break

            if vertice not in self.visitados:
                self.caminho.append(f"{vertice_a} -> {vertice}")
                self.visitados.append(vertice) 
                g_inicio = vertice

                if g_inicio == g_final:
                    self.tem = True
                    return

                self.tem_caminho(g_inicio, g_final)

    def avançar_t(self, t):
        for i in range(0, t):
            e = int(self.taxa_recuperaçao * self.I)
            v = int(self.taxa_virulencia * self.X)
            self.R = self.R + e
            print(f"Contatos Adequados: {self.X} \nNovos Infectados: {v}")
            self.I = self.I - e + v
            self.S = self.S - v
            self.X = randint(1, int(0.2*self.S))
            self.info()
            print("")
        
    def info(self):
        print(f"I: {(self.I)} \nS: {(self.S)} \nR: {(self.R)}")
        #print(f"Valores Reais:\nI: {self.I} \nS: {self.S} \nR: {self.R}")
        print(f"Total: {round(self.I) + round(self.S) + round(self.R)}")


arq_grafo = "Algoritmos Grafos/grafo.txt"

grafo = Grafo(arq_grafo)
grafo.printar_matriz()
print(grafo.grafo)
grafo.info()
grafo.avançar_t(50)
#grafo.info()
