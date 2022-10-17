class Grafo:
    def __init__(self, arquivo):
        self.vertices = []
        self.grafo = dict()
        self.matriz = []
        self.i = 0
        self.arquivo = open(arquivo)
        self.grafo_atual = 0
        self.visitados = []
        self.caminho = []

        self.tem = False

        self.montar_grafo()

    def montar_grafo(self):
        linhas = self.arquivo.readlines()
        
        for linha in linhas:
            linha = linha.split(",")
            vertice_a = linha[0]
            vertice_b = linha[1].strip("\n")

            if vertice_a in self.grafo.keys():
                self.grafo[vertice_a].append(vertice_b)
                if vertice_b not in self.grafo.keys():
                    self.vertices.append(vertice_b)
                    self.grafo[vertice_b] = []
            else:
                self.vertices.append(vertice_a)
                self.grafo[vertice_a] = [vertice_b]

        self.vertices.sort()

    def printar_matriz(self):  
        print("   ", end="")
        for vertice in self.vertices:  # primeira linha
            print(vertice, end="  ")
            self.matriz.append(self.grafo[vertice])
        print("")

        for linha in self.vertices:    #resto das linhas
            print(linha, end="  ")
            for coluna in self.vertices:
                print(f"{self.grafo[linha].count(coluna)}  ", end="")
            print("")

    def tem_caminho(self, g_inicio, g_final):
        vertice_a = g_inicio        # vertice principal, do qual sai o loop
        for vertice in self.grafo[g_inicio]:
            if vertice not in self.visitados and self.tem == False:
                self.caminho.append(f"{vertice_a} -> {vertice}")
                self.visitados.append(vertice) 
                g_inicio = vertice

                if g_inicio == g_final:
                    self.tem = True
                    return

                self.tem_caminho(g_inicio, g_final)


arq_grafo = "IC/Algoritmos Grafos/grafo.txt"

grafo = Grafo(arq_grafo)
grafo.printar_matriz()
print(grafo.grafo)
grafo.tem_caminho("9", "5")
print(grafo.visitados)
print(grafo.tem)
print(grafo.caminho)

# print(grafo.grafo)
# print(grafo.matriz)