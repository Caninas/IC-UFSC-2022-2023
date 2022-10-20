class Grafo:
    def __init__(self, arquivo):
        self.vertices = []
        self.grafo = dict()
        self.arquivo = open(arquivo)
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



arq_grafo = "Algoritmos Grafos/grafo.txt"

grafo = Grafo(arq_grafo)
grafo.printar_matriz()
print(grafo.grafo)
grafo.tem_caminho("2", "2")
print(grafo.caminho)
