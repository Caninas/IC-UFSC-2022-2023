arq_grafo = open("IC\Algoritmos Grafos\grafo.txt", "r")

linhas = arq_grafo.readlines()

dic = dict()
matriz = []
passado = -1
vertices = []
i = -1

for linha in linhas:
    linha = linha.split(",")
    vertice_a = linha[0]
    vertice_b = linha[1].strip("\n")

    if vertice_a != passado:
        dic[vertice_a] = []
        i += 1
        dic[vertice_a].append(vertice_b)
        # matriz.append([])
        # matriz[i].append(vertice_b)
    else:
        dic[vertice_a].append(vertice_b)
        #matriz[i].append(vertice_b)
    passado = vertice_a

print(dic , vertices)

for vertice in vertices:
    print("   ", vertice, end="")

print("\n", end="")
for vertice in vertices:
    print(matriz[vertices.index(vertice)])

    print(vertice, "  X", "\n")
    
#for i in matriz:



