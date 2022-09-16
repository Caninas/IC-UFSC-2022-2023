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

    dic[vertice_a] = []
    if vertice_a != passado:
        i += 1
        vertices.append(vertice_a)
        matriz.append([])
        matriz[i].append(vertice_b)
    else:
        matriz[i].append(vertice_b)
    passado = vertice_a

print(matriz , vertices)

for vertice in vertices:
    print("   ", vertice, end="")

print("\n", end="")
for vertice in vertices:
    print(matriz[vertices.index(vertice)])

    print(vertice, "  X", "\n")
    
#for i in matriz:



