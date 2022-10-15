def montar_grafo(arquivo):
    vertices = []
    grafo = dict()
    passado = -1
    i = -1

    linhas = arq_grafo.readlines()
    
    for linha in linhas:
        linha = linha.split(",")
        vertice_a = linha[0]
        vertice_b = linha[1].strip("\n")

        if vertice_a in grafo.keys():
            grafo[vertice_a].append(vertice_b)
        else:
            vertices.append(vertice_a)
            grafo[vertice_a] = [vertice_b]

    vertices.sort()
    print(grafo, vertices)
    return grafo, vertices


def printar_matriz(grafo, vertices_ordem):  
    matriz = []

    print("   ", end="")
    for vertice in vertices_ordem:  # primeira linha
        print(vertice, end="  ")
        matriz.append(grafo[vertice])
    print("")

    for linha in vertices_ordem:    #resto das linhas
        print(linha, end="  ")
        for coluna in vertices_ordem:
            print(f"{grafo[linha].count(coluna)}  ", end="")
        print("")


arq_grafo = open("Algoritmos Grafos\grafo.txt", "r")

grafo, vertices = montar_grafo(arq_grafo)
printar_matriz(grafo, vertices)

arq_grafo.close()

def tem_caminho(g_inicio, g_final, i):  #arrumar
    if len(grafo[g_inicio]) == i+1:
        i = 0

    print("g_inicio:", g_inicio, "indice:", i)
    print(grafo[g_inicio][i])

    g_inicio = grafo[g_inicio][i]

    if g_inicio == g_final:
        return True

    i += 1
    return tem_caminho(g_inicio, g_final, i)


i = 0
print(tem_caminho("1","9",i))