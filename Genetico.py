import importlib.util
import sys, os
import networkx as nx
import random

spec = importlib.util.spec_from_file_location("Modelo", "Instancia RJ/Modelo.py")
modelo = importlib.util.module_from_spec(spec)
sys.modules["Modelo"] = modelo
spec.loader.exec_module(modelo)


arquivo_adjacencias = "./Instancias\outros\zona sul modificada ciclos/adjacencias_zona_sul.txt"
arquivo_final =  "../Instancia RJ/Instancias/normal (real)/arquivo_final.txt" #"./Instancias/outros\zona sul/arquivo_final.txt"
arquivo_final_flo = "../Instancia Florianopolis/Instancias/normal/arquivo_final.txt"
arquivo_ID_nomes = "./Instancias/nova relaçao ID - bairros.txt"
tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"


resultados_arvore_profundidade = "../Resultados/resultados_arvore_profundidade.txt"
SIRxTdeVerticesTXT_profundidade = "./Resultados/SIR_vertice_por_tempo_PROFUNDIDADE.txt"
resultados_arvore_largura = "./Resultados/resultados_arvore_largura.txt"
SIRxTdeVerticesTXT_largura = "./Resultados/SIR_vertice_por_tempo_LARGURA.txt"
path_picos_heuristica = "../Resultados/picos_por_arvores_e_arestas_largura.txt"

# print(os.listdir())
# mod = modelo.Modelo(arquivo_final)
# mod.printar_grafico_ID_MAXINFECT_arvore()


#def ler_resultados() -> dict:
    # arquivo_resultados = open(path_picos_heuristica, "r", encoding="utf-8")
    # resultados = []

    # resultados = [{"arvore": "Grupo 1", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 50)},
    #               {"arvore": "Grupo 2", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 10)},]
    
    # for i in range(4):
    #     arquivo_resultados.readline()

    # for linha in arquivo_resultados:
    #     if linha[0] == "I":
    #         inicio_arvore = linha.strip().split(":")[0][7:15]
    #         print(inicio_arvore)
        
    #     if linha[0] == " ":
    #         if linha[1] == " ":


    #     #id, dia_pico, pico = linha.strip().split(", ")

    #     #resultados[int(id)] = float(pico)
        
    # return resultados


def retirar_ciclos(grafo: nx.Graph, inicio: int, v_anterior: int, visitados: set) -> None:
    # ACHAR COMO FINALIZAR A BUSCA
    visitados = set()
    
    if inicio in visitados: #?
        return

    visitados.add(inicio)

    for vizinho in grafo.edges(inicio):
        vizinho = vizinho[1]
        if vizinho in visitados:
            grafo.remove_edge(inicio, vizinho)
            continue

        if vizinho != v_anterior:
            retirar_ciclos(grafo, vizinho, inicio, visitados)


def cruzar_arvores(arvore1: nx.Graph, arvore2: nx.Graph) -> nx.Graph:
    filho = nx.symmetric_difference(arvore1, arvore2)
    
    pass

def escolher_melhores_arvores(geracao: int) -> list:
    pass

    # escolhe as arvores com melhores números de picos na geraçao especificada
    # geraçao 0 = arvores de testes anteriores (iniciais)


def algoritmo_genetico(tipo_arvore="largura"):      
    #resultados = ler_resultados()
    resultados = [{"arvore": "Grupo 1", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 50)},
                  {"arvore": "Grupo 2", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 10)},]
    
    melhores = []
    m = modelo.Modelo(arquivo_final_flo, True)

    melhores = {id: arvore for id, arvore in enumerate(sorted(resultados, key=lambda x: x["pico"][1])[0:20])}

    #melhores = dict(sorted(resultados.items(), key=lambda x:x[1])[0:20])


    def busca_em_profundidade(v, anterior, visitados, grafo):
        visitados.add(v)

        for vizinho in grafo.edges(v):
            vizinho = vizinho[1]
            if vizinho not in visitados:
                anterior[vizinho] = v
                busca_em_profundidade(vizinho, anterior, visitados, grafo)

    def busca_em_largura(inicio, grafo_original):
        fila = []
        visitados = {inicio}
        anterior = {}
        fila.append(inicio)

        while len(fila):
            v = fila.pop(0)

            for vizinho in grafo_original.edges(v):
                vizinho = vizinho[1]
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    fila.append(vizinho)
                    anterior[vizinho] = v

        return anterior

    grafo_original = m.grafo.copy()

    # criação das arvores a partir das buscas, para depois fazer adições, remoções das arvores selecionadas
    # nos cruzamentos
    
    # arvores melhores montadas
    arvores_melhores = dict()
    arvores_g0 = open("./arvores_g0.txt", "w", encoding="utf-8", buffering=1)

    for v_inicial_arvore in sorted(m.grafo.nodes(), key=lambda x: int(x.split(" ")[1])):
        if tipo_arvore == "largura":
            anterior = busca_em_largura(v_inicial_arvore, grafo_original)

        # arvore de busca que inicia em v_inicial_arvore
        g = nx.DiGraph()
        g.add_nodes_from(grafo_original.nodes(data=True))
        
        for vertice, ant in anterior.items():
            g.add_weighted_edges_from([(ant, vertice, grafo_original.get_edge_data(ant, vertice)["beta"]),
                                        (vertice, ant, grafo_original.get_edge_data(vertice, ant)["beta"])], weight="beta")
        
        
        # melhores ordenado por vertice inicial da arvore ["arvore"] #! nao precisa? como fazer
        # montando arvores melhor (adição, remoção) a partir da arvore de busca do grupo X montada acima
        for id, arvore in melhores.items():
            if arvore["arvore"] == v_inicial_arvore:    
                t = nx.DiGraph(g, pais={"arvore": arvore["arvore"], "adiçao": arvore["adiçao"], "remoçao": arvore["remoçao"]})
                
                
                for node in t.nodes():
                    del t.nodes[node]["id"]

                t.add_weighted_edges_from([(arvore["adiçao"][0], arvore["adiçao"][1], grafo_original.get_edge_data(arvore["adiçao"][0], arvore["adiçao"][1])["beta"]),
                                            (arvore["adiçao"][1], arvore["adiçao"][0], grafo_original.get_edge_data(arvore["adiçao"][1], arvore["adiçao"][0])["beta"])], weight="beta")
                t.remove_edges_from([(arvore["remoçao"][0], arvore["remoçao"][1]),(arvore["remoçao"][1], arvore["remoçao"][0])])

                arvores_melhores[id] = t
                
                arvores_g0.write(f"{id}, pais: {(arvore['arvore'], arvore['adiçao'], arvore['remoçao'])}, {g.edges(data=True)}\n")
    
    arvores_g0.close()
    #print(melhores)
    #print(arvores_melhores)

    filhos = dict()
    id_filho = 0
    arvores_g1 = open("./arvores_g1.txt", "a", encoding="utf-8", buffering=1)

    #print(arvores_melhores[0].nodes(data=True))
    # criando os cruzamentos entre as melhores
    for id1, arvore1 in arvores_melhores.items():
        #arvore1 = (arvore1["arvore"], arvore1["adiçao"], arvore1["remoçao"])
        for id2, arvore2 in arvores_melhores.items():
            #arvore2 = (arvore2["arvore"], arvore2["adiçao"], arvore2["remoçao"])

            if id1 != id2 and random.randint(0, 1):
                filho = nx.DiGraph(arvore1, pais=(id1, id2))
                filho.add_edges_from(arvore2.edges(data=True))
                for ciclo in nx.simple_cycles(filho):
                    if len(ciclo) > 2:
                        filho.remove_edge(ciclo[0], ciclo[1])


                #! mutaçao aqui 1%
                # por fim chance de mutação (adiciona uma aresta, que ja existe no grafo original e remover ciclos)

                filhos[id_filho] = filho
                arvores_g1.write(f"{id_filho}, pais: {(id1, id2)}, {filho.edges(data=True)}\n")
                
                id_filho += 1
    
    arvores_g1.close()


    for id, arvore_filha in filhos.items():
        m.grafo = arvore_filha
        m.resetar_grafo()

        m.avançar_tempo_movimentacao_otimizado(printar=True)

        # pico = [x+y for x, y in zip(pico, [m.pico_infectados, m.tempo_pico, m.t])]
        # pico = [int(x/3) for x in pico]

        # arquivo_picos.write(f"  {x}-{y} {self.pico_infectados} {self.tempo_pico} {self.t}\n")
        # arquivo_log.write(f"{self.SIRxTdeVertices}\n")

    
    # rodar arvores_cruzadas modelo, salvar dia, pico em uma arquivo e SIRt em outro

    # selecionar 20 melhores

    #return
    


    # apos cruzar todas, salvar arvores em arquivo e começar a rodar elas no modelo
        
    # graficos

    #print(resultados)



algoritmo_genetico("largura")





    
# arvores_g0 = open("./arvores_g0.txt", "a", encoding="utf-8", buffering=1)

# for arvore in melhores.keys():
#     anterior = {}
#     visitados = set()
#     busca_em_profundidade(id_nomes[arvore], anterior, visitados, grafo_original)

#     m.grafo.remove_edges_from(list(m.grafo.edges()))

#     for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
#         m.grafo.add_edge(ant, vertice)

#     for vertice in m.grafo.nodes(data=True):                         # setar betas novamente
#         vertice[1]["beta"] = 1 / (len(m.grafo.edges(vertice[0])) + 1)
    
#     arvores[arvore] = m.grafo.copy()
#     arvores_g0.write(f"{id_nomes[arvore]}: {m.grafo.edges()}\n")

#     m.grafo = grafo_original

# arvores_g0.close()

# # fazer OR das arvores, dps a busca em profundidade removendo os ciclos
# # busca vai percorrendo vizinho (que nao é o anteriormente visitado), se achar algum que ja foi visitado
# # ciclo, portanto remover ultima aresta visitada
# # isso já é suficiente? precisa de mais buscas?
# #print(cruzamentos)

# arvores_g1 = open("./arvores_g1.txt", "a", encoding="utf-8")
# filhos = []

# for arvore1, arvore2 in cruzamentos:
#     #print(arvore1, arvore2)

#     #print(arvore1.edges())
#     #print(arvore2.edges())
#     arvore_filho = nx.compose(arvores[arvore1], arvores[arvore2])
#     #print(arvore1.edges() == arvore2.edges())
#     #print(arvore_filho.edges("Centro"))
#     #print(arvore_filho)

#     visitados = set()
#     while True:
#         try:
#             a = nx.find_cycle(arvore_filho, "Flamengo")
#             arvore_filho.remove_edge(a[-1][0], a[-1][1])
#             print("Removendo aresta de ciclo:", a[0], a[-1])
#             #break
#         except nx.exception.NetworkXNoCycle:
#             break

#     arvores_g1.write(f"{id_nomes[arvore1]}, {id_nomes[arvore2]}: {arvore_filho.edges()}\n")

#     filhos.append(arvore_filho)
#     #print(a)
#     #retirar_ciclos(arvore_filho, "Flamengo", "Flamengo", visitados)         

# #cruzar_arvores(arvore1, arvore2)
# arvores_g1.close()

# print(len(filhos))