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

# print(os.listdir())
# mod = modelo.Modelo(arquivo_final)
# mod.printar_grafico_ID_MAXINFECT_arvore()


def ler_resultados() -> dict:
    arquivo_resultados = open(resultados_arvore_profundidade, "r", encoding="utf-8")
    resultados = dict()

    for linha in arquivo_resultados:
        if linha == "\n":
            break

        id, dia_pico, pico = linha.strip().split(", ")

        resultados[int(id)] = float(pico)
        
    return resultados


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


def algoritmo_genetico(tipo_arvore):      
    #resultados = ler_resultados()
    resultados = [{"arvore": "Grupo 1", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 50)},
                  {"arvore": "Grupo 2", "adiçao": ("Grupo 3", "Grupo 4"), "remoçao": ("Grupo 2", "Grupo 3"), "pico": (2, 10)},]
    
    melhores = []
    cruzamentos = []
    m = modelo.Modelo(arquivo_final_flo, True)

    melhores = sorted(list(sorted(resultados, key=lambda x: x["pico"][1])[0:20]), key=lambda x: int(x["arvore"].split(" ")[1]))
    #print(melhores)
    a = (1, 2)

    #melhores = dict(sorted(resultados.items(), key=lambda x:x[1])[0:20])
    #print(melhores)
    #print(melhores[0]["adição"].__str__())


    arvores = dict()

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
    arvores_melhores = []

    for v_inicial_arvore in sorted(m.grafo.nodes(), key=lambda x: int(x.split(" ")[1])):
        if tipo_arvore == "largura":
            anterior = busca_em_largura(v_inicial_arvore, grafo_original)

        g = nx.DiGraph()
        g.add_nodes_from(grafo_original)

        for vertice, ant in anterior.items():
            g.add_weighted_edges_from([(ant, vertice, grafo_original.get_edge_data(ant, vertice)["beta"]),
                                        (vertice, ant, grafo_original.get_edge_data(vertice, ant)["beta"])], weight="beta")

        # melhores ordenado por vertice inicial da arvore ["arvore"] #! nao precisa? como fazer
        # montando arvores melhor (adição, remoção) a partir da arvore de busca do grupo X montada acima
        for arvore in melhores:
            if arvore["arvore"] == v_inicial_arvore:    
                t = nx.DiGraph(g)
                
                t.add_weighted_edges_from([(arvore["adiçao"][0], arvore["adiçao"][1], grafo_original.get_edge_data(arvore["adiçao"][0], arvore["adiçao"][1])["beta"]),
                                            (arvore["adiçao"][1], arvore["adiçao"][0], grafo_original.get_edge_data(arvore["adiçao"][1], arvore["adiçao"][0])["beta"])], weight="beta")
                t.remove_edges_from([(arvore["remoçao"][0], arvore["remoçao"][1]),(arvore["remoçao"][1], arvore["remoçao"][0])])
                
                arvores_melhores.append(t)
    

    # criando os cruzamentos entre as melhores
    for arvore1 in melhores:
        arvore1 = (arvore1["arvore"], arvore1["adiçao"], arvore1["remoçao"])
        for arvore2 in melhores:
            arvore2 = (arvore2["arvore"], arvore2["adiçao"], arvore2["remoçao"])
            
            if arvore1 != arvore2 and random.randint(0, 1):   
                cruzamentos.append((arvore1, arvore2))


    # fazendo o cruzamentos das arvores de fato (salvar arvores montadas, arvores_g1)
    # uma em cima da outra, loop remover ciclos (ordem muda por iteração?)
    # por fim chance de mutação (adiciona uma aresta, ou 1-6 arestas, que ja existe no grafo original e remover ciclos)
    # -> nova arvore, salvar pais ao lado da arvore montada
    arvores_cruzadas = []

    
    # rodar arvores_cruzadas modelo, salvar dia, pico em uma arquivo e SIRt em outro

    # selecionar 20 melhores

    return
    
    
    arvores_g0 = open("./arvores_g0.txt", "a", encoding="utf-8", buffering=1)

    for arvore in melhores.keys():
        anterior = {}
        visitados = set()
        busca_em_profundidade(id_nomes[arvore], anterior, visitados, grafo_original)

        m.grafo.remove_edges_from(list(m.grafo.edges()))

        for vertice, ant in anterior.items():   # recriar grafo a partir de anterior
            m.grafo.add_edge(ant, vertice)

        for vertice in m.grafo.nodes(data=True):                         # setar betas novamente
            vertice[1]["beta"] = 1 / (len(m.grafo.edges(vertice[0])) + 1)
        
        arvores[arvore] = m.grafo.copy()
        arvores_g0.write(f"{id_nomes[arvore]}: {m.grafo.edges()}\n")

        m.grafo = grafo_original

    arvores_g0.close()

    # fazer OR das arvores, dps a busca em profundidade removendo os ciclos
    # busca vai percorrendo vizinho (que nao é o anteriormente visitado), se achar algum que ja foi visitado
    # ciclo, portanto remover ultima aresta visitada
    # isso já é suficiente? precisa de mais buscas?
    #print(cruzamentos)

    arvores_g1 = open("./arvores_g1.txt", "a", encoding="utf-8")
    filhos = []

    for arvore1, arvore2 in cruzamentos:
        #print(arvore1, arvore2)

        #print(arvore1.edges())
        #print(arvore2.edges())
        arvore_filho = nx.compose(arvores[arvore1], arvores[arvore2])
        #print(arvore1.edges() == arvore2.edges())
        #print(arvore_filho.edges("Centro"))
        #print(arvore_filho)

        visitados = set()
        while True:
            try:
                a = nx.find_cycle(arvore_filho, "Flamengo")
                arvore_filho.remove_edge(a[-1][0], a[-1][1])
                print("Removendo aresta de ciclo:", a[0], a[-1])
                #break
            except nx.exception.NetworkXNoCycle:
                break

        arvores_g1.write(f"{id_nomes[arvore1]}, {id_nomes[arvore2]}: {arvore_filho.edges()}\n")

        filhos.append(arvore_filho)
        #print(a)
        #retirar_ciclos(arvore_filho, "Flamengo", "Flamengo", visitados)         

    #cruzar_arvores(arvore1, arvore2)
    arvores_g1.close()

    print(len(filhos))




    # apos cruzar todas, salvar arvores em arquivo e começar a rodar elas no modelo
        
    # graficos

    #print(resultados)

def testar_arvores(arvores):
    for arvore in arvores:
        pass
    
def heuristica_arvores_vizinhas(self, tipo_arvore, path_arquivo_log: str, path_arquivo_picos: str, ):
    path_arquivo_log = f"{path_arquivo_log}_{tipo_arvore}.txt"
    path_arquivo_picos = f"{path_arquivo_picos}_{tipo_arvore}.txt"

    arquivo_log = open(path_arquivo_log, "a", encoding="utf-8", buffering=1)
    arquivo_picos = open(path_arquivo_picos, "a", encoding="utf-8", buffering=1)
    arquivo_picos.write(f"Inicio da arvore:\n v-u adicionada (cria ciclo) pico dia_pico fim_espalhamento\n  x-y removida (volta a ser arvore) pico dia_pico fim_espalhamento\n")

    grafo_original = self.grafo.copy()
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


algoritmo_genetico("largura")