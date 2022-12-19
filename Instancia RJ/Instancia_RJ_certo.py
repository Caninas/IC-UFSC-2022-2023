import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode

class Grafo:
    def __init__(self, arq_final):                          # arquivo no formato | int(id), adjs(sep=", ")/propriedade1(id), propriedade2(name).... |
        self.arquivo = open(arq_final, encoding="utf-8")
        self.grafo = nx.Graph()

    def GerarGrafo(self):
        for linha in self.arquivo:
            adj, propriedades = linha.split("/")

            self.grafo





num_bairros = {}

def arquivotxt(adjacencias, nomes, destino):            # modela os dados no arquivo final (que sera lido)
    for linha in nomes:
        linha = linha.strip()

        if linha != "":
            num, bairro = linha.split(" ", maxsplit=1)
            num = int(num)

            #nome = unidecode(bairro.lower()).replace(" ", "_")

            num_bairros[num] = bairro

    bairros_adjs = {}
    cidades_ja_visitadas = set()
    escrever = []

    for linha in adjacencias:
        linha = linha.strip()
        
        if linha != "":
            num_bairro = int(linha.split(" ", maxsplit=1)[0])

            if num_bairro not in cidades_ja_visitadas:
                cidades_ja_visitadas.add(num_bairro)

                adjs = [x for x in linha.split(" ")[1:]]
                for cidade in adjs:
                    if cidade in cidades_ja_visitadas:
                        adjs.remove(cidade)

                bairros_adjs[num_bairro] = adjs

                escrever.append(f"{num_bairro}" + f"{', ' if len(adjs) else ''}" + f"{', '.join(adjs)}/{num_bairro}, {num_bairros[num_bairro]}\n")

    destino.writelines(escrever)




os.chdir(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia RJ")

adjacencias = open("./txt/adjacencias.txt", "r", encoding="utf-8")
nomes = open(r"./txt/bairros apenas.txt", "r", encoding="utf-8")
destino = open(r"./txt/arquivo_final.txt", "w", encoding="utf-8")

arquivotxt(adjacencias, nomes, destino)


#g = Grafo(r"./txt/arquivo_final.txt")
#instancia_grafo = g.GerarGrafo()