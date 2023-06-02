import xlrd
import os


adjacencias = open(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia Florianopolis\regions_16.txt", "r", encoding="utf-8")

#nomes = open(nomes, "r", encoding="utf-8")
destino = open(r"C:\Users\rasen\Documents\Programação\IC Iniciação Científica\Instancia Florianopolis\arquivo_final.txt", "w", encoding="utf-8")
#tabela_populaçao = xlrd.open_workbook(populaçao).sheet_by_index(1) # index 1 - tabela 2010

nome_bairros_por_ID = {}

def gerar_arquivo_destino():
    # nome_bairros_por_ID = {}

    # for linha in nomes:                # salvar relação ID - bairros
    #     linha = linha.strip()

    #     if linha != "":
    #         num, bairro = linha.split(" ", maxsplit=1)
    #         num = int(num)

    #         nome_bairros_por_ID[num] = bairro
    
    # cidades_ja_visitadas = set()
    # linhasTextoFinal = []
    qtd_bairros, qtd_arestas = adjacencias.readline().split(" ") 
    populaçao_s = [int(pop) for pop in adjacencias.readline().split(" ")]
    população_i = [int(pop) for pop in adjacencias.readline().split(" ")]

    print(populaçao_s, população_i)
    grupos = [f"Grupo {i}" for i in range(1, 17)]
    adjs = [[] for i in range(0,16)]
    print(adjs)
    for linha in adjacencias:                   # ler adjacencias e montar string final
        linha = linha.strip()
        
        if linha != "":
            v, u, beta = linha.split(" ")

            v, u = int(v), int(u)
            beta = float(beta)

            adjs[v].append([u, beta])

            #linhasTextoFinal.append(f"{nome_bairro}" + f"{', ' if len(adjs) else ''}" +
                #f"{', '.join(adjs)}/{num_bairro}, {populaçao[nome_bairro]}, {1/(len(adjs) + 1)}\n")
    
    for grupo, adj in enumerate(adjs):
        for v in adj:
            v[0] = grupos[v[0]]
        #adjs[grupo][0] = grupos[adj[0]]
        destino.write(f"Grupo {grupo + 1}, ")
        destino.write(f"{', '.join([v[0] for v in adj])}/{grupo+1}, {populaçao_s[grupo]}, {população_i[grupo]}, {', '.join([str(v[1]) for v in adj])}\n")

    print(adjs)
    
    #destino.writelines(linhasTextoFinal)


    # for num_bairro, adj in adjs.items():      verifica a consistencia das adjacencias
    #     for bairro2 in adj:                   aresta xy e yx devem existir
    #         if bairro2 not in adj:
    #             #print("not")
    #             adjs[num_bairro_nome[bairro2]].add(nome_bairros_por_ID[num_bairro])

gerar_arquivo_destino()
# arquivo_adjacencias = "./txts/normal (real)/adjacencias.txt"
# arquivo_destino = "./txts/normal (real)/arquivo_final.txt"
# arquivo_ID_nomes = "./txts/relaçao ID - bairros.txt"
# tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"

# t = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_destino, tabela_populaçao)
# t.somar_adjacencias_1()