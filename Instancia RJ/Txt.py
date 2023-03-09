import xlrd
import os


class Txt:            # modela os dados no arquivo final (que sera lido)
    def __init__(self, adjacencias, nomes, destino, populaçao):
        self.adjacencias = open(adjacencias, "r", encoding="utf-8")
        self.nomes = open(nomes, "r", encoding="utf-8")
        self.destino = open(destino, "w", encoding="utf-8")
        self.tabela_populaçao = xlrd.open_workbook(populaçao).sheet_by_index(1) # index 1 - tabela 2010

        self.nome_bairros_por_ID = {}

    def close(self):
        self.adjacencias.close()
        self.nomes.close()
        self.destino.close()

    def leitura_populaçao(self):
        populaçao = {}

        for rx in range(7, self.tabela_populaçao.nrows):                            # ler populaçao
            nome = str(self.tabela_populaçao.cell(rx,0).value).strip()

            if nome in self.nome_bairros_por_ID.values():
                populaçao[nome] = int(self.tabela_populaçao.cell(rx,1).value)
        
        return populaçao

    def gerar_arquivo_destino(self):
        self.nome_bairros_por_ID = {}

        for linha in self.nomes:                # salvar relação ID - bairros
            linha = linha.strip()

            if linha != "":
                num, bairro = linha.split(" ", maxsplit=1)
                num = int(num)

                self.nome_bairros_por_ID[num] = bairro
        
        cidades_ja_visitadas = set()
        linhasTextoFinal = []

        populaçao = self.leitura_populaçao()   # leitura da tabela de população
        
        for linha in self.adjacencias:                   # ler adjacencias e montar string final
            linha = linha.strip()
            
            if linha != "":
                num_bairro = int(linha.split(" ", maxsplit=1)[0])
                nome_bairro = self.nome_bairros_por_ID[num_bairro]

                adjs = {self.nome_bairros_por_ID[int(x)] for x in linha.split(" ")[1:]}

                linhasTextoFinal.append(f"{nome_bairro}" + f"{', ' if len(adjs) else ''}" +
                    f"{', '.join(adjs)}/{num_bairro}, {populaçao[nome_bairro]}, {1/(len(adjs) + 1)}\n")

        
        self.destino.writelines(linhasTextoFinal)
        self.close()


        # for num_bairro, adj in adjs.items():      verifica a consistencia das adjacencias
        #     for bairro2 in adj:                   aresta xy e yx devem existir
        #         if bairro2 not in adj:
        #             #print("not")
        #             adjs[num_bairro_nome[bairro2]].add(self.nome_bairros_por_ID[num_bairro])


    def gerar_arquivo_destino_wesley(self):
        self.nome_bairros_por_ID = {}

        for linha in self.nomes:
            linha = linha.strip()

            if linha != "":
                num, bairro = linha.split(" ", maxsplit=1)
                num = int(num)

                if num >= 13:
                    num -= 1

                self.nome_bairros_por_ID[num-1] = bairro
        
        N = len(self.nome_bairros_por_ID.keys()) # numero de vertices
        M = 0                                 # numero de arestas

        adjs = {}
        txt_adjs = ""
        
        for linha in self.adjacencias:                   # ler adjacencias e ontar string final
            linha = linha.strip()
            
            if linha != "":
                num_bairro = int(linha.split(" ", maxsplit=1)[0])
                #num_bairro_nome[self.nome_bairros_por_ID[num_bairro]] = num_bairro
                adjs[num_bairro] = [str(int(x)) for x in linha.split(" ")[1:]]
                for adj in adjs[num_bairro]:
                    txt_adjs += f"{num_bairro} {adj} {1/(len(adjs[num_bairro]) + 1)}\n"
        
        print(adjs)
        for num, adj in adjs.items():
            for num2 in adj:
                if not str(num) in adjs[int(num2)]:
                    print(num, num2)

        for num, adj in adjs.items():
            M += len(adj)

        self.destino.write(f"{N} {M}\n")            # primeira linha

        self.populaçao = self.leitura_populaçao()

        for i in range(len(self.nome_bairros_por_ID)):
            self.destino.write(f"{self.populaçao[self.nome_bairros_por_ID[i]]}" + f"{' ' if i != N-1 else ''}")

        self.destino.write("\n")

        for i in range(len(self.nome_bairros_por_ID)):    
            self.destino.write("0" + f"{' ' if i != N-1 else ''}")

        self.destino.write("\n")

        for i in range(len(self.nome_bairros_por_ID)):    
            self.destino.write("0" + f"{' ' if i != N-1 else ''}")

        self.destino.write("\n")

        self.destino.writelines(txt_adjs)
        a = open("./adja.txt", "w", encoding="utf-8")
        txt_a = ""
    

        print(adjs)
        for num, adj in adjs.items():
            txt_a += f"{num}" + f"{' ' if len(adj) else ''}" + " ".join(adj) + "\n"

        a.writelines(txt_a)

    def ajuste_arquivo_adjacencias(self):
        adjs = {}

        self.nome_bairros_por_ID = {}

        for linha in self.nomes:                # salvar relação ID - bairros
            linha = linha.strip()

            if linha != "":
                num, bairro = linha.split(" ", maxsplit=1)
                num = int(num)

                self.nome_bairros_por_ID[num] = bairro

        num_bairro_nome = {}

        for linha in self.adjacencias:                   # ler adjacencias e montar string final
            linha = linha.strip()
            
            if linha != "":
                num_bairro = int(linha.split(" ", maxsplit=1)[0])
                nome_bairro = self.nome_bairros_por_ID[num_bairro]

                num_bairro_nome[nome_bairro] = num_bairro
                adjs[num_bairro] = {str(x) for x in linha.split(" ")[1:]}


        for num_bairro, adj in adjs.items():      # verifica a consistencia das adjacencias
            for bairro2 in adj:                   # aresta xy e yx devem existir
                if str(num_bairro) not in adjs[int(bairro2)]:
                    print(num_bairro, bairro2, "not")
                    adjs[int(bairro2)].add(str(num_bairro))
        
        adj = open("adj novo.txt", "w", encoding="utf-8")

        for i in range(160):
            adj.write(f"{i+1} {' '.join(adjs[i+1])}\n")

# os.chdir(r"D:\Programação\IC Iniciação Científica\Instancia RJ")

# # "./txts/normal (real)/adjacencias.txt"
# # "./txts/otimizado/adjacencias.txt"
# arquivo_adjacencias = "./txts/normal (real)/adjacencias.txt"
# arquivo_destino = "./txts/normal (real)/arquivo_final.txt"
# arquivo_ID_nomes = "./txts/relaçao ID - bairros.txt"
# tabela_populaçao = "./tabelas/Tabela pop por idade e grupos de idade (2973).xls"

# t = Txt(arquivo_adjacencias, arquivo_ID_nomes, arquivo_destino, tabela_populaçao)
# t.gerar_arquivo_destino()