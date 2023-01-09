import xlrd

class Txt:            # modela os dados no arquivo final (que sera lido)
    def __init__(self, adjacencias, nomes, destino, populaçao):
        self.adjacencias = open(adjacencias, "r", encoding="utf-8")
        self.nomes = open(nomes, "r", encoding="utf-8")
        self.destino = open(destino, "w", encoding="utf-8")
        self.populaçao = xlrd.open_workbook(populaçao).sheet_by_index(1)

    def close(self):
        self.adjacencias.close()
        self.nomes.close()
        self.destino.close()

    def leitura_populaçao(self, nome_bairros):
        populaçao_dict = {}

        for rx in range(7, self.populaçao.nrows):                            # ler populaçao
            nome = str(self.populaçao.cell(rx,0).value).strip()

            if nome in nome_bairros.values():
                populaçao_dict[nome] = int(self.populaçao.cell(rx,1).value)
        
        return populaçao_dict

    def gerar_arquivo_destino(self):
        nome_bairros = {}

        for linha in self.nomes:
            linha = linha.strip()

            if linha != "":
                num, bairro = linha.split(" ", maxsplit=1)
                num = int(num)

                nome_bairros[num] = bairro
        
        cidades_ja_visitadas = set()
        linhas_texto = []

        populaçao = self.leitura_populaçao(nome_bairros)

        adjs = {}
        num_bairro_nome = {}

        for linha in self.adjacencias:                   # ler adjacencias e ontar string final
            linha = linha.strip()
            
            if linha != "":
                num_bairro = int(linha.split(" ", maxsplit=1)[0])
                num_bairro_nome[nome_bairros[num_bairro]] = num_bairro
                adjs[num_bairro] = {nome_bairros[int(x)] for x in linha.split(" ")[1:]}

                # adjs = [nome_bairros[int(x)] for x in linha.split(" ")[1:]]

                # linhas_texto.append(f"{nome_bairros[num_bairro]}" + f"{', ' if len(adjs) else ''}" +
                #     f"{', '.join(adjs)}/{num_bairro}, {populaçao[nome_bairros[num_bairro]]}, {1/(len(adjs) + 1)}\n")

        for num_bairro, adj in adjs.items():
            for bairro2 in adj:
                adjs[num_bairro_nome[bairro2]].add(nome_bairros[num_bairro])

        for num_bairro, adj in adjs.items():
            linhas_texto.append(f"{nome_bairros[num_bairro]}" + f"{', ' if len(adj) else ''}" +
                    f"{', '.join(adj)}/{num_bairro}, {populaçao[nome_bairros[num_bairro]]}, {1/(len(adj) + 1)}\n")

        self.destino.writelines(linhas_texto)
        self.close()
