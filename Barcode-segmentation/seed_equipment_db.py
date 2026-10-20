"""
Cria o banco SQLite de EXEMPLO com equipamentos das fotos de teste.

Os códigos abaixo foram lidos das imagens em ../imagens/imagens. Alguns
equipamentos estão cadastrados só pelo serial, outros só pelo MAC, para
testar a busca em colunas diferentes. As demais fotos ficam de propósito
"não cadastradas", para testar o caso found=false.

Uso:
  python seed_equipment_db.py            (recria o banco do zero)
"""
import os

import equipment_repository as repo

EQUIPAMENTOS = [
    # (descricao, fabricante, serial, mac, ean, status)  — foto de origem
    ("ONT ZTE (imagem_um)", "ZTE", "ZTEYHAAR9911286", "B47CA61680C2", "7908842868815", "em estoque"),
    ("ONT ZTE (imagem_doze)", "ZTE", "ZTEYH86N6919192", None, None, "instalado"),
    ("ONU Huawei (imagem_trinta_e_nove)", "Huawei", "2150083734EGM1069398", "607ECDBBE8B0", None, "em estoque"),
    ("ONU Fiberhome (imagem_oito)", "Fiberhome", "FHTT92BC0558", None, None, "em manutenção"),
    # Cadastrado só pelo MAC: a busca precisa achar pelo MAC lido.
    ("Roteador (imagem_dois)", None, None, "80EA0B890CB0", None, "instalado"),
    ("Equipamento (imagem_quarenta)", None, "M08183215G02154062", None, None, "em estoque"),
]


def main():
    if os.path.exists(repo.DB_PATH):
        os.remove(repo.DB_PATH)
    with repo._connect() as connection:
        connection.executemany(
            "INSERT INTO equipamentos (descricao, fabricante, serial, mac, ean, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            EQUIPAMENTOS,
        )
    print(f"Banco de exemplo criado em {repo.DB_PATH} com {len(EQUIPAMENTOS)} equipamentos.")


if __name__ == "__main__":
    main()
