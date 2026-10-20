"""
Consulta de equipamentos pelo código de barras.

PROVISÓRIO: usa um banco SQLite de exemplo (criado por seed_equipment_db.py)
enquanto não se sabe qual é o banco real, nem em qual coluna o código fica.
Por isso a busca compara o código lido com serial, MAC e EAN ao mesmo tempo.

Para ligar no banco real, basta reimplementar find_by_codes() mantendo o
mesmo contrato: recebe uma lista de códigos e devolve {código: equipamento}
só para os códigos encontrados.
"""
import os
import sqlite3

DB_PATH = os.environ.get(
    "API_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "equipamentos.db"),
)

# Colunas onde o código lido pode estar. Ajustar quando o banco real for definido.
CODE_COLUMNS = ("serial", "mac", "ean")

SCHEMA = """
CREATE TABLE IF NOT EXISTS equipamentos (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    descricao TEXT NOT NULL,
    fabricante TEXT,
    serial    TEXT UNIQUE,
    mac       TEXT UNIQUE,
    ean       TEXT,
    status    TEXT
);
CREATE INDEX IF NOT EXISTS idx_equipamentos_serial ON equipamentos (serial);
CREATE INDEX IF NOT EXISTS idx_equipamentos_mac ON equipamentos (mac);
CREATE INDEX IF NOT EXISTS idx_equipamentos_ean ON equipamentos (ean);
"""


def normalize_code(code):
    return str(code).strip().upper()


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def find_by_codes(codes):
    """Devolve {código_normalizado: dict do equipamento} para os encontrados."""
    normalized = sorted({normalize_code(c) for c in codes if str(c).strip()})
    if not normalized:
        return {}

    placeholders = ",".join("?" for _ in normalized)
    where = " OR ".join(f"{column} IN ({placeholders})" for column in CODE_COLUMNS)
    query = f"SELECT * FROM equipamentos WHERE {where}"

    with _connect() as connection:
        rows = connection.execute(query, normalized * len(CODE_COLUMNS)).fetchall()

    found = {}
    for row in rows:
        equipment = dict(row)
        for column in CODE_COLUMNS:
            value = equipment.get(column)
            if value and normalize_code(value) in normalized:
                found[normalize_code(value)] = {
                    **equipment, "campo_encontrado": column,
                }
    return found


def is_available():
    try:
        with _connect() as connection:
            connection.execute("SELECT 1 FROM equipamentos LIMIT 1")
        return True
    except sqlite3.Error:
        return False
