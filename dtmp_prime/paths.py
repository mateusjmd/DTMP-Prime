"""
Caminhos canônicos do projeto DTMP-Prime.

Todos os caminhos derivam de ``__file__``, portanto o pipeline funciona a partir
de qualquer diretório de trabalho.

Todos os diretórios podem ser sobrescritos por variável de ambiente, o que
permite apontar para um scratch do ambiente local sem editar código:

    DTMP_DATA_DIR, DTMP_MODELS_DIR, DTMP_NORMS_DIR,
    DTMP_DNABERT_DIR, DTMP_EMBEDDINGS_DIR, DTMP_GENOME
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Raiz do repositório: <root>/dtmp_prime/paths.py -> <root>
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "dtmp_prime"


def _env_dir(var: str, default: Path) -> Path:
    """Diretório sobrescrevível por variável de ambiente."""
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# --------------------------------------------------------------------------
# Diretórios
# --------------------------------------------------------------------------
DATA = _env_dir("DTMP_DATA_DIR", ROOT / "data")
EXAMPLES = DATA / "examples"
MODELS = _env_dir("DTMP_MODELS_DIR", ROOT / "models")
NORMS = _env_dir("DTMP_NORMS_DIR", ROOT / "feature_norms")
DNABERT_DIR = _env_dir("DTMP_DNABERT_DIR", ROOT / "dnabert")
EMBEDDINGS = _env_dir("DTMP_EMBEDDINGS_DIR", ROOT / "embeddings")
RESULTS = _env_dir("DTMP_RESULTS_DIR", ROOT / "results")
WEB = ROOT / "web"
DOCS = ROOT / "docs"

DEFAULT_CONFIG = ROOT / "config.yaml"

# --------------------------------------------------------------------------
# Nomes de planilhas do Table-S5 / dataset principal
# --------------------------------------------------------------------------
SHEET_TRAIN = "Library 1 (HT-training, test)"
SHEET_POSITION_TYPE = "Library 2 (Position, Type)"

# Extensões aceitas para os datasets, em ordem de preferência.
_TABLE_EXTS = (".xlsx", ".xls", ".txt", ".tsv", ".csv")


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def ensure_dir(path) -> Path:
    """Cria o diretório (e pais) se não existir. Retorna o Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model(name: str) -> Path:
    """Caminho de um artefato de modelo dentro de ``models/``."""
    return MODELS / name


def norm(name: str) -> Path:
    """Caminho de um arquivo de normalização dentro de ``feature_norms/``."""
    return NORMS / name


def require(path, what: str, hint: str | None = None) -> Path:
    """
    Garante que um arquivo exista, com mensagem legível em vez de traceback.
    """
    p = Path(path)
    if not p.exists():
        msg = f"{what} não encontrado: {p}"
        if hint:
            msg += f"\n  -> {hint}"
        raise FileNotFoundError(msg)
    return p


def dataset(stem: str) -> Path:
    """
    Resolve um dataset de ``data/`` sem depender da extensão.
    """
    for ext in _TABLE_EXTS:
        candidate = DATA / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{stem}' não encontrado em {DATA} "
        f"(extensões testadas: {', '.join(_TABLE_EXTS)}).\n"
        f"  -> Baixe os dados conforme data/README.md, ou aponte DTMP_DATA_DIR "
        f"para o diretório que os contém."
    )


def load_table(stem: str, sheet_name: str | None = None, header: int = 1):
    """
    Carrega um dataset de ``data/`` despachando para o leitor correto.

    ``.xlsx``/``.xls`` -> ``pd.read_excel`` (com ``sheet_name``)
    ``.txt``/``.tsv``  -> ``pd.read_csv(sep='\\t')``
    ``.csv``           -> ``pd.read_csv``

    Substitui as chamadas ``pd.read_excel('DataSet/...xlsx', ...)`` espalhadas
    pelo projeto, que fixavam simultaneamente o diretório e o formato.
    """
    import pandas as pd

    path = dataset(stem)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name, header=header)
    sep = "," if suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, header=header)


# --------------------------------------------------------------------------
# Artefatos nomeados
# --------------------------------------------------------------------------
DATASET_MAIN = "DTMP-Data-49300"
TABLE_S5 = "Table-S5"

DNABERT_EMBEDDINGS = EMBEDDINGS / "dnabert_embeddings.npy"

# Modelos XGBoost do fallback (ramo use_transformer=False)
PE2_MODEL = MODELS / "PE2_model_final.pkl"
PE3_MODEL = MODELS / "PE3_model_final.pkl"


def genome_fasta(configured: str | None = None) -> Path:
    """
    Resolve o genoma de referência e valida o índice ``.fai``.
    """
    value = configured or os.environ.get("DTMP_GENOME")
    if not value:
        raise FileNotFoundError(
            "Genoma de referência não definido.\n"
            "  -> Defina 'genome_fasta' no config.yaml (caminho absoluto para o .fa) "
            "ou exporte a variável DTMP_GENOME."
        )
    fa = Path(value).expanduser()
    if not fa.is_absolute():
        fa = (ROOT / fa).resolve()
    require(fa, "Genoma de referência (FASTA)",
            "Verifique o caminho em 'genome_fasta' no config.yaml.")
    fai = fa.with_suffix(fa.suffix + ".fai")
    if not fai.exists():
        raise FileNotFoundError(
            f"Índice do genoma não encontrado: {fai}\n"
            f"  -> Gere-o com: samtools faidx {fa}\n"
            f"     (o bedtools getfasta usado pelo pipeline exige esse índice)"
        )
    return fa
