"""
Compatibilidade de desserialização com o layout de módulos anterior ao refactor.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

# --------------------------------------------------------------------------
# Mapa dos nomes de topo antigos -> módulos atuais
#
# Inclui as variantes de capitalização observadas no __pycache__ do repositório
# original (Utilite, Target_mutation, Train_models, Evaluate)
# --------------------------------------------------------------------------
LEGACY_ALIASES = {
    # Núcleo do pipeline
    "Train_models": "dtmp_prime.train_models",
    "train_models": "dtmp_prime.train_models",
    "Evaluate": "dtmp_prime.evaluate",
    "Utilite": "dtmp_prime.utilite",
    "Target_mutation": "dtmp_prime.target_mutation",
    "target_mutation": "dtmp_prime.target_mutation",
    "Read_data": "dtmp_prime.read_data",
    "read_data": "dtmp_prime.read_data",
    "tokenizer_order3": "dtmp_prime.tokenizer_order3",
    "features_inferencia": "dtmp_prime.features_inferencia",
    "wide_target": "dtmp_prime.wide_target",
    "dnabert_embed": "dtmp_prime.dnabert_embed",
    # Reconstrução
    "Encoding_8xL": "reconstruction.Encoding_8xL",
    "alignment_73": "reconstruction.alignment_73",
}


class _LegacyLoader(importlib.abc.Loader):
    """Devolve o módulo atual sob o nome antigo, sem reexecutar código."""

    def __init__(self, target_name: str):
        self.target_name = target_name

    def create_module(self, spec):
        return importlib.import_module(self.target_name)

    def exec_module(self, module):
        pass


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if path is not None:
            return None
        target_name = LEGACY_ALIASES.get(fullname)
        if target_name is None:
            return None
        return importlib.util.spec_from_loader(fullname, _LegacyLoader(target_name))


_finder = None


def install_legacy_module_aliases() -> None:
    """Instala o finder. Idempotente — pode ser chamada várias vezes."""
    global _finder
    if _finder is None:
        _finder = _LegacyFinder()
        sys.meta_path.insert(0, _finder)


def register_alias(legacy_name: str, current_name: str) -> None:
    """Registra um alias adicional em tempo de execução.

    Útil se aparecer um `.pt` referenciando um nome que não está no mapa::

        from dtmp_prime import compat
        compat.register_alias("Meu_Modulo_Antigo", "dtmp_prime.train_models")
    """
    LEGACY_ALIASES[legacy_name] = current_name
    install_legacy_module_aliases()


# --------------------------------------------------------------------------
# Carregamento tolerante
# --------------------------------------------------------------------------
def _guess_current_module(legacy_name: str) -> str | None:
    """Tenta casar um nome legado desconhecido com um módulo atual pelo basename."""
    candidates = {}
    for pkg in ("dtmp_prime", "reconstruction"):
        try:
            package = importlib.import_module(pkg)
        except ImportError:
            continue
        pkg_path = getattr(package, "__path__", [None])[0]
        if not pkg_path:
            continue
        import os
        for fname in os.listdir(pkg_path):
            if fname.endswith(".py") and not fname.startswith("__"):
                candidates[fname[:-3].lower()] = f"{pkg}.{fname[:-3]}"
    return candidates.get(legacy_name.lower())


def torch_load(path, map_location=None, max_retries: int = 8, **kwargs):
    """`torch.load` que resolve nomes de módulo legados automaticamente."""
    import torch

    install_legacy_module_aliases()
    kwargs.setdefault("weights_only", False)

    for _ in range(max_retries):
        try:
            return torch.load(path, map_location=map_location, **kwargs)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None)
            if not missing or missing in LEGACY_ALIASES:
                raise
            guess = _guess_current_module(missing)
            if guess is None:
                raise ModuleNotFoundError(
                    f"O arquivo {path} referencia o módulo '{missing}', que não existe "
                    f"na estrutura atual e não pôde ser resolvido automaticamente.\n"
                    f"  -> Registre o mapeamento manualmente:\n"
                    f"     from dtmp_prime import compat\n"
                    f"     compat.register_alias('{missing}', 'dtmp_prime.<modulo_atual>')",
                    name=missing,
                ) from exc
            print(f"[compat] '{missing}' resolvido para '{guess}' (modelo salvo antes do refactor).")
            register_alias(missing, guess)

    raise RuntimeError(f"Não foi possível resolver os módulos legados de {path} em {max_retries} tentativas.")


# --------------------------------------------------------------------------
# Migração definitiva
# --------------------------------------------------------------------------
def migrar(path, destino=None, map_location="cpu"):
    """Regrava um `.pt` legado com as referências de módulo atuais.

    O arquivo resultante carrega com `torch.load` puro, sem depender deste módulo.
    Por segurança, o original é preservado como `<nome>.pre-refactor.bak`.
    """
    import shutil
    from pathlib import Path

    import torch

    src = Path(path)
    dst = Path(destino) if destino else src
    model = torch_load(src, map_location=map_location)

    if dst == src:
        backup = src.with_suffix(src.suffix + ".pre-refactor.bak")
        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"[compat] backup do original em {backup}")

    torch.save(model, dst)
    print(f"[compat] migrado: {dst}  (classe agora referenciada como "
          f"{model.__class__.__module__}.{model.__class__.__qualname__})")
    return dst


def _cli():
    import argparse

    ap = argparse.ArgumentParser(
        description="Migra modelos .pt salvos antes do refactor para os caminhos de módulo atuais."
    )
    ap.add_argument("--migrar", nargs="+", metavar="ARQUIVO.pt", required=True)
    ap.add_argument("--destino", default=None,
                    help="arquivo de saída (só com um único arquivo de entrada); "
                         "por padrão sobrescreve, guardando backup .pre-refactor.bak")
    args = ap.parse_args()

    if args.destino and len(args.migrar) > 1:
        ap.error("--destino só pode ser usado com um único arquivo.")

    for p in args.migrar:
        migrar(p, args.destino)


if __name__ == "__main__":
    _cli()
