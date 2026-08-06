# -*- coding: latin-1 -*-

#!/usr/bin/env python3
"""
Fase 2b — Cálculo das 14 features na inferência.

Fórmulas validadas contra o Table-S5 (fidelidade 93-100%).
A ORDEM das features é idêntica à do treino (FEAT_IDX = [8,11,13..24]).
"""
import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction as gc
from RNA import fold_compound

# Scaffold padrão do sgRNA (mesmo do config.yaml do projeto)
SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
POLYT = "TTTTTT"


def _rc(s):
    return str(Seq(s).reverse_complement())


def _mfe(s):
    try:
        return round(fold_compound(s).mfe()[1], 1)
    except Exception:
        return 0.0


def _tm(seq, nn_table):
    try:
        return mt.Tm_NN(seq=Seq(seq), nn_table=nn_table)
    except Exception:
        return 0.0


def compute_14_features(pbs, rtt, guide, deepspcas9):
    """
    Retorna lista de 14 floats na ORDEM EXATA do treino.

    pbs, rtt : strings das sequências (do rawX: PBS_seq, RTT_seq)
    guide    : sgRNA_seq (GN19, com G artificial)
    deepspcas9 : valor já computado no rawX
    """
    pbs = pbs.upper(); rtt = rtt.upper(); guide = guide.upper()
    pbs_rtt = pbs + rtt

    feats = [
        # 0: Tm1 — PBS, híbrido DNA/RNA
        _tm(pbs, mt.R_DNA_NN1),
        # 1: Tm4 — RTT, híbrido DNA/RNA
        _tm(rtt, mt.R_DNA_NN1),
        # 2-4: GC counts
        pbs.count('G') + pbs.count('C'),
        rtt.count('G') + rtt.count('C'),
        pbs_rtt.count('G') + pbs_rtt.count('C'),
        # 5-7: GC contents (fração × 100)
        100 * gc(pbs),
        100 * gc(rtt),
        100 * gc(pbs_rtt),
        # 8: MFE_1 — pegRNA completo + polyT
        _mfe(guide + SCAFFOLD + _rc(pbs_rtt) + POLYT),
        # 9: MFE_2 — scaffold + extensão + polyT
        _mfe(SCAFFOLD + _rc(pbs_rtt) + POLYT),
        # 10: MFE_3 — extensão + polyT
        _mfe(_rc(pbs_rtt) + POLYT),
        # 11: MFE_4 — spacer (guide)
        _mfe(guide),
        # 12: MFE_5 — spacer + scaffold
        _mfe(guide + SCAFFOLD),
        # 13: DeepSpCas9 — já computado no rawX
        float(deepspcas9),
    ]
    return feats


class FeatureNormalizer:
    """Carrega as stats de normalização salvas no treino e aplica z-score."""
    def __init__(self, norm_path='feature_norm.npz'):
        data = np.load(norm_path, allow_pickle=True)
        self.mean = data['mean']
        self.std = data['std']
        assert len(self.mean) == 14, f"Esperava 14 features, norm tem {len(self.mean)}"

    def normalize(self, feats):
        arr = np.array(feats, dtype=float)
        return ((arr - self.mean) / self.std).tolist()