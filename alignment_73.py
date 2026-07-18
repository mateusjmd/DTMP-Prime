# -*- coding: latin-1 -*-

#!/usr/bin/env python3
"""
Fase 3 — Construção do alinhamento PegRNA-DNA de 73 posições (on-target).

LIMITE DE REPRODUTIBILIDADE: o paper afirma "lengths equal to 73" mas NÃO
especifica a composição das 73 posições nem como o alinhamento é construído.
A decomposição abaixo é INFERÊNCIA NOSSA, documentada como tal.

Decomposição adotada: [wide target 47bp] + [extensão 3' (RTT+PBS), até 26bp] = 73.
No on-target, peg = dna (match perfeito) -> linhas de mismatch/gap ficam
zeradas, refletindo que não há desalinhamento. O encoding captura então a
composição de sequência (linhas de nucleotídeo com -1) e o PAM.
"""
from Encoding_8xL import encode_pair_padded, GAP

WIDE_LEN = 47
EXT_LEN = 26          # 73 - 47
TOTAL_LEN = 73


def build_73_alignment(wide_target, pbs, rtt):
    wide = wide_target.upper()[:WIDE_LEN].ljust(WIDE_LEN, GAP)
    ext = (rtt + pbs).upper()[:EXT_LEN]
    # on-target: peg = dna (match perfeito) -> sem mismatches espúrios
    peg = wide + ext
    dna = wide + ext
    peg = peg[:TOTAL_LEN].ljust(TOTAL_LEN, GAP)
    dna = dna[:TOTAL_LEN].ljust(TOTAL_LEN, GAP)
    return peg, dna


def encode_candidate(wide_target, pbs, rtt):
    """Retorna a matriz 8x73 pronta para o modelo."""
    peg, dna = build_73_alignment(wide_target, pbs, rtt)
    return encode_pair_padded(peg, dna, pam_len=3, target_len=TOTAL_LEN)