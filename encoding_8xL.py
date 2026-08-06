# -*- coding: latin-1 -*-

#!/usr/bin/env python3
"""
Fase 3 — Encoding 8xL conforme o paper DTMP-Prime, validado 100% contra a Figura 6B.

Matriz 8 x L por par PegRNA-DNA alinhado. Linhas (ordem A-T-G-C do paper):
  [0] A   [1] T   [2] G   [3] C
  [4] DNA    — presença de nucleotídeo na fita DNA
  [5] PegRNA — presença de nucleotídeo na fita PegRNA
  [6] Mis    — mismatch entre as duas
  [7] PAM    — 3 posições finais do PegRNA

Convenção de sinais (revelada pelo gabarito 6B, não pelo texto):
  match    ? linha do nucleotídeo = -1
  mismatch ? linha de AMBOS nucleotídeos = +1
  gap      ? linha do nt presente = +1

Regra de canto: mismatch numa posição de PAM suprime as linhas Mis e PegRNA
(inferida de 1 exemplo; pode precisar de refino para casos raros).
"""
import numpy as np

# Ordem EXATA do paper conforme Figura 6B: A, T, G, C
NT_ROW = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
GAP = '-'
SEQ_LEN = 73


def encode_pair(peg, dna, pam_len=3):
    """
    peg, dna : sequências ALINHADAS (mesmo comprimento, '-' nos gaps).
    pam_len  : nº de posições finais (não-gap) do PegRNA que são PAM.
    Retorna matriz numpy 8 x L.
    """
    assert len(peg) == len(dna), "Sequências alinhadas devem ter mesmo comprimento"
    L = len(peg)
    M = np.zeros((8, L), dtype=np.float32)

    nt_pos = [i for i in range(L) if peg[i].upper() in NT_ROW]
    pam_set = set(nt_pos[-pam_len:]) if len(nt_pos) >= pam_len else set(nt_pos)

    for x in range(L):
        d = dna[x].upper(); p = peg[x].upper()
        d_nt = d in NT_ROW; p_nt = p in NT_ROW
        is_pam = x in pam_set

        if d_nt and p_nt:
            if d == p:
                # match ? -1 na linha do nucleotídeo; DNA e PegRNA presentes
                M[NT_ROW[d], x] = -1
                M[4, x] = 1
                M[5, x] = 1
            else:
                # mismatch ? +1 em ambos os nucleotídeos
                M[NT_ROW[d], x] = 1
                M[NT_ROW[p], x] = 1
                M[4, x] = 1
                if is_pam:
                    # PAM suprime Mis e PegRNA nesta posição
                    M[5, x] = 0
                    M[6, x] = 0
                else:
                    M[5, x] = 1
                    M[6, x] = 1
        elif d_nt and not p_nt:
            # gap no PegRNA (nt só no DNA)
            M[NT_ROW[d], x] = 1
            M[4, x] = 1
        elif p_nt and not d_nt:
            # gap no DNA (nt só no PegRNA)
            M[NT_ROW[p], x] = 1
            M[5, x] = 1
        # gap em ambos ? tudo 0

    # Regra 8: PAM
    for x in pam_set:
        M[7, x] = 1

    return M


def encode_pair_padded(peg, dna, pam_len=3, target_len=SEQ_LEN):
    """Versão com padding/truncamento para comprimento fixo (73)."""
    M = encode_pair(peg, dna, pam_len)
    L = M.shape[1]
    if L < target_len:
        M = np.concatenate([M, np.zeros((8, target_len - L), dtype=np.float32)], axis=1)
    elif L > target_len:
        M = M[:, :target_len]
    return M