# -*- coding: latin-1 -*-

# wide_target.py
COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')

def revcomp(seq):
    return seq.upper().translate(COMPLEMENT)[::-1]

def extract_wide_target(target_fa, cut_position, strand):
    """
    Reconstrói a Wide target sequence de 47bp na convenção do Table-S5:
    [4bp upstream][20bp protospacer][3bp PAM][20bp downstream]

    Offsets calibrados empiricamente contra sgRNA_seq + PAM:
      strand + : proto_start = cut - 17
      strand - : proto_start = cut - 4  (janela espelhada + revcomp)

    Retorna str de 47bp, ou None se a janela sair dos limites de target_fa.
    """
    fa = target_fa.upper()

    if strand == '+':
        proto_start = cut_position - 17
        wide_start  = proto_start - 4
        wide_end    = wide_start + 47
        if wide_start < 0 or wide_end > len(fa):
            return None
        return fa[wide_start:wide_end]

    else:  # strand '-'
        proto_start = cut_position - 4
        # No -strand, a janela de 47bp no +strand é [proto_start-23 : proto_start+24],
        # depois revcomp para ler no -strand em 5'?3'.
        w_start = proto_start - 23
        w_end   = proto_start + 24     # 47bp
        if w_start < 0 or w_end > len(fa):
            return None
        return revcomp(fa[w_start:w_end])