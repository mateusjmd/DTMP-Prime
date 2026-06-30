# tokenizer_order3.py
from Utilite import revcomp   # já disponível no seu ambiente

_B = list('ACGT')
CHAR2ID    = {c: i+1 for i, c in enumerate(_B)}
CHAR2ID_O2 = {f'{a}{b}': i*4+j+1
              for i,a in enumerate(_B) for j,b in enumerate(_B)}
CHAR2ID_O3 = {f'{a}{b}{c}': i*16+j*4+k+1
              for i,a in enumerate(_B)
              for j,b in enumerate(_B)
              for k,c in enumerate(_B)}

MAX_TARGET = 47   # janela alvo — confira contra seu xlsx de treino
MAX_PBS    = 17   # confira contra max_PBS_length do treino
MAX_RT     = 24   # confira contra max_RTT_length do treino
#MAX_RT     = 20   # confira contra max_RTT_length do treino

def _tok1(seq, n):
    s = seq.upper()
    ids = [CHAR2ID.get(c, 0) for c in s]
    return (ids + [0]*n)[:n]

def _tok2(seq, n):
    s = seq.upper()
    ids = [CHAR2ID_O2.get(s[i:i+2], 0) for i in range(len(s)-1)]
    m = n - 1
    return (ids + [0]*m)[:m]

def _tok3(seq, n):
    s = seq.upper()
    ids = [CHAR2ID_O3.get(s[i:i+3], 0) for i in range(len(s)-2)]
    m = n - 2
    return (ids + [0]*m)[:m]

def build_input_from_rawX(raw_row, target_fa=None):
    """
    raw_row : linha de self.rawX (tem PBS_seq, RTT_seq, sgRNA_seq, cut_position, strand)
    target_fa: self.target_fa do target_mutation pai (para extrair janela 47bp)
    """
    pbs = raw_row['PBS_seq']
    rt  = raw_row['RTT_seq']

    if target_fa is not None:
        c      = int(raw_row['cut_position'])
        strand = raw_row['strand']
        if strand == '+':
            target = target_fa[max(0, c-20) : max(0, c-20)+47]
        else:
            raw_t  = target_fa[max(0, c-3) : max(0, c-3)+47]
            target = revcomp(raw_t)
    else:
        # fallback: spacer 20bp, padded a 47bp via _tok1
        target = raw_row['sgRNA_seq']

    return (
        _tok1(target, MAX_TARGET), _tok1(pbs, MAX_PBS), _tok1(rt, MAX_RT),
        _tok2(target, MAX_TARGET), _tok2(pbs, MAX_PBS), _tok2(rt, MAX_RT),
        _tok3(target, MAX_TARGET), _tok3(pbs, MAX_PBS), _tok3(rt, MAX_RT),
    )