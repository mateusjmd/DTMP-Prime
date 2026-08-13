# tokenizer_order3.py  versão corrigida

from .wide_target import extract_wide_target

_B = list('ACGT')
CHAR2ID    = {c: i+1 for i, c in enumerate(_B)}
CHAR2ID_O2 = {f'{a}{b}': i*4+j+1
              for i,a in enumerate(_B) for j,b in enumerate(_B)}
CHAR2ID_O3 = {f'{a}{b}{c}': i*16+j*4+k+1
              for i,a in enumerate(_B)
              for j,b in enumerate(_B)
              for k,c in enumerate(_B)}

MAX_TARGET = 47
MAX_PBS    = 17   # deve ser igual ao max_PBS_length usado no treino
MAX_RT     = 20   # deve ser igual ao max_RTT_length usado no treino

def _tok1(seq, n):
    s = seq.upper()
    ids = [CHAR2ID.get(c, 0) for c in s]
    return (ids + [0] * n)[:n]

def _tok2(seq, n):
    s = seq.upper()
    ids = [CHAR2ID_O2.get(s[i:i+2], 0) for i in range(len(s) - 1)]
    m = n - 1
    return (ids + [0] * m)[:m]

def _tok3(seq, n):
    s = seq.upper()
    ids = [CHAR2ID_O3.get(s[i:i+3], 0) for i in range(len(s) - 2)]
    m = n - 2
    return (ids + [0] * m)[:m]

def tokenize_row(raw_row, target_fa):
    """
    raw_row: linha de self.rawX (tem PBS_seq, RTT_seq, sgRNA_seq)
    """
    pbs    = raw_row['PBS_seq']     # PBS em orientação pegRNA 5'?3'
    rt     = raw_row['RTT_seq']     # RTT em orientação pegRNA 5'?3'
    # Target = wide target 47bp reconstruída do genoma (não mais o spacer 20bp)
    target = extract_wide_target(target_fa,
                                 int(raw_row['cut_position']),
                                 raw_row['strand'])
    if target is None or len(target) != 47:
        return None   # candidato de borda  sinaliza para pular

    return (
        _tok1(target, MAX_TARGET), _tok1(pbs, MAX_PBS), _tok1(rt, MAX_RT),
        _tok2(target, MAX_TARGET), _tok2(pbs, MAX_PBS), _tok2(rt, MAX_RT),
        _tok3(target, MAX_TARGET), _tok3(pbs, MAX_PBS), _tok3(rt, MAX_RT),
    )