# tokenizer_order3.py
_ID2CHAR   = list('ACGT')
CHAR2ID    = {c: i+1 for i, c in enumerate(_ID2CHAR)}
CHAR2ID_O2 = {f'{a}{b}': i*4+j+1
              for i,a in enumerate(_ID2CHAR) for j,b in enumerate(_ID2CHAR)}
CHAR2ID_O3 = {f'{a}{b}{c}': i*16+j*4+k+1
              for i,a in enumerate(_ID2CHAR)
              for j,b in enumerate(_ID2CHAR)
              for k,c in enumerate(_ID2CHAR)}

# ?? devem ser IDÊNTICOS aos do treino
MAX_TARGET = 47
MAX_PBS    = 17
MAX_RT     = 20

# Nomes das colunas descobertos na Etapa 1
PBS_SEQ_COL    = 'PBS_GC'
RTT_SEQ_COL    = 'RTT_GC'
TARGET_SEQ_COL = 'target_to_sgRNA'

def _tok1(seq, n):
    s = seq.upper()
    return [CHAR2ID[c] for c in s] + [0]*(n-len(s))

def _tok2(seq, n):
    s = seq.upper()
    return [CHAR2ID_O2[s[i:i+2]] for i in range(len(s)-1)] + [0]*(n-1-max(0,len(s)-1))

def _tok3(seq, n):
    s = seq.upper()
    return [CHAR2ID_O3[s[i:i+3]] for i in range(len(s)-2)] + [0]*(n-2-max(0,len(s)-2))

def tokenize_row(row):
    """Converte uma linha do self.X em 9 listas de inteiros."""
    target = row[TARGET_SEQ_COL]
    pbs    = row[PBS_SEQ_COL]
    rt     = row[RTT_SEQ_COL]
    return (
        _tok1(target, MAX_TARGET), _tok1(pbs, MAX_PBS), _tok1(rt, MAX_RT),
        _tok2(target, MAX_TARGET), _tok2(pbs, MAX_PBS), _tok2(rt, MAX_RT),
        _tok3(target, MAX_TARGET), _tok3(pbs, MAX_PBS), _tok3(rt, MAX_RT),
    )