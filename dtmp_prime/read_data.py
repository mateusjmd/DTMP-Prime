# Read New Date

#!/usr/bin/env python3

"""
author: Feng Liu
"""

import numpy as np
import pandas as pd
import re

from . import paths


one_hot_encoding = {'A': (1, 0, 0, 0),
                    'C': (0, 1, 0, 0),
                    'G': (0, 0, 1, 0),
                    'T': (0, 0, 0, 1)}


def complement_seq(seq):
    """get the complementary sequence of the input sequence
    Args:
        seq: the input sequence
    Returns:
        complementary sequence
    """

    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return "".join([complement_map[s.upper()] for s in list(seq)])


def reverse_seq(seq):
    """get the reversed sequence of the input sequence
    Args:
        seq: the input sequence
    Returns:
        reversed sequence
    """

    return "".join(reversed(list(seq)))


def read_data_for_sl():
    """obtain the data in NBT for various types of Shallow Learning methods"""

    #data = pd.read_excel('../Supplementary Table 4.xlsx', sheet_name='Library 1 (HT-training, test)', header=1)
    data = paths.load_table(paths.DATASET_MAIN, sheet_name=paths.SHEET_TRAIN, header=1)
    # data = df.loc[:, :]

    MAX_PBS = max(data["PBS length"])  # 17
    MAX_RT = max(data["RT length"])  # 20
    MAX_PBS_RT = max(data["PBS-RT length"])  # 37
    max_len_Target = 47
    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    data_x = []
    data_y = []
    for i, row in data.iterrows():
        x_PBS = []
        temp = row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper()
        # temp = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][:row["PBS length"]])).upper()  # reversed and complementary sequence
        for j in range(len(temp)):
            x_PBS += one_hot_encoding[temp[j]]
        x_PBS += (0, 0, 0, 0) * (MAX_PBS - len(temp))
        x_RT = []
        temp = row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper()
        # temp = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][row["PBS length"]:])).upper()   # reversed and complementary sequence
        for j in range(len(temp)):
            x_RT += one_hot_encoding[temp[j]]
        x_RT += (0, 0, 0, 0) * (MAX_RT - len(temp))
        x_Target = []
        temp = row.iloc[2].upper()
        for j in range(max_len_Target):
            x_Target += one_hot_encoding[temp[j]]
        x_other = list(row[list(range(5, 8)) + list(range(9, 26))])
        x = x_RT + x_PBS + x_Target + x_other
        data_x.append(x)
        data_y.append(row['Measured PE efficiency'] / 1)
        # data_y.append(row['Measured PE efficiency'] if row['Measured PE efficiency'] >= 0 else 0)

    return np.array(data_x), np.array(data_y)


def read_data_for_sl_position_and_type(flag='Position', sheet_name=None):   #flag= Position or Type
    """obtain the data in NBT for various types of Shallow Learning methods"""

    # NOTA (divergência herdada do repositório original): esta função sempre leu
    # a Library 1, apesar do nome. Comportamento preservado; passe
    # sheet_name=paths.SHEET_POSITION_TYPE para usar a Library 2.
    df = paths.load_table(paths.DATASET_MAIN, sheet_name=sheet_name or paths.SHEET_TRAIN, header=1)
    data = data.replace('na', 0)
    # data = df.loc[:, :]

    MAX_PBS = max(data["PBS length"])  # 13
    MAX_RT = max(data["RT length"])  # 24
    MAX_PBS_RT = max(data["PBS-RT length"])  # 37
    max_len_Target = 47
    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    data_x = []
    data_y = []
    for i, row in data.iterrows():
        if not re.match(f'{flag}' + r'-\w+', row[0], re.I):
            continue
        x_PBS = []
        temp = row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper()
        # temp = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][:row["PBS length"]])).upper()  # reversed and complementary sequence
        for j in range(len(temp)):
            x_PBS += one_hot_encoding[temp[j]]
        x_PBS += (0, 0, 0, 0) * (MAX_PBS - len(temp))
        x_RT = []
        temp = row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper()
        # temp = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][row["PBS length"]:])).upper()   # reversed and complementary sequence
        for j in range(len(temp)):
            x_RT += one_hot_encoding[temp[j]]
        x_RT += (0, 0, 0, 0) * (MAX_RT - len(temp))
        x_Target = []
        temp = row.iloc[2].upper()
        for j in range(max_len_Target):
            x_Target += one_hot_encoding[temp[j]]
        x_other = list(row[list(range(5, 25))])
        x = x_RT + x_PBS + x_Target + x_other
        data_x.append(x)
        data_y.append(row['Measured PE efficiency'] / 1)
        # data_y.append(row['Measured PE efficiency'] if row['Measured PE efficiency'] >= 0 else 0)

    return np.array(data_x), np.array(data_y)


def read_data_for_rnn():
    """obtain the data in NBT for GRU"""

    #df = pd.read_excel('../Supplementary Table 4.xlsx', sheet_name='Library 1 (HT-training, test)', header=1)
    df = paths.load_table(paths.DATASET_MAIN, sheet_name=paths.SHEET_TRAIN, header=1)
    # raw_data = df.iloc[:, [2, 4, 5, 26]]
    data = {'Target': [], 'RT': [], 'PBS': [], 'Other': [], 'Efficiency': []}

    MAX_PBS = max(df["PBS length"])  # 17
    MAX_RT = max(df["RT length"])  # 20
    MAX_PBS_RT = max(df["PBS-RT length"])  # 37
    max_len_Target = 47
    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    for i, row in df.iterrows():
        temp = [one_hot_encoding[s] for s in list(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())]
        # temp = [one_hot_encoding[s] for s in list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())))]  # reversed and complementary sequence of PBS
        for i in range(len(temp), MAX_PBS):
            temp.append((0, 0, 0, 0))
        data['PBS'].append(temp)
        temp = [one_hot_encoding[s] for s in list(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())]
        # temp = [one_hot_encoding[s] for s in list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())))]  # reversed and complementary sequence of RT
        for i in range(len(temp), MAX_RT):
            temp.append((0, 0, 0, 0))
        data['RT'].append(temp)
        data['Efficiency'].append(row['Measured PE efficiency'] / 1)
        data['Other'].append(tuple(row.iloc[list(range(5, 8)) + list(range(9, 26))]))
        data['Target'].append([one_hot_encoding[s] for s in list(row.iloc[2].upper())])

    return pd.DataFrame(data)


def read_data_for_rnn_position_and_type(flag='Position', sheet_name=None):
    """obtain the data in NBT for GRU"""

    # NOTA (divergência herdada do repositório original): apesar do nome e do
    # comentário abaixo, estas funções sempre leram a Library 1, não a Library 2.
    # O comportamento foi preservado para não alterar resultados já publicados;
    # passe sheet_name=paths.SHEET_POSITION_TYPE para usar a Library 2.
    #df = pd.read_excel('.../Supplementary Table 4.xlsx', sheet_name='Library 2 (Position, Type)', header=1)
    df = paths.load_table(paths.DATASET_MAIN, sheet_name=sheet_name or paths.SHEET_TRAIN, header=1)
    df = df.replace('na', 0)
    # raw_data = df.iloc[:, [2, 4, 5, 26]]
    data = {'Target': [], 'RT': [], 'PBS': [], 'Other': [], 'Efficiency': []}

    MAX_PBS = max(df["PBS length"])  # 17
    MAX_RT = max(df["RT length"])  # 20
    MAX_PBS_RT = max(df["PBS-RT length"])  # 37
    max_len_Target = 47
    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    for i, row in df.iterrows():
        if not re.match(f'{flag}' + r'-\w+', row[0], re.I):
            continue

        temp = [one_hot_encoding[s] for s in list(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())]
        # temp = [one_hot_encoding[s] for s in list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())))]  # reversed and complementary sequence of PBS
        for i in range(len(temp), MAX_PBS):
            temp.append((0, 0, 0, 0))
        data['PBS'].append(temp)
        temp = [one_hot_encoding[s] for s in list(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())]
        # temp = [one_hot_encoding[s] for s in list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())))]  # reversed and complementary sequence of RT
        for i in range(len(temp), MAX_RT):
            temp.append((0, 0, 0, 0))
        data['RT'].append(temp)
        data['Efficiency'].append(row['Measured PE efficiency'] / 1)
        data['Other'].append(tuple(row.iloc[list(range(5, 25))]))
        data['Target'].append([one_hot_encoding[s] for s in list(row.iloc[2].upper())])

    return pd.DataFrame(data)


#def read_data_of_for_transformer(max_len_Target=47, MAX_PBS=17, MAX_RT=20):
#    """obtain the data in NBT for transformer"""
#
#    #df = pd.read_excel('../Supplementary Table 4.xlsx', sheet_name='Library 1 (HT-training, test)', header=1)
#    df = paths.load_table(paths.DATASET_MAIN, sheet_name=paths.SHEET_TRAIN, header=1)
#    # raw_data = df.iloc[:, [2, 4, 5, 26]]
#    data = {'Target': [], 'RT': [], 'PBS': [], 'Efficiency': []}
#    # data = {'Target': [], 'RT': [], 'PBS': [], 'Other': [], 'Efficiency': []}
#
#    MAX_PBS = max(MAX_PBS, max(df["PBS length"]))   # 17
#    MAX_RT = max(MAX_RT, max(df["RT length"]))  # 20
#    MAX_PBS_RT = max(df["PBS-RT length"])  # 37
#
#    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')
#
#    id2char = list('ACGT')
#    char2id = {char: i+1 for i, char in enumerate(id2char)}
#
#    for i, row in df.iterrows():
#        temp = [char2id[s] for s in list(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())]
#        for j in range(len(temp), MAX_PBS):
#            # temp.insert(0, 0)
#            temp.append(0)
#        data['PBS'].append(temp)
#        temp = [char2id[s] for s in list(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())]
#        for j in range(len(temp), MAX_RT):
#            temp.append(0)
#        data['RT'].append(temp)
#        data['Efficiency'].append(row['Measured PE efficiency'] / 1)
#        # data['Other'].append(list(row.iloc[list(range(5, 8)) + list(range(9, 26))]))
#        data['Target'].append([char2id[s] for s in list(row.iloc[2].upper())])
#
#    return pd.DataFrame(data)

def read_data_transformer_split(max_len_Target=47, MAX_PBS=17, MAX_RT=20):
    df = paths.load_table(paths.TABLE_S5, sheet_name=paths.SHEET_TRAIN, header=1)

    split_col  = df.columns[0]   # 'Datat set name'
    wide_col   = df.columns[1]
    ext_col    = df.columns[3]
    pbs_len_c  = df.columns[4]
    rtt_len_c  = df.columns[5]
    eff_col    = df.columns[25]  # 'Measured PE efficiency'

    df['Target'] = df[wide_col].str.upper()
    assert (df['Target'].str.len() == 47).all(), "Wide target != 47bp"

    def split_ext(row):
        ext = str(row[ext_col]).upper()
        rtt_len = int(row[rtt_len_c]); pbs_len = int(row[pbs_len_c])
        return pd.Series({'RTT': ext[:rtt_len], 'PBS': ext[rtt_len:rtt_len+pbs_len]})
    df[['RTT','PBS']] = df.apply(split_ext, axis=1)
    df['Efficiency'] = df[eff_col].astype(float)

    train = df[df[split_col] == 'HT-Training'][['Target','PBS','RTT','Efficiency']]
    test  = df[df[split_col] == 'HT-Test'][['Target','PBS','RTT','Efficiency']]
    return train.reset_index(drop=True), test.reset_index(drop=True)

#FEATURE_COLS_IDX = list(range(8, 25))   # as 17 features

#def read_data_transformer_features(save_norm_path='feature_norm.npz'):
#    import numpy as np
#    df = pd.read_excel('Table-S5.xlsx',
#                       sheet_name='Library 1 (HT-training, test)', header=1)
#
#    split_col = df.columns[0]
#    wide_col  = df.columns[1]
#    ext_col   = df.columns[3]
#    pbs_len_c = df.columns[4]
#    rtt_len_c = df.columns[5]
#    eff_col   = df.columns[25]
#    feat_cols = [df.columns[i] for i in FEATURE_COLS_IDX]
#
#    df['Target'] = df[wide_col].str.upper()
#    assert (df['Target'].str.len() == 47).all()
#
#    def split_ext(row):
#        ext = str(row[ext_col]).upper()
#        rl = int(row[rtt_len_c]); pl = int(row[pbs_len_c])
#        return pd.Series({'RTT': ext[:rl], 'PBS': ext[rl:rl+pl]})
#    df[['RTT','PBS']] = df.apply(split_ext, axis=1)
#    df['Efficiency'] = df[eff_col].astype(float)
#
#    train = df[df[split_col] == 'HT-Training'].reset_index(drop=True)
#    test  = df[df[split_col] == 'HT-Test'].reset_index(drop=True)
#
#    # -- Normalização: stats Só do treino --------------------------
#    feat_mean = train[feat_cols].mean().values
#    feat_std  = train[feat_cols].std().values
#    feat_std[feat_std == 0] = 1.0   # evita divisóo por zero
#
#    # salva para reuso na inferência é CRÍTICO
#    np.savez(save_norm_path, mean=feat_mean, std=feat_std,
#             cols=[c.split('\\n')[0].strip() for c in feat_cols])
#
#    def norm_feats(sub):
#        return ((sub[feat_cols].values - feat_mean) / feat_std).astype('float32')
#
#    train_feats = norm_feats(train)
#    test_feats  = norm_feats(test)
#
#    return (train[['Target','PBS','RTT','Efficiency']], train_feats,
#               test[['Target','PBS','RTT','Efficiency']],  test_feats)

# Read_New_Date.py — nova função para Fase 2a
def read_data_transformer_order3_features(max_len_Target=47, MAX_PBS=17, MAX_RT=20,
                                          save_norm_path=None, dnabert_npy=None,
                                          dnabert_path=None, dnabert_finetune=False):
    """
    Carrega Table-S5 para treino do Transformer COM as 17 features escalares.
    Correções vs original:
      - Target usa iloc[1] (Wide 47bp), NÃO iloc[2] (Guide 20bp)  [Fase 1]
      - Coluna 'Other' com 17 features z-score (stats só do HT-Training)  [Fase 2]
      - Retorna treino/teste separados pelo split nativo  [Fase 5]
    """
    import numpy as np
    from reconstruction.alignment_73 import encode_candidate
    from .dnabert_embed import seq_to_kmers

    # Caminhos ancorados na raiz do repositório (ver dtmp_prime/paths.py).
    save_norm_path = paths.norm(save_norm_path or 'feature_norm.npz')
    paths.ensure_dir(save_norm_path.parent)
    dnabert_path = dnabert_path or paths.DNABERT_DIR

    df = paths.load_table(paths.TABLE_S5, sheet_name=paths.SHEET_TRAIN, header=1)
    # -- NOVO: cria o tokenizer uma vez, se for fine-tune --
    tokenizer = None
    if dnabert_finetune:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(str(dnabert_path))

    # -- Fase 4: embeddings pré-computados (OPCIONAL) --
    # Antes o np.load era incondicional, o que fazia as Fases 2 e 3 exigirem um
    # artefato da Fase 4 mesmo com o ramo DNABERT desligado — acoplamento que
    # contraria a medição isolada da contribuição de cada componente.
    dnabert_emb = None
    if dnabert_npy is not None:
        npy_path = paths.Path(dnabert_npy)
        if not npy_path.is_absolute():
            npy_path = paths.EMBEDDINGS / npy_path
        paths.require(npy_path, "Arquivo de embeddings DNABERT",
                      "Gere-o com: python -m dtmp_prime.precompute_embeddings")
        dnabert_emb = np.load(npy_path)   # (N, 768), mesma ordem do df
        assert len(dnabert_emb) == len(df), \
            f"Embeddings ({len(dnabert_emb)}) != registros ({len(df)})"

    # IMPORTANTE: guarda a posição original ANTES de qualquer filtragem/split,
    # para indexar o .npy corretamente
    df = df.reset_index(drop=True)
    df['_orig_pos'] = range(len(df))   # posição no .npy

    split_col = df.columns[0]      # 'Datat set name'
    #FEAT_IDX  = list(range(8, 25)) # as 17 features
    FEAT_IDX = [8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    feat_cols = [df.columns[i] for i in FEAT_IDX]

    MAX_PBS = max(MAX_PBS, max(df["PBS length"]))
    MAX_RT  = max(MAX_RT,  max(df["RT length"]))

    id2char = list('ACGT')
    char2id    = {c: i+1 for i, c in enumerate(id2char)}
    char2id_o2 = {f'{a}{b}': i*4 + j + 1
                  for i,a in enumerate(id2char) for j,b in enumerate(id2char)}
    char2id_o3 = {f'{a}{b}{c}': i*16 + j*4 + k + 1
                  for i,a in enumerate(id2char) for j,b in enumerate(id2char)
                  for k,c in enumerate(id2char)}

    # -- Normalização: média/desvio Só do HT-Training ------------------
    train_mask = df[split_col] == 'HT-Training'
    feat_mean = df.loc[train_mask, feat_cols].mean().values
    feat_std  = df.loc[train_mask, feat_cols].std().values
    feat_std[feat_std == 0] = 1.0
    np.savez(save_norm_path, mean=feat_mean, std=feat_std,
             cols=[c.split('\n')[0].strip() for c in feat_cols])

    def build_split(sub_df):
        data = {'Target': [], 'Target_o2': [], 'Target_o3': [],
                'RT': [], 'RT_o2': [], 'RT_o3': [],
                'PBS': [], 'PBS_o2': [], 'PBS_o3': [],
                'Other': [], 'Efficiency': [], 'Encoding': [], 'Dnabert': [], 'Dnabert_ids': [], 'Dnabert_mask': []}

        for _, row in sub_df.iterrows():
            ext = str(row["3' extension sequence of pegRNA"]).upper()
            pbs_len = int(row["PBS length"])
            # variáveis explícitas e persistentes:
            pbs = ext[:pbs_len]
            rtt = ext[pbs_len:]

            # -- PBS (usa 'pbs') --
            t = [char2id[pbs[j]] for j in range(len(pbs))]
            t += [0]*(MAX_PBS - len(t));                 data['PBS'].append(t)
            t = [char2id_o2[pbs[j:j+2]] for j in range(len(pbs)-1)]
            t += [0]*(MAX_PBS-1 - len(t));               data['PBS_o2'].append(t)
            t = [char2id_o3[pbs[j:j+3]] for j in range(len(pbs)-2)]
            t += [0]*(MAX_PBS-2 - len(t));               data['PBS_o3'].append(t)

            # -- RT (usa 'rtt') --
            t = [char2id[rtt[j]] for j in range(len(rtt))]
            t += [0]*(MAX_RT - len(t));                  data['RT'].append(t)
            t = [char2id_o2[rtt[j:j+2]] for j in range(len(rtt)-1)]
            t += [0]*(MAX_RT-1 - len(t));                data['RT_o2'].append(t)
            t = [char2id_o3[rtt[j:j+3]] for j in range(len(rtt)-2)]
            t += [0]*(MAX_RT-2 - len(t));                data['RT_o3'].append(t)

            # -- Target: wide 47bp (Fase 1) --
            wide_target = row.iloc[1].upper()
            assert len(wide_target) == 47
            data['Target'].append([char2id[wide_target[j]] for j in range(len(wide_target))])
            data['Target_o2'].append([char2id_o2[wide_target[j:j+2]] for j in range(len(wide_target)-1)])
            data['Target_o3'].append([char2id_o3[wide_target[j:j+3]] for j in range(len(wide_target)-2)])

            # -- Other: 14 features (Fase 2) --
            raw_feats = row[feat_cols].values.astype(float)
            data['Other'].append(((raw_feats - feat_mean) / feat_std).tolist())

            # -- Encoding: matriz 8x73 (Fase 3) --  ? agora pbs, rtt e wide_target existem
            enc_matrix = encode_candidate(wide_target, pbs, rtt)
            data['Encoding'].append(enc_matrix.tolist())

            # -- Fase 4: embedding pré-computado desta linha (se houver) --
            if dnabert_emb is not None:
                orig_pos = int(row['_orig_pos'])
                data['Dnabert'].append(dnabert_emb[orig_pos].tolist())

            if dnabert_finetune:
                from .dnabert_embed import seq_to_kmers
                kmers = seq_to_kmers(wide_target)
                encoded = tokenizer.batch_encode_plus(
                    [kmers], add_special_tokens=True,
                    max_length=64, pad_to_max_length=True, return_tensors='pt')
                data['Dnabert_ids'].append(encoded['input_ids'][0].tolist())
                data['Dnabert_mask'].append(encoded['attention_mask'][0].tolist())

            data['Efficiency'].append(row['Measured PE efficiency'])

        # Descarta colunas opcionais não preenchidas (DNABERT desligado,
        # fine-tune desligado). Sem isto, pd.DataFrame(data) estoura com
        # "arrays must all be same length" — falha cujo sintoma fica longe da causa.
        n_rows = len(data['Efficiency'])
        data = {k: v for k, v in data.items() if len(v) == n_rows}
        return pd.DataFrame(data)

    train_df = build_split(df[train_mask])
    test_df  = build_split(df[df[split_col] == 'HT-Test'])
    print(f"Fase 2a — treino: {len(train_df)}, teste: {len(test_df)}, features: {len(feat_cols)}")
    return train_df, test_df


def read_data_for_transformer_position_and_type(flag='Position', max_len_Target=47, MAX_PBS=17, MAX_RT=20, sheet_name=None):   #flag= Position or Type
    """obtain the data in NBT for transformer"""

    # NOTA (divergência herdada do repositório original): apesar do nome e do
    # comentário abaixo, estas funções sempre leram a Library 1, não a Library 2.
    # O comportamento foi preservado para não alterar resultados já publicados;
    # passe sheet_name=paths.SHEET_POSITION_TYPE para usar a Library 2.
    #df = pd.read_excel('.../Supplementary Table 4.xlsx', sheet_name='Library 2 (Position, Type)', header=1)
    df = paths.load_table(paths.DATASET_MAIN, sheet_name=sheet_name or paths.SHEET_TRAIN, header=1)
    df = df.replace('na', 0)
    # df.iloc[:, 10] = df.iloc[:, 10].replace('na', 48)
    # df.iloc[:, 12] = df.iloc[:, 12].replace('na', -7.2)
    # raw_data = df.iloc[:, [2, 4, 5, 26]]
    data = {'Target': [], 'RT': [], 'PBS': [], 'Efficiency': []}
    # data = {'Target': [], 'RT': [], 'PBS': [], 'Other': [], 'Efficiency': []}

    MAX_PBS = max(MAX_PBS, max(df["PBS length"]))  # max(df["PBS length"]) == 13, but 17 better than 13
    MAX_RT = max(MAX_RT, max(df["RT length"]))  # 24, max(df["RT length"]) == 24
    MAX_PBS_RT = max(df["PBS-RT length"])  # 37

    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    id2char = list('ACGT')
    char2id = {char: i+1 for i, char in enumerate(id2char)}

    for i, row in df.iterrows():
        if not re.match(f'{flag}' + r'-\w+', row[0], re.I):
            continue
        # temp = [char2id[s] for s in list(row['3\' extension sequence of pegRNA'][:row["PBS length"]].upper())]
        temp = [char2id[s] for s in
                list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'].upper()))[:row["PBS length"]])]
        for j in range(len(temp), MAX_PBS):
            # temp.insert(0, 0)
            temp.append(0)
        data['PBS'].append(temp)
        # temp = [char2id[s] for s in list(row['3\' extension sequence of pegRNA'][row["PBS length"]:].upper())]
        temp = [char2id[s] for s in
                list(reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'].upper()))[row["PBS length"]:])]
        for j in range(len(temp), MAX_RT):
            temp.append(0)
        data['RT'].append(temp)
        data['Efficiency'].append(row['Measured PE efficiency'] / 1)
        # data['Other'].append(list(row.iloc[list(range(5, 25))]))
        data['Target'].append([char2id[s] for s in list(row.iloc[2].upper())])

    return pd.DataFrame(data)


def read_data_of_for_transformer_order3(max_len_Target=47, MAX_PBS=17, MAX_RT=20):
    df = paths.load_table(paths.TABLE_S5, sheet_name=paths.SHEET_TRAIN, header=1)

    # Nomes reais das colunas (por índice, pois têm quebras de linha nos headers)
    wide_col   = df.columns[1]   # Wide target sequence (47bp)
    ext_col    = df.columns[3]   # 3' extension of pegRNA (PBS+RTT)
    pbs_len_col = df.columns[4]  # PBS length
    rtt_len_col = df.columns[5]  # RTT length
    eff_col    = df.columns[-1]  # Measured PE efficiency (confirmar índice)

    df['Target'] = df[wide_col].str.upper()   # já é 47bp, uso direto
    assert (df['Target'].str.len() == 47).all(), \
        f"Wide target não tem 47bp em todos os registros! " \
        f"Encontrados comprimentos: {df['Target'].str.len().unique()}" 

    # Decompor a 3' extension em PBS e RTT usando os comprimentos tabelados
    # A 3' extension é [RTT][PBS] na orientação 5'?3' do pegRNA
    def split_ext(row):
        ext = str(row[ext_col]).upper()
        pbs_len = int(row[pbs_len_col])
        rtt_len = int(row[rtt_len_col])
        rtt = ext[:rtt_len]
        pbs = ext[rtt_len:rtt_len + pbs_len]
        return pd.Series({'RTT': rtt, 'PBS': pbs})

    df[['RTT','PBS']] = df.apply(split_ext, axis=1)
    df['Efficiency'] = df[eff_col].astype(float)

    return df[['Target','PBS','RTT','Efficiency']]


def read_data_for_transformer_position_and_type_order3(flag='Position', max_len_Target=47, MAX_PBS=17, MAX_RT=20, sheet_name=None):   #flag= Position or Type
    """obtain the data in NBT for transformer"""

    # NOTA (divergência herdada do repositório original): apesar do nome e do
    # comentário abaixo, estas funções sempre leram a Library 1, não a Library 2.
    # O comportamento foi preservado para não alterar resultados já publicados;
    # passe sheet_name=paths.SHEET_POSITION_TYPE para usar a Library 2.
    #df = pd.read_excel('.../Supplementary Table 4.xlsx', sheet_name='Library 2 (Position, Type)', header=1)
    df = paths.load_table(paths.DATASET_MAIN, sheet_name=sheet_name or paths.SHEET_TRAIN, header=1)
    df = df.replace('na', 0)
    # df.iloc[:, 10] = df.iloc[:, 10].replace('na', 48)
    # df.iloc[:, 12] = df.iloc[:, 12].replace('na', -7.2)
    # raw_data = df.iloc[:, [2, 4, 5, 26]]
    data = {'Target': [], 'Target_o2': [], 'Target_o3': [], 'RT': [], 'RT_o2': [], 'RT_o3': [],
            'PBS': [], 'PBS_o2': [], 'PBS_o3': [], 'Efficiency': []}
    # data = {'Target': [], 'RT': [], 'PBS': [], 'Efficiency': []}
    # data = {'Target': [], 'RT': [], 'PBS': [], 'Other': [], 'Efficiency': []}

    MAX_PBS = max(MAX_PBS, max(df["PBS length"]))  # max(df["PBS length"]) == 13, but 17 better than 13
    MAX_RT = max(MAX_RT, max(df["RT length"]))  # 24, max(df["RT length"]) == 24
    MAX_PBS_RT = max(df["PBS-RT length"])  # 37

    print(f'Maximum length of (Target, PBS, RT, PBS+RT): ({max_len_Target}, {MAX_PBS}, {MAX_RT}, {MAX_PBS_RT})')

    id2char = list('ACGT')
    char2id = {char: i + 1 for i, char in enumerate(id2char)}
    char2id_o2 = {f'{char}{char_j}': i * len(id2char) + j + 1
                  for i, char in enumerate(id2char) for j, char_j in enumerate(id2char)}
    char2id_o3 = {f'{char}{char_j}{char_k}': i * len(id2char) * len(id2char) + j * len(id2char) + k + 1
                  for i, char in enumerate(id2char) for j, char_j in enumerate(id2char) for k, char_k in
                  enumerate(id2char)}

    for i, row in df.iterrows():
        if not re.match(f'{flag}' + r'-\w+', row[0], re.I):
            continue

        seq = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'].upper()))[:row["PBS length"]]
        temp = [char2id[seq[j]] for j in range(0, len(seq))]
        for j in range(len(temp), MAX_PBS):
            # temp.insert(0, 0)
            temp.append(0)
        data['PBS'].append(temp)
        temp = [char2id_o2[seq[j:j + 2]] for j in range(0, len(seq) - 1)]
        for j in range(len(temp), MAX_PBS - 1):
            # temp.insert(0, 0)
            temp.append(0)
        data['PBS_o2'].append(temp)
        temp = [char2id_o3[seq[j:j + 3]] for j in range(0, len(seq) - 2)]
        for j in range(len(temp), MAX_PBS - 2):
            # temp.insert(0, 0)
            temp.append(0)
        data['PBS_o3'].append(temp)

        seq = reverse_seq(complement_seq(row['3\' extension sequence of pegRNA'].upper()))[row["PBS length"]:]
        temp = [char2id[seq[j]] for j in range(0, len(seq))]
        for j in range(len(temp), MAX_RT):
            temp.append(0)
        data['RT'].append(temp)
        temp = [char2id_o2[seq[j:j + 2]] for j in range(0, len(seq) - 1)]
        for j in range(len(temp), MAX_RT - 1):
            # temp.insert(0, 0)
            temp.append(0)
        data['RT_o2'].append(temp)
        temp = [char2id_o3[seq[j:j + 3]] for j in range(0, len(seq) - 2)]
        for j in range(len(temp), MAX_RT - 2):
            # temp.insert(0, 0)
            temp.append(0)
        data['RT_o3'].append(temp)

        seq = row.iloc[2].upper()
        data['Target'].append([char2id[seq[j]] for j in range(0, len(seq))])
        data['Target_o2'].append([char2id_o2[seq[j:j + 2]] for j in range(0, len(seq) - 1)])
        data['Target_o3'].append([char2id_o3[seq[j:j + 3]] for j in range(0, len(seq) - 2)])

        data['Efficiency'].append(row['Measured PE efficiency'] / 1)
        # data['Other'].append(list(row.iloc[list(range(5, 25))]))
        # data['Target'].append([char2id[s] for s in list(row.iloc[2].upper())])

    return pd.DataFrame(data)

