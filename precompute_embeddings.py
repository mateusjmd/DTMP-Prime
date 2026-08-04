# -*- coding: latin-1 -*-

#!/usr/bin/env python3
"""
Pré-computa os embeddings DNABERT de todos os registros do Table-S5, UMA vez.
Salva num .npz indexado, para o treino carregar sem passar pelo BERT a cada época.
"""
import numpy as np
import pandas as pd
import torch
from dnabert_embed import DNABERTEmbedder

def precompute(batch_size=64):
    df = pd.read_excel('DataSet/Table-S5.xlsx',
                       sheet_name='Library 1 (HT-training, test)', header=1)
    wide_targets = df.iloc[:, 1].str.upper().tolist()   # coluna 1 = wide 47bp

    embedder = DNABERTEmbedder()
    all_emb = []
    for start in range(0, len(wide_targets), batch_size):
        chunk = wide_targets[start:start+batch_size]
        emb = embedder.embed_batch(chunk)   # (chunk, 768)
        all_emb.append(emb.numpy())
        if start % (batch_size*20) == 0:
            print(f"  {start}/{len(wide_targets)}")

    embeddings = np.concatenate(all_emb, axis=0)   # (43149, 768)
    np.save('dnabert_embeddings.npy', embeddings)
    print(f"Salvo: dnabert_embeddings.npy shape {embeddings.shape}")

if __name__ == '__main__':
    precompute()