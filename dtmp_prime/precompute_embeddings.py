#!/usr/bin/env python3
"""
Pré-computa os embeddings DNABERT de todos os registros do Table-S5, UMA vez.
Salva num .npz indexado, para o treino carregar sem passar pelo BERT a cada época.
"""
import numpy as np
import pandas as pd
import torch
from . import paths
from .dnabert_embed import DNABERTEmbedder

def precompute(batch_size=64, out_path=None):
    out_path = paths.Path(out_path) if out_path else paths.DNABERT_EMBEDDINGS
    paths.ensure_dir(out_path.parent)

    df = paths.load_table(paths.TABLE_S5, sheet_name=paths.SHEET_TRAIN, header=1)
    wide_targets = df.iloc[:, 1].str.upper().tolist()   # coluna 1 = wide 47bp

    embedder = DNABERTEmbedder()
    all_emb = []
    for start in range(0, len(wide_targets), batch_size):
        chunk = wide_targets[start:start+batch_size]
        emb = embedder.embed_batch(chunk)   # (chunk, 768)
        all_emb.append(emb.numpy())
        if start % (batch_size*20) == 0:
            print(f"  {start}/{len(wide_targets)}")

    embeddings = np.concatenate(all_emb, axis=0)   # (N, 768)
    np.save(out_path, embeddings)
    print(f"Salvo: {out_path} shape {embeddings.shape}")

if __name__ == '__main__':
    precompute()