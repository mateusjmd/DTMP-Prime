# -*- coding: latin-1 -*-
#!/usr/bin/env python3
"""
Fase 4 — Extração de embeddings do DNABERT-6 (congelado, Abordagem 1).

API do transformers 2.5 (empacotado pelo DNABERT):
  - tokenização via batch_encode_plus (não tok(...))
  - saída do modelo é TUPLA: out[0] = last_hidden_state
  - pooling via token [CLS] (posição 0)

O DNABERT fica CONGELADO (eval + no_grad). Os embeddings são determinísticos,
então podem ser pré-computados uma vez (ver precompute_embeddings.py).
"""
import torch
from transformers import BertModel, BertTokenizer

DNABERT_PATH = '/home/mateus25032/work/DNABERT/DNA_bert_6'   # ajuste ao caminho real no Heisenberg
KMER = 6


def seq_to_kmers(seq, k=KMER):
    """Converte 'ATCGATCG' em '6-mers sobrepostos separados por espaço'."""
    seq = seq.upper()
    return ' '.join(seq[i:i+k] for i in range(len(seq) - k + 1))


class DNABERTEmbedder:
    def __init__(self, path=DNABERT_PATH, device=None, max_length=512):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertModel.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()                 # CONGELADO
        for p in self.model.parameters(): # garante que não treina
            p.requires_grad = False
        self.max_length = max_length

    @torch.no_grad()
    def embed_batch(self, sequences):
        """
        sequences: lista de strings de nucleotídeos (ex: wide targets 47bp)
        Retorna tensor (len(sequences), 768) — embedding [CLS] de cada uma.
        """
        kmer_texts = [seq_to_kmers(s) for s in sequences]

        # API transformers 2.5: batch_encode_plus
        encoded = self.tokenizer.batch_encode_plus(
            kmer_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        out = self.model(input_ids, attention_mask=attention_mask)
        last_hidden = out[0]                    # (batch, seq_len, 768) — TUPLA na 2.5
        cls = last_hidden[:, 0, :]              # token [CLS] ? (batch, 768)
        return cls.cpu()