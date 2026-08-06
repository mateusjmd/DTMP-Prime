"""
train_transformer.py — Treino do TransformerEncoderDecoderModelOrder3
Dependências: torch, pandas, numpy  (NENHUMA dependência DNABERT/transformers)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # raiz do repositório
import torch

# Importa diretamente do notebook convertido (ver nota abaixo)
from dtmp_prime.train_models import TransformerEncoderDecoderModelOrder3, train_and_test_transformer_order3, save_model
import dtmp_prime.read_data as rnd
import dtmp_prime.evaluate as evaluate_model
from dtmp_prime import paths

# -- 1. Carregar e tokenizar os dados --------------------------------------
# ADAPTAR: apontar para seu dataset real e mapear colunas
df_order3 = rnd.read_data_of_for_transformer_order3(
    max_len_Target=47, MAX_PBS=17, MAX_RT=20
)

# -- 2. Split treino/teste --------------------------------------------------
from sklearn.model_selection import train_test_split
X = df_order3.iloc[:, :-1]      # Target, PBS, RT (+ _o2, _o3)
y = df_order3['Efficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# -- 3. Hiperparâmetros (valores usados no repositório) ---------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyperparameters = {
    'embedding_size':     512,
    'hidden_size':        [2048, 2048, 2048],
    'hidden_size_fully':  [256, 64],
    'output_size':        1,
    'nhead':              8,
    'num_encoder_layers': [6, 6, 6],
    'drop_out':           0.1,
    # A chave 'other_size' aparecia duas vezes neste dicionário (0 aqui e 17 no
    # final): o Python mantinha silenciosamente a última. Mantido o valor 17.
    'other_size':         17,
    'lr':                 1e-4,
    'weight_decay':       1e-5,
    'epoch_num':          100,
    'batch_size':         128,
    'best_epoch':         True,
    'transfer':           False,
    'freezing':           False,
    'device':             device,
}

# -- 4. Treinar -----------------------------------------------------------
transformer = train_and_test_transformer_order3(
    X_train, X_test, y_train, y_test, hyperparameters
)

# -- 5. Salvar ------------------------------------------------------------
save_model(transformer, model_dir=paths.MODELS, model_name='pegRNA_Transformer.pt')
print(f"Modelo salvo em {paths.model('pegRNA_Transformer.pt')}")