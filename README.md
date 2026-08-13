# DTMP-Prime — Reconstrução, Análise e Validação

Reconstrução e extensão do [**DTMP-Prime**](https://github.com/alipanahiCRISPR/DTMP-Prime) (*Deep Transformer-based Model for Predicting Prime editing*),
ferramenta de desenho e pontuação de pegRNAs para *prime editing*, originalmente descrita por [*Alipanahi, Safari e Khanteymoori (2024)*](https://doi.org/10.1016/j.omtn.2024.102370).

Este repositório é um *fork* fundamentado em dois objetivos simultâneos:

1. Tornar o DTMP-Prime uma ferramenta de inferência utilizável por biólogos sem familiaridade
   computacional, a partir de uma interface gráfica de uso simples, intuitivo e rápido.
2. Reconstruir fielmente uma versão operacional do repositório original, aferindo-se a 
   contribuição isolada de cada componente arquitetural do trabalho original e 
   validando o modelo resultante a validação contra dados experimentais independentes.

## Motivação

O repositório original encontrava-se incompleto — desprovido de artefatos de
treinamento, de módulos referenciados no código e de arquivos de modelo — e
continha inconsistências entre a implementação e a arquitetura descrita no
artigo. A operação da ferramenta pressupunha, além disso, familiaridade com
ambientes Python, formatos de bioinformática e edição manual de configuração,
o que restringia seu uso pelo público de biólogos sem fundamentos suficientes
nos aspectos operacionais da bioinformática.

A reconstrução foi conduzida em cinco fases, cada uma isolando uma variável
arquitetural. O resultado central é que apenas as 14 *features* escalares
contribuem com a maior parte do sinal preditivo, de modo que o *encoding* 8×L 
e os *embeddings* do DNABERT (sejam *freezed* ou ajustados) degradam o desempenho 
quando acrescentados à formulação do modelo, porque a informação de sequência 
que codificam já é capturada pelos *k-mers*, e o sinal preditivo real reside 
nas variáveis termodinâmicas e enzimáticas.

O modelo de produção é, por isso, o da **Fase 2a**: o mais simples dos quatro
e com desempenho mais fiel em relação ao originalmente relatado no artigo.

## Documentação

A documentação técnico-científica do projeto, com a metodologia de cada fase, os
resultados de cada protocolo de validação e a validação externa contra dados
independentes, encontra-se disponível em: **[`DTMP-Prime.pdf`](docs/latex/DTMP-Prime.pdf)**

Leia-a antes de interpretar qualquer resultado produzido pela ferramenta. Em
particular, a Seção 4 (Validação Externa) delimita o domínio de aplicabilidade do modelo.

## Estrutura do Projeto

A estrutura final do projeto antes de ser executado localmente deve se apresentar sob a seguinte forma:

```
DTMP-Prime/
├── data/
│   ├── examples/                    # Arquivos de entrada de exemplo
│   │   └── example.fa               # Exemplo de input múltiplo via arquivo FASTA
│   ├── DTMP-Data-49300.xlsx         # Conjunto consolidado de 49.300 registros
│   ├── README.md
│   ├── Table-S1.xlsx                # Tabelas suplementares do artigo original
│   ├── Table-S2.xlsx
│   ├── Table-S3.xlsx
│   ├── Table-S4.xlsx
│   └── Table-S5.xlsx                # Conjunto de treino: sequências e features
│
├── dnabert/                         # DNABERT-6 (download via git-lfs)
│   ├── ...
│   └── README.md
│
├── docs/
│   ├── DTMP-Prime.pdf               # Documentação técnico-científica
│   └── README.md
│
├── dtmp_prime/                      # Pacote principal
│   ├── __init__.py
│   ├── compat.py                    # Aliases de módulos legados em pesos serializados
│   ├── dnabert_embed.py             # Interface com o DNABERT-6
│   ├── evaluate.py                  # Métricas e protocolos de avaliação
│   ├── features_inferencia.py       # Reconstrução das 14 features escalares
│   ├── legacy_pegsub.py             # Compatibilidade com o fallback XGBoost
│   ├── main.py                      # Ponto de entrada da CLI
│   ├── paths.py                     # Resolução centralizada de caminhos
│   ├── precompute_embeddings.py     # Pré-cálculo dos freezed embeddings
│   ├── read_data.py                 # Leitura da Tabela S5 e montagem dos tensores
│   ├── README.md
│   ├── target_mutation.py           # Busca de protospacers, construção de PBS e RTT
│   ├── tokenizer_order3.py          # Tokenização em k-mers de ordens 1, 2 e 3
│   ├── train_models.py              # Definição das arquiteturas e rotina de treino
│   ├── utilite.py                   # Utilidades de sequência e scores
│   └── wide_target.py               # Extração do wide target de 47 pb
│
├── embeddings/                      # Embeddings pré-calculados (download do Zenodo)
│   ├── dnabert_embeddings.npy
│   └── README.md
│
├── feature_norms/                   # Estatísticas de normalização
│   ├── feature_norm.npz             # Fase 2a — modelo de produção
│   ├── feature_norm_cv.npz          # Validação cruzada por linha
│   ├── feature_norm_fase3.npz       # Fase 3
│   ├── feature_norm_fase4.npz       # Fase 4
│   └── feature_norm_group.npz       # GroupKFold por alvo
│
├── models/                          # Pesos treinados não versionados (download do Zenodo)
│   ├── phase2a_model.pt
│   ├── phase3_model.pt
│   ├── phase4_model_no-DNABERT.pt
│   ├── phase4_model_w-DNABERT.pt
│   └── README.md
│
├── reconstruction/                  # Componentes reconstruídos do artigo
│   ├── validators/
│   │   └── __init__.py
│   ├── __init__.py
│   ├── alignment_73.py              # Alinhamento pegRNA-DNA de 73 posições
│   ├── encoding_8xL.py              # Encoding 8×L
│   └── README.md
│
├── training/
│   └── transformer_training.py      # Treino por fase arquitetural
│
├── web/                             # Interface gráfica
│   ├── __init__.py
│   ├── app.py                       # Servidor Flask e API de tarefas
│   ├── frontend.html                # Interface completa em arquivo único
│   └── README.md
│
├── config.yaml                      # Configuração dos parâmetros do pipeline
├── README.md
├── requirements-dev.txt             # Dependências de desenvolvimento e validação
└── requirements.txt                 # Dependências de inferência
```

Os diretórios `models/`, `dnabert/` e `embeddings/` são preenchidos durante a instalação das 
dependências desse projeto e não têm seu conteúdo versionado.

---

## Instalação

### Pré-requisitos

* Python 3.9 ou superior
* `git` e `git-lfs` (este último apenas para os pesos do DNABERT)
* GPU com CUDA é opcional para inferência e recomendada para treino

### 1. Clonar e criar o ambiente

```bash
git clone https://github.com/mateusjmd/DTMP-Prime.git
cd DTMP-Prime

conda create -n dtmp-prime python=3.9 -y
conda activate dtmp-prime
```

### 2. Instalar as dependências

Para efetuar apenas as inferências, basta instalar as dependências registradas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Para reproduzir integralmente o trabalho localmente e utilizá-lo para desenvolvimentos posteriores, é
necessário instalar as dependências existentes em `requirements-dev.txt`, que, além de instalar as
dependências basais para inferência, oferece suporte às demais bibliotecas usadas para validação e
teste de robustez dos *scripts*:

```bash
pip install -r requirements-dev.txt
```

### 3. Baixar os pesos dos modelos treinados e *embeddings* do DNABERT (Zenodo)

Os arquivos de pesos dos modelos treinados e os *embeddings* do DNABERT não são versionados neste repositório, sendo necessário 
baixá-los do depósito no [**Zenodo**](https://doi.org/10.5281/zenodo.21921579) e alocá-los no diretório `models/` e `embeddings/`, respectivamente.

Para fins de inferência, é necessário apenas o download dos pesos referentes ao modelo de produção, então alocando-o no diretório `models/`:

```bash
mv phase2a_model.pt models/
```

Para fins de reprodutibilidade integral do trabalho e desenvolvimentos posteriores, é necessário o download de todos os modelos, bem como dos *embeddings*:
```bash
# Pesos dos modelos treinados
mv phase2a_model.pt models/
mv phase3_model.pt models/
mv phase4_model_no-DNABERT.pt models/
mv phase4_model_w-DNABERT.pt models/

# Embeddings do DNABERT
mv dnabert_embeddings.npt embeddings/
```

> **📌 Observação:** Os arquivos de normalização (`feature_norms/*.npz`) já acompanham o
> repositório. O pareamento entre modelo e normalização é obrigatório e não é
> verificado em tempo de execução, de modo que um pareamento incorreto produz predições
> silenciosamente erradas.

| Modelo              | Normalização correspondente |
|---------------------|-----------------------------|
| `phase2a_model.pt`  | `feature_norm.npz`          |
| `phase3_model.pt`   | `feature_norm_fase3.npz`    |
| `phase4_model_*.pt` | `feature_norm_fase4.npz`    |

### 4. DNABERT (somente para desenvolvimento)

Esta etapa é necessária apenas para reproduzir as Fases 4 e 5, não sendo exigida
para a inferência.

> **📌 Observação:** É recomendável que esta dependência seja instalada em um ambiente
> Conda novo, diferenciando-o do ambiente `dtmp-prime`, a fim de evitar conflitos de
> compatibilidade da versão antiga necessária à biblioteca `transformers` com as demais.

```bash
conda create -n dnabert python=3.10 -y
pip install transformers==2.5.0

git clone https://huggingface.co/zhihan1996/DNA_bert_6 dnabert/DNA_bert_6
cd dnabert/DNA_bert_6 && git lfs pull && cd ../..
```

O `git lfs pull` é indispensável: sem ele, o clone traz apenas os ponteiros LFS,
e o carregamento do modelo falha ou — pior — devolve pesos aleatórios.

A versão 2.5.0 da `transformers` é anterior à interface atual da biblioteca: o
*tokenizer* não é diretamente invocável (usa-se `batch_encode_plus`) e a saída
do modelo é uma tupla, cujo primeiro elemento contém os estados ocultos.

### Verificação da instalação

```bash
python -c "import torch, RNA, Bio, skbio, yaml, flask; print('dependências ok')"
python -c "from genet.predict import SpCas9; print('genet ok')"
python -c "from dtmp_prime import paths; print(paths.ROOT)"
```

---

## Uso

### Interface gráfica

```bash
python web/app.py
```

Abra `http://localhost:5000`. A interface conduz por quatro etapas: seleção da
variante, seleção do pegRNA, seleção do sgRNA e resumo do desenho,
com as sequências exibidas por composição de bases.

**Fluxo típico:**

1. Na aba **Entrada**, escolha entre as abas *Sequências manuais* *Arquivo FASTA*. 
   
   **a)** Para o primeiro caso, e cole os pares selvagem (REF) e mutante (ALT), de modo que
   um painel de comparação exibe a diferença detectada, confira que ela
   corresponde à edição pretendida antes de submeter.

   **b)** Para o segundo caso, arraste e solte ou selecione do seu dispositivo um arquivo FASTA
   contendo os mesmo pares REF e ALT (a formatação do arquivo é descrita a seguir, na subseção "Entradas").

2. Na aba **Configuração**, ajuste as faixas de PBS e RTT se necessário (os
   valores padrão cobrem o domínio de treino do modelo).
3. Submeta clicando em **Executar** e acompanhe o registro de execução.
4. Ao final, percorra os candidatos ou baixe os arquivos brutos.

> **📌 Observação:** O valor padrão de 10 nt para o parâmetro `max_target_to_sgRNA`
> descarta *protospacers* cuja distância à edição o exceda. Se a ferramenta 
> não retornar o candidato esperado, aumente este valor antes de concluir que
> não há solução.

### Linha de comando

Para efetuar a inferência por linha de comando no terminal, execute o seguinte
comando no diretório raiz do projeto:

```bash
python -m dtmp_prime.main -f entrada.fa -c config.yaml -o resultados/
```

### Entradas

O arquivio FASTA usado como *input* deve conter pares de sequências, com 
sufixos `_ref` e `_alt`, podendo incluir múltiplos pares de sequências, 
relativas a diferentes mutações. A posição, o alelo de referência e o alternativo 
são inferidos por alinhamento.

```
>CFTR_F508del_ref
ACTTCACTTCTAATGGTGATTATGGG...
>CFTR_F508del_alt
ACTTCACTTCTAATGGTGATTATGGG...
```

> **📌 Observação:** A ordem `ref` e `alt` deve ser respeitada, de modo que 
> a primeira é a sequência de partida e a segunda, a desejada. Invertê-las 
> transforma uma correção em uma instalação da mutação, com geometria de
> RTT distinta.

### Saídas

O diretório informado como argumento da *flag* `-o` recebe:

| Arquivo               | Conteúdo                                                           |
|-----------------------|--------------------------------------------------------------------|
| `topX_pegRNAs.csv`    | Melhores candidatos por variante, ordenados por eficiência predita |
| `rawX_pegRNAs.csv.gz` | Todos os candidatos gerados, sem filtragem                         |
| `X_p_pegRNAs.csv.gz`  | Matriz de *features* numéricas com a predição                      |
| `summary.csv`         | Uma linha por variante: contagens de PE2, PE3, PE3b e dPAM         |

Colunas principais de `topX_pegRNAs.csv`: `sgRNA_seq` (espaçador), `PBS_seq`,
`RTT_seq`, `pegRNA_extension` (extensão 3′ = RTT + PBS), `ngRNA_name`,
`predicted_efficiency`.

### Escolha do modelo

Em `config.yaml`, mantenha exatamente um único bloco descomentado. O padrão é o de
produção, mas que pode ser substituído para fins de reprodução e/ou desenvolvimento
do projeto, utilizando-se os demais pesos dos modelos treinados, bem como o *encoding*
proposto no artigo:

```yaml
use_transformer: true
transformer_model: phase2a_model.pt
use_encoding: false
feature_norm_path: feature_norm.npz
```

---

## Domínio de aplicabilidade

A validação externa (documentação, Seção 4) estabelece limites que não são
verificados em tempo de execução:

* **PBS até 17 nt e RTT até 20 nt:** Fora dessa faixa a predição torna-se extrapolação,
  pois o *tokenizer* trunca o RTT em 20 nt, de modo que comprimentos maiores não são
  distinguidos pelo ramo sequencial do modelo.
* **Indiferença do modelo ao sgRNA:** As predições PE2 e PE3 são
  numericamente idênticas, sendo a classificação exibida uma anotação do candidato, 
  e não fator do *score*.
* **O modelo não identifica ótimos interiores de PBS:** Reproduz a direção do
  efeito no ramo ascendente da curva experimental, mas extrapola
  monotonicamente para além do ótimo.

O DTMP-Prime deve ser tratado como um instrumento de priorização de candidatos para
triagem experimental, não de seleção de candidato único.

## Referências

* **Trabalho original:**
Alipanahi, R.; Safari, L.; Khanteymoori, A. DTMP-Prime: A deep transformer-based
model for predicting prime editing efficiency and PegRNA activity.
*Molecular Therapy Nucleic Acids*, v. 35, n. 3, 2024.
[doi:10.1016/j.omtn.2024.102263](https://doi.org/10.1016/j.omtn.2024.102370).

* **Validação externa:**
Sousa, A. A. *et al.* Systematic optimization of prime editing for the efficient
functional correction of *CFTR* F508del in human airway epithelial cells.
*Nature Biomedical Engineering*, v. 9, p. 7–21, 2025.
[doi:10.1038/s41551-024-01233-3](https://doi.org/10.1038/s41551-024-01233-3).

* **_Prime editing:_**
Anzalone, A. V. *et al.* Search-and-replace genome editing without double-strand
breaks or donor DNA. *Nature*, v. 576, p. 149–157, 2019.
[doi:10.1038/s41586-019-1711-4](https://doi.org/10.1038/s41586-019-1711-4).

* **DNABERT:**
Ji, Y. *et al.* DNABERT: pre-trained Bidirectional Encoder Representations from
Transformers model for DNA-language in genome. *Bioinformatics*, v. 37, n. 15,
p. 2112–2120, 2021.
[doi:10.1093/bioinformatics/btab083](https://doi.org/10.1093/bioinformatics/btab083)

* **Pesos dos modelos treinados:**
Depósito Zenodo: [doi:10.5281/zenodo.21921579](https://doi.org/10.5281/zenodo.21921579).

## Licença e citação

Este repositório mantém a licença do trabalho original. Ao utilizá-lo, cite o
artigo de [*Alipanahi, Safari e Khanteymoori (2024)*](https://doi.org/10.1016/j.omtn.2024.102370) e, em caso de uso dos pesos ou das análises
aqui produzidas, o depósito Zenodo correspondente.