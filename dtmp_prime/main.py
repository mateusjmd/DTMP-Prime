#!/usr/bin/env python3
"""
DTMP-Prime — ponto de entrada do pipeline de desenho de pegRNAs.

Execute como módulo do pacote (garante que os imports relativos resolvam
independentemente do diretório de trabalho):

    python -m dtmp_prime.main -f entrada.vcf -c config.yaml -o resultados/
"""
import os
import sys
import warnings
import argparse
import datetime
import getpass

import yaml
import pandas as pd

from . import paths
from .utilite import *          # noqa: F401,F403  (fasta2vcf, vcf2fasta, etc.)
from .target_mutation import *  # noqa: F401,F403  (classe target_mutation)

warnings.filterwarnings("ignore")

# Final model
def print_parameters(myDict):
	myGroup = {}
	myGroup['Prime Editing'] = ['genome_fasta','scaffold','n_jobs','debug','PE2_model','PE3_model','extend_length']
	myGroup['PBS searching'] = ['min_PBS_length','max_PBS_length']
	myGroup['RTT searching'] = ['min_RTT_length','max_RTT_length','min_distance_RTT5','max_max_RTT_length']
	myGroup['sgRNA searching'] = ['gRNA_search_space','sgRNA_length','offset','PAM','max_target_to_sgRNA','max_max_target_to_sgRNA']
	myGroup['ngRNA searching'] = ['max_ngRNA_distance']
	for k in myGroup:
		print_group(myDict,myGroup[k],k)


def print_group(myDict,myList,group_title):
	print ("-------- Parameter Group: %s --------"%(group_title))
	for l in myList:
		print ("%s: %s"%(l,myDict[l]))


def _resolve_artifact(value, key):
	"""Resolve caminho de artefato: absoluto passa direto; relativo ancora na raiz.

	Sem isto, valores como 'feature_norm_fase3.npz' no config.yaml resolveriam
	contra o diretório de trabalho de quem chamou o pipeline (que, no caso da
	interface web, é diferente da raiz do projeto).
	"""
	if not value:
		return value
	from pathlib import Path
	p = Path(str(value)).expanduser()
	if p.is_absolute():
		return str(p)
	# tenta o diretório canônico do tipo de artefato, depois a raiz
	candidates = []
	if key == "feature_norm_path":
		candidates.append(paths.NORMS / p)
	elif key in ("transformer_model", "PE2_model", "PE3_model"):
		candidates.append(paths.MODELS / p)
	candidates.append(paths.ROOT / p)
	for c in candidates:
		if c.exists():
			return str(c)
	return str(candidates[0])


# Define default parameters
def get_parameters(config):
	# return dict
	parameters = {}
	# default parameters
	pre_defined_list = {}
	#------------ Prime Editing related-----------
	# Sem default embutido: o antigo "/home/yli11/.../hg19.fa" era herança do
	# EasyPrime, apontava para hg19 (contradizendo o hg38 do config.yaml) e não
	# existe fora da máquina original. Resolvido/validado por paths.genome_fasta().
	pre_defined_list["genome_fasta"] = None
	pre_defined_list["n_jobs"] = -1
	pre_defined_list["scaffold"] = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
	pre_defined_list["debug"] = 0
	pre_defined_list["extend_length"] = 1000 # extracting +- 1000bp center at target pos from the genome, in 99.9% cases, you don't need to change this. If change to less than 500, will trigger fasta input mode, may cause error.
	# introduction of pretrained models
	# You can use DEEPPRIME.
	pre_defined_list["PE2_model"] = str(paths.PE2_MODEL)
	pre_defined_list["PE3_model"] = str(paths.PE3_MODEL)

	#------------ PBS -----------
	pre_defined_list["min_PBS_length"] = 10
	pre_defined_list["max_PBS_length"] = 15

	#------------ RTT -----------
	pre_defined_list["min_RTT_length"] = 10
	pre_defined_list["max_RTT_length"] = 20 # if no candidate is found, this value will be increased by 5, max to max_max_RTT_length
	pre_defined_list["max_max_RTT_length"] = 50
	pre_defined_list["min_distance_RTT5"] = 5

	#------------ sgRNA -----------
	pre_defined_list["gRNA_search_space"] = 200
	pre_defined_list["sgRNA_length"] = 20
	pre_defined_list["offset"] = -3
	pre_defined_list["PAM"] = "NGG"
	pre_defined_list["max_target_to_sgRNA"] = 10 # if no candidate is found, this value will be increased by 5, max to max_max_target_to_sgRNA
	pre_defined_list["max_max_target_to_sgRNA"] = 30

	#------------ ngRNA ------------
	pre_defined_list["max_ngRNA_distance"] = 100 # if no candidate is found, this value will be increased by 20, max to max_max_ngRNA_distance
	pre_defined_list["max_max_ngRNA_distance"] = 200
	pre_defined_list["search_iteration"] = 1 # not affect anything

	# ---------- Transformer-based Model ----------
	pre_defined_list["use_transformer"] = False
	pre_defined_list["transformer_model"] = None

	# ---------- Produção ou Completude ---------
	pre_defined_list["use_encoding"] = False
	pre_defined_list["feature_norm_path"] = str(paths.norm("feature_norm.npz"))

	manifest_data = {}
	if config:
		try:
			with open(config, 'r', encoding='utf-8') as f:
				manifest_data = yaml.load(f, Loader=yaml.FullLoader) or {}
		except FileNotFoundError:
			print("Arquivo de configuração não encontrado: %s. Usando os padrões." % config)
		except yaml.YAMLError as e:
			print("Erro ao interpretar o YAML de configuração (%s): %s. Usando os padrões." % (config, e))
	else:
		print("Nenhum arquivo de configuração informado. Usando os padrões.")

	for p in pre_defined_list:
		value = manifest_data.get(p, None)
		parameters[p] = pre_defined_list[p] if value is None else value

	# Caminhos de artefatos podem vir relativos no config: resolve contra a raiz
	# do repositório, e não contra o diretório de trabalho corrente.
	for key in ("transformer_model", "feature_norm_path", "PE2_model", "PE3_model"):
		parameters[key] = _resolve_artifact(parameters.get(key), key)

	# O genoma NÃO é validado aqui: só o modo VCF o consulta. Validar cedo faria
	# uma execução perfeitamente válida em FASTA falhar por causa de um caminho de
	# genoma irrelevante para ela. A validação ocorre no ramo VCF de main().
	return parameters


# NOTA: o bloco _load() que existia aqui carregava "Utilite.py" e
# "Target_mutation.py" por caminho absoluto. Ele só era necessário quando um dos
# módulos se chamava "Target mutation.py" (com espaço, não importável). Com os
# nomes normalizados para minúsculas e o pacote dtmp_prime formado, os imports
# relativos do topo do arquivo bastam — e o _load quebrava em sistemas
# case-sensitive, pois os arquivos reais são utilite.py e target_mutation.py.

#Output: The output folder will contain:
#1. all pegRNA + ngRNA combination for the input vcf file
#2. top1 pegRNA + ngRNA combination for each variant
#3. visualization of the top1s [TODO]
#4. a summary file of each variant

# Receiving inputs as files
def my_args():
	mainParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="pegRNA design")
	username = getpass.getuser()

	mainParser.add_argument('-f','--input_file',  help="vcf or fasta",required=True)
	mainParser.add_argument('-c','--config',  help="A YAML file specifying parameters",default=None)

	mainParser.add_argument('-o','--output',  help="output dir",default="easy_prime_%s_%s_result_dir"%(username,str(datetime.date.today())))

	#add parameters above
	args = mainParser.parse_args()
	return args


def _looks_like_fasta(path):
	"""Detecta FASTA pela primeira linha não vazia (>) ou pela extensão."""
	if str(path).lower().endswith((".fa", ".fasta", ".fna", ".fas")):
		return True
	try:
		with open(path, "r", encoding="utf-8", errors="replace") as fh:
			for line in fh:
				line = line.strip()
				if not line:
					continue
				return line.startswith(">")
	except OSError:
		pass
	return False


# Calling the two steps that I have defined in the above section
def run_steps(t,**kwargs):
	t.init(**kwargs)
	t.search(**kwargs)
	t.predict(**kwargs)

	if kwargs.get("debug", 0):
		print("COLUNAS rawX:", t.rawX.columns.tolist())
		print("topX shape:", t.topX.shape)

	return [t.topX,t.rawX,t.X_p,t.found_PE3b,t.found_PE3,t.found_dPAM,t.found_PE2,t.N_sgRNA_found]


def main():

	# I receive the parameters as a file.
	# I call the above class for each target.
	args = my_args()
	# ------------------------- get parameters ----------------------------------------------

	parameters = get_parameters(args.config)
	print_parameters(parameters)

	# ---- Detecção explícita do formato de entrada -------------------------
	# Antes, a leitura era um try/except nu: qualquer erro no modo VCF (inclusive
	# genoma ausente ou bedtools fora do PATH) caía silenciosamente no modo FASTA
	# e depois num "Can't read ..." genérico, escondendo a causa real.
	input_path = args.input_file
	if not os.path.exists(input_path):
		sys.exit("Arquivo de entrada não encontrado: %s" % input_path)

	is_fasta = _looks_like_fasta(input_path)

	if is_fasta:
		print("Lendo arquivo FASTA: %s" % input_path)
		vcf = fasta2vcf(input_path)
		print(vcf)
	else:
		print("Lendo arquivo VCF: %s" % input_path)
		# Validação do genoma no ponto em que ele é de fato necessário.
		try:
			parameters["genome_fasta"] = str(paths.genome_fasta(parameters.get("genome_fasta")))
		except FileNotFoundError as e:
			sys.exit(
				"%s\n"
				"  -> Alternativamente, forneça a entrada em FASTA (par ref/alt), que\n"
				"     dispensa o genoma de referência." % e
			)
		try:
			vcf = pd.read_csv(input_path, comment="#", sep="\t", header=None)
			vcf[1] = vcf[1].astype(int)
			vcf = vcf.drop_duplicates(2)   # remove nomes duplicados
			vcf[3] = [x.upper() for x in vcf[3]]
			vcf[4] = [x.upper() for x in vcf[4]]
			vcf[5] = vcf2fasta(vcf, **parameters)
			vcf = vcf[list(range(6))]
		except Exception as e:
			sys.exit(
				"Falha ao processar %s como VCF: %s\n"
				"  -> Verifique se o arquivo tem as colunas CHROM/POS/ID/REF/ALT separadas\n"
				"     por tabulação e se o bedtools está disponível no PATH."
				% (input_path, e)
			)

	variant_list = vcf[2].tolist()


## for each target, create target mutation class
	my_targets = [target_mutation(*r) for i,r in vcf.iterrows()]



#find best pegRNAs
# backend can affect this parallization, if so, user show use n_jobs=1
	if parameters['n_jobs'] == 1:
		df_list = [run_steps(t,**parameters) for t in my_targets]
	else:
		from joblib import Parallel, delayed
		df_list = Parallel(n_jobs=parameters['n_jobs'],verbose=10)(delayed(run_steps)(t,**parameters) for t in my_targets)


	# save output
	# Either with these codes or the previously written codes, just print the line above
	os.makedirs(args.output, exist_ok=True)
	summary = pd.DataFrame([x[3:8] for x in df_list]).astype(int)
	summary.columns = ['found_PE3b','found_PE3','found_dPAM','found_PE2',"N_sgRNA_found"]
	summary.index = variant_list
	summary.to_csv("%s/summary.csv"%(args.output),index=True)

	df_top = pd.concat([x[0] for x in df_list])
	if df_top.shape[0]==0:
		print ("no pegRNA were found for the input file: %s"%(args.input_file))
		sys.exit()
	df_top = df_top.sort_values(by="predicted_efficiency", ascending=False)
	df_top.to_csv("%s/topX_pegRNAs.csv"%(args.output),index=False)

	df_all = pd.concat([x[1] for x in df_list])
	df_all = df_all.sort_values(by="predicted_efficiency", ascending=False)
	df_all.to_csv("%s/rawX_pegRNAs.csv.gz"%(args.output),index=False,compression="gzip")

	X_p = pd.concat([x[2] for x in df_list])
	X_p = X_p.sort_values(by="predicted_efficiency", ascending=False)
	X_p.to_csv("%s/X_p_pegRNAs.csv.gz"%(args.output),index=True,compression="gzip")




if __name__ == "__main__":
	main()