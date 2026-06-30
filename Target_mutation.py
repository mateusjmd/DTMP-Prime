import sys
import subprocess
import pickle
import pandas as pd
import numpy as np

# Functions imported from Utilite.py at runtime via main.py's _load() mechanism.
# When running standalone, import them here:
try:
    from Utilite import (revcomp, get_gRNA_cut_site, is_gRNA_valid,
                         run_pam_finder, sub_fasta_single, get_opposite_strand,
                         distance_matrix, global_alignments, GC_content,
                         call_RNAplfold, is_dPAM, target_to_RTT5_feature,
                         get_DeepSpCas9_score, force_recommend_dPAM_PE3b)
except ImportError:
    pass  # injected by main.py _load() into caller namespace

# Target mutation
# A number of influential features on the accuracy of the editing output are defined here, and some are defined in the above cell and the sgRNA class

# (nick_to_pegRNA', 'target_to_pegRNA', 'target_to_RTT5','aln_ref_alt_mis', 'aln_ref_alt_del', 'aln_ref_alt_ins',  'PBS_GC', 'RTS_GC','PBS_length', 'RTS_length', 0, 1, 2, 3, 4, 5, 6, 7,
# (nick_to_pegRNA,target_to_pegRNA,'is_dPAM' ,'aln_ref_alt_mis', 'aln_ref_alt_del', 'aln_ref_alt_ins')define here

# (target_to_RTT5, RTS_GC, RTS_length,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ) defined  in sgRNA class / find_RTT
# (PBS_GC,PBS_length):  defined in sgRNA class /find_PBS



#solve the problem when user specification of ref, alt contain redundancy ATTTT-> ATTT, should be T -> "" G - > GC will  be "" C
def find_mutation_pos(pos,ref,alt):
	count=0
	for i in range(min(len(ref),len(alt))):
		x=ref[i]
		y=alt[i]
		if x != y:
			return pos,ref[i:],alt[i:]
		else:
			pos+=1
			count+=1
	return pos,ref[count:],alt[count:]


# ─────────────────────────────────────────────────────────────────────────────
# sgRNA class — missing from the repository; reconstructed from:
#   • run_sgRNA_search() call signature  (Target mutation.py)
#   • search()  constructor call         (Target mutation.py)
#   • feature_for_prediction list        (target_mutation.__init__)
#   • target_to_RTT5_feature()           (Utilite.py)
#   • get_gRNA_cut_site() convention     (Utilite.py)
#   • call_RNAplfold / is_dPAM / GC_content (Utilite.py)
#
# Coordinate conventions confirmed on sample data:
#   mutation_pos  : 1-indexed position within target_fa
#   cut_position  : for + strand = end-3 (0-indexed)
#                   for - strand = start+4 (≈1-indexed; use cut-1 for slicing)
# ─────────────────────────────────────────────────────────────────────────────

class sgRNA:
    def __init__(self,
                 chr, start, end, seq, sgRNA_name, strand, cut_position,
                 mutation_pos, mutation_ref, mutation_alt,
                 user_target_pos, user_ref, user_alt,
                 offset, target_to_sgRNA, variant_id,
                 dist_dict, opposite_strand_sgRNAs, all_sgRNA_df,
                 target_fa, scaffold_seq, PAM, DeepSpCas9):

        self.chr                   = chr
        self.start                 = start          # 0-indexed, BED
        self.end                   = end            # 0-indexed exclusive, BED
        self.seq                   = seq            # 20-bp spacer
        self.sgRNA_name            = sgRNA_name
        self.strand                = strand
        self.cut_position          = cut_position   # from get_gRNA_cut_site
        self.mutation_pos          = mutation_pos   # 1-indexed in target_fa
        self.mutation_ref          = mutation_ref
        self.mutation_alt          = mutation_alt
        self.user_target_pos       = user_target_pos
        self.user_ref              = user_ref
        self.user_alt              = user_alt
        self.offset                = offset
        self.target_to_sgRNA       = target_to_sgRNA
        self.variant_id            = variant_id
        self.dist_dict             = dist_dict
        self.opposite_strand_sgRNAs = opposite_strand_sgRNAs
        self.all_sgRNA_df          = all_sgRNA_df
        self.target_fa             = target_fa
        self.scaffold_seq          = scaffold_seq if scaffold_seq else \
            "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
        self.PAM                   = PAM
        self.DeepSpCas9            = DeepSpCas9

        # outputs filled by find_RTT / find_PBS / find_nick_gRNA / get_rawX_and_X
        self.rtt_records  = []   # list of dicts, one per RTT length
        self.pbs_records  = []   # list of dicts, one per PBS length
        self.ngRNA_records = []  # list of dicts, one per valid ngRNA
        self.no_ngRNA     = True
        self.rawX         = pd.DataFrame()
        self.X            = pd.DataFrame()

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _alt_seq(self):
        """Replace ref with alt in target_fa (mutation_pos is 1-indexed)."""
        m = self.mutation_pos - 1           # 0-indexed
        return (self.target_fa[:m]
                + self.mutation_alt
                + self.target_fa[m + len(self.mutation_ref):])

    def _cut_0idx(self):
        """
        Return 0-indexed position for Python slicing.
        + strand: cut_position is already 0-indexed (end - 3).
        - strand: cut_position is 1-indexed-like (start + 4); subtract 1.
        """
        if self.strand == '+':
            return self.cut_position
        else:
            return self.cut_position - 1

    def _seq_for_strand(self, seq, rng_start, rng_end, apply_revcomp=True):
        """Extract a slice and reverse-complement it (for pegRNA orientation)."""
        sub = seq[rng_start:rng_end]
        if apply_revcomp:
            return revcomp(sub)
        return sub

    # ─── step 1: find all RTT candidates ──────────────────────────────────────

    def find_RTT(self, min_RTT_length=10, max_RTT_length=20,
                 min_distance_RTT5=5, scaffold=None, **kwargs):
        """
        Build one RTT record per valid length.

        For + strand: RTT covers alt_seq[cut : cut+rtt_len]  (revcomped for pegRNA)
        For - strand: RTT covers alt_seq[cut_0-rtt_len : cut_0] (revcomped for pegRNA)

        The mutation must fall inside the covered region.
        min_distance_RTT5 enforces a minimum overhang past the mutation.
        """
        alt = self._alt_seq()
        c   = self._cut_0idx()
        m   = self.mutation_pos - 1          # 0-indexed mutation start
        alt_len = len(self.mutation_alt)

        sc_len = len(self.scaffold_seq)
        self.rtt_records = []

        for rtt_len in range(min_RTT_length, max_RTT_length + 1):

            if self.strand == '+':
                rng_s, rng_e = c, c + rtt_len
                # mutation must start inside the RTT region
                if m < rng_s or m + alt_len > rng_e:
                    continue
                # min_distance_RTT5: at least this many bases past the mutation
                if rng_e - (m + alt_len) < min_distance_RTT5:
                    continue
                rtt_genomic = alt[rng_s:rng_e]

            else:  # '-' strand
                rng_s, rng_e = c - rtt_len, c
                if rng_s < 0:
                    continue
                if m < rng_s or m + alt_len > rng_e:
                    continue
                if (m + alt_len - 1) - rng_s < min_distance_RTT5 - 1:
                    continue
                rtt_genomic = alt[rng_s:rng_e]

            if len(rtt_genomic) != rtt_len:
                continue

            rtt_pegRNA = revcomp(rtt_genomic)   # pegRNA sequence, 5'→3'

            # target_to_RTT5: bases from mutation to the 5' end of RTT in pegRNA
            if self.strand == '+':
                target_to_RTT5 = rtt_len - alt_len - (m - rng_s) + 1
            else:
                target_to_RTT5 = rtt_len - alt_len - (rng_e - 1 - m) + 1

            # RNA folding: scaffold + RTT (first 10 nt) determines pairing prob
            try:
                fold_seq   = self.scaffold_seq + rtt_pegRNA
                fold_vals  = call_RNAplfold(fold_seq, sc_len)
            except Exception:
                fold_vals = [0.0] * 10

            self.rtt_records.append({
                'rtt_seq'       : rtt_pegRNA,
                'rtt_genomic'   : rtt_genomic,
                'RTT_length'    : rtt_len,
                'RTT_GC'        : GC_content(rtt_pegRNA),
                'is_dPAM'       : is_dPAM(self.PAM, rtt_pegRNA, self.offset),
                'target_to_RTT5': target_to_RTT5,
                'fold_vals'     : fold_vals,
            })

    # ─── step 2: find all PBS candidates ──────────────────────────────────────

    def find_PBS(self, min_PBS_length=10, max_PBS_length=15, **kwargs):
        """
        For both strands the PBS hybridises to the 3' flap of the nicked strand,
        which sits DOWNSTREAM of the cut (+ strand) or UPSTREAM (- strand) in
        genomic (+) coordinates.

        PBS (pegRNA, 5'→3') = revcomp(ref_seq region that is the 3' flap)
        """
        ref = self.target_fa
        c   = self._cut_0idx()
        self.pbs_records = []

        for pbs_len in range(min_PBS_length, max_PBS_length + 1):

            if self.strand == '+':
                rng_s, rng_e = c, c + pbs_len
            else:
                rng_s, rng_e = c - pbs_len, c

            if rng_s < 0 or rng_e > len(ref):
                continue

            pbs_region = ref[rng_s:rng_e]
            if len(pbs_region) != pbs_len:
                continue

            pbs_pegRNA = revcomp(pbs_region)

            self.pbs_records.append({
                'pbs_seq'   : pbs_pegRNA,
                'PBS_length': pbs_len,
                'PBS_GC'    : GC_content(pbs_pegRNA),
            })

    # ─── step 3: find nick gRNA candidates ────────────────────────────────────

    def find_nick_gRNA(self, max_ngRNA_distance=100, **kwargs):
        """
        Search opposite-strand sgRNAs within max_ngRNA_distance of this cut.
        PE3b: mutation position falls within the ngRNA protospacer region.
        """
        if self.opposite_strand_sgRNAs.empty:
            self.no_ngRNA = True
            return

        my_cut = self.cut_position
        mut_0  = self.mutation_pos - 1
        self.ngRNA_records = []

        for _, row in self.opposite_strand_sgRNAs.iterrows():
            row = row.tolist()
            # row: [chr, start, end, seq, name, strand]
            ng_start, ng_end, ng_strand, ng_name = row[1], row[2], row[5], row[4]

            ng_cut = get_gRNA_cut_site(ng_start, ng_end, ng_strand, self.offset)
            dist   = abs(my_cut - ng_cut)

            if dist > max_ngRNA_distance:
                continue

            # PE3b: mutation falls within ngRNA protospacer window [ng_start, ng_end)
            is_pe3b = int(ng_start <= mut_0 < ng_end)

            self.ngRNA_records.append({
                'ng_name'              : ng_name,
                'ng_cut'               : ng_cut,
                'ng_start'             : ng_start,
                'ng_end'               : ng_end,
                'ng_strand'            : ng_strand,
                'sgRNA_distance_to_ngRNA': ng_cut - my_cut - 1,
                'is_PE3b'              : is_pe3b,
            })

        self.no_ngRNA = len(self.ngRNA_records) == 0

    # ─── step 4: combine into rawX and X DataFrames ───────────────────────────

    def get_rawX_and_X(self, debug=0, **kwargs):
        """
        Cross-product of RTT × PBS × ngRNA (or a null-ngRNA row for PE2).
        rawX : human-readable, all sequences and features.
        X    : numeric features matching feature_for_prediction.
        """
        if not self.rtt_records or not self.pbs_records:
            self.rawX = pd.DataFrame()
            self.X    = pd.DataFrame()
            return

        # PE2-only placeholder when no ngRNA found
        null_ng = {
            'ng_name'               : '0',
            'ng_cut'                : float('nan'),
            'ng_start'              : 0,
            'ng_end'                : 0,
            'ng_strand'             : '.',
            'sgRNA_distance_to_ngRNA': float('nan'),
            'is_PE3b'               : 0,
        }
        ng_list = self.ngRNA_records if self.ngRNA_records else [null_ng]

        rows_raw = []
        rows_X   = []

        for rtt in self.rtt_records:
            for pbs in self.pbs_records:
                for ng in ng_list:

                    # unique index: pegRNA_name + PBS_len + RTT_len + ngRNA
                    peg_ext  = rtt['rtt_seq'] + pbs['pbs_seq']
                    row_name = (f"{self.sgRNA_name}"
                                f"_PBS{pbs['PBS_length']}"
                                f"_RTT{rtt['RTT_length']}"
                                f"_ng{ng['ng_name']}")
                    if rtt['is_dPAM']:
                        row_name += '_dPAM'
                    if ng['is_PE3b']:
                        row_name += '_PE3b'

                    fold = rtt['fold_vals']

                    raw = {
                        'variant'                  : self.variant_id,
                        'sgRNA_name'               : self.sgRNA_name,
                        'sgRNA_seq'                : self.seq,
                        'strand'                   : self.strand,
                        'cut_position'             : self.cut_position,
                        'PBS_seq'                  : pbs['pbs_seq'],
                        'RTT_seq'                  : rtt['rtt_seq'],
                        'pegRNA_extension'         : peg_ext,
                        'PBS_length'               : pbs['PBS_length'],
                        'PBS_GC'                   : pbs['PBS_GC'],
                        'RTT_length'               : rtt['RTT_length'],
                        'RTT_GC'                   : rtt['RTT_GC'],
                        'is_dPAM'                  : rtt['is_dPAM'],
                        'is_PE3b'                  : ng['is_PE3b'],
                        'target_to_sgRNA'          : self.target_to_sgRNA,
                        'target_to_RTT5'           : rtt['target_to_RTT5'],
                        'DeepSpCas9'               : self.DeepSpCas9,
                        'sgRNA_distance_to_ngRNA'  : ng['sgRNA_distance_to_ngRNA'],
                        'ngRNA_name'               : ng['ng_name'],
                        **{str(i): fold[i] for i in range(10)},
                    }
                    raw_series       = pd.Series(raw, name=row_name)

                    x = {
                        **{str(i): fold[i] for i in range(10)},
                        'DeepSpCas9'               : self.DeepSpCas9,
                        'sgRNA_distance_to_ngRNA'  : ng['sgRNA_distance_to_ngRNA'],
                        'is_dPAM'                  : rtt['is_dPAM'],
                        'is_PE3b'                  : ng['is_PE3b'],
                        'RTT_GC'                   : rtt['RTT_GC'],
                        'RTT_length'               : rtt['RTT_length'],
                        'PBS_GC'                   : pbs['PBS_GC'],
                        'PBS_length'               : pbs['PBS_length'],
                        # N_sub/del/ins added by target_mutation.search()
                        'N_subsitution'            : 0,
                        'N_deletion'               : 0,
                        'N_insertions'             : 0,
                        'target_to_sgRNA'          : self.target_to_sgRNA,
                        'target_to_RTT5'           : rtt['target_to_RTT5'],
                    }
                    x_series = pd.Series(x, name=row_name)

                    rows_raw.append(raw_series)
                    rows_X.append(x_series)

        self.rawX = pd.DataFrame(rows_raw) if rows_raw else pd.DataFrame()
        self.X    = pd.DataFrame(rows_X)   if rows_X   else pd.DataFrame()

        if debug >= 10 and not self.rawX.empty:
            print(f"  {self.sgRNA_name}: {len(self.rawX)} pegRNA candidates")

class target_mutation:
	def __init__(self,chr,pos,name,ref,alt,target_fa,**kwargs):
	  #sgRNA name: chr_start_end_strand_seq
		#target_mutation name: id_chr_pos_ref_alt
    #pos is corrected, and the corrected pos, ref, alt is used
    #	target_fa is the +-1000 extended sequences

		self.chr = chr
		self.target_pos = pos
		self.name = name.replace("/","_").replace(",","_")
		self.ref = ref
		self.alt = alt
		self.target_fa = target_fa
		self.debug_folder = "easy_prime_debug_files"
		self.dist_dict = {}
		self.strand_dict = {}
		self.rawX = pd.DataFrame()
		self.X = pd.DataFrame()
		self.X_p = pd.DataFrame()
		self.topX = pd.DataFrame()
		self.allX = pd.DataFrame()
		self.pegRNA_flag=True
		## flags
		self.found_PE3b = False
		self.found_PE3 = False
		self.found_PE2 = False
		self.found_dPAM = False
		self.N_sgRNA_found = 0


		# self.feature_for_prediction = ["sgRNA_distance_to_ngRNA","target_to_sgRNA","target_to_RTT5","N_subsitution","N_deletion","N_insertions","PBS_GC","RTT_GC","PBS_length","RTT_length",'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',"is_dPAM"] # match the order of training features
		self.feature_for_prediction = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','DeepSpCas9',"sgRNA_distance_to_ngRNA","is_dPAM",'is_PE3b','RTT_GC', 'RTT_length', 'PBS_GC', 'PBS_length', 'N_subsitution', 'N_deletion', 'N_insertions',"target_to_sgRNA","target_to_RTT5"] # match the order of training features
		# self.feature_rename = ["ngRNA_pos","Target_pos","Target_end_flank","N_subsitution","N_deletion","N_insertions","PBS_GC","RTT_GC","PBS_length","RTT_length",'Folding_DS_1', 'Folding_DS_2', 'Folding_DS_3', 'Folding_DS_4', 'Folding_DS_5', 'Folding_DS_6', 'Folding_DS_7', 'Folding_DS_8', 'Folding_DS_9','Folding_DS_10',"is_dPAM"]
		self.PE3_model_feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'cas9_score', 'nick_to_pegRNA', 'dPAM', 'PE3b', 'RTT_GC', 'RTT_length', 'PBS_GC', 'PBS_length', 'N_subsitution', 'N_deletion', 'N_insertions', 'Target_pos', 'Target_end_flank']
		self.PE2_model_feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'cas9_score', 'RTT_GC', 'RTT_length', 'PBS_GC', 'PBS_length', 'N_subsitution', 'N_deletion', 'N_insertions', 'Target_pos', 'Target_end_flank','dPAM']



		self.mutation_pos,self.mutation_ref,self.mutation_alt = find_mutation_pos(pos,ref,alt)
		self.sgRNA_strand_df={}
		self.sgRNA_strand_df["+"]=pd.DataFrame()
		self.sgRNA_strand_df["-"]=pd.DataFrame()
		self.valid_init_sgRNA = pd.DataFrame()
		self.all_sgRNA = pd.DataFrame() # used to find ngRNA

		#---------------- features --------------------------------------------------------------------------------------------------------------------------
		# target mutation feature
		self.ref_alt = global_alignments(self.ref,self.alt)
		self.sgRNA_target_distance_dict = {} ## contain valid and invalid sgRNA in the key, but the latter with  distance <0
		self.DeepSpCas9_dict = {}
		self.sgRNA_target_dPAM_dict = {} ## contain binary values of whether the target affect this sgRNA PAM
		# sgRNA distance to ngRNA
		self.dist_dict = {} ## sgRNA_ngRNA_distance_dict


	def init(self,gRNA_search_space=200,search_iteration=1,sgRNA_length=20,PAM="NGG",offset=-3,debug=0,genome_fasta=None,max_RTT_length=40,min_distance_RTT5=5,max_target_to_sgRNA=10,max_max_target_to_sgRNA=30,**kwargs):
		#first step: search sgRNA
		#second step: search PBS
		#search for candidate sgRNAs around target mutation
		#Input:1-	gRNA_search_space: extend pos by +- gRNA_search_space
    #2-search_iteration: if in the search space defined by gRNA_search_space, we fail to find sgRNAs, we will extend the gRNA_search_space further to find at least one sgRNA. (no need to increase it)
		#Output:chr, start, end, sgRNA name, seq, strand, cut_position, valid
		#These will be used later: self.offset ,self.PAM

		if debug>0:
			subprocess.call("mkdir -p %s"%(self.debug_folder),shell=True)		
		self.offset = offset
		self.PAM = PAM


		### find all sgRNA given a sequence
		for i in range(search_iteration):
			extend = gRNA_search_space*(i+1)
			if i >=1:
				print ("No sgRNA were found using %s gRNA_search_space"%(extend))
			## modified for fasta input
			start = max(self.mutation_pos-extend,0)
			end = self.mutation_pos+extend
			if len(self.target_fa) <= extend*2:
				search_fa = self.target_fa
				start = 0
			else:
				search_fa = sub_fasta_single(self.target_fa,self.target_pos, start,end)
			df = run_pam_finder(search_fa,"N"*sgRNA_length,self.PAM,start,self.chr)
			## df contains all sgRNAs
			self.N_sgRNA_found = df.shape[0]

			if df.shape[0] > 0:
				self.DeepSpCas9_dict = get_DeepSpCas9_score(df[4].unique().tolist())
			try:
				df[1] = df[1].astype(int)
				df[2] = df[2].astype(int)
				## sgRNA name
				df[4] = df[0]+"_"+df[1].astype(str)+"_"+df[2].astype(str)+"_"+df[5].astype(str)+"_"+df[3].astype(str)
				df.index = df[4].to_list()
				df['cut'] = [get_gRNA_cut_site(x[1],x[2],x[5],self.offset) for i,x in df.iterrows()]
				df['target_distance'] = [is_gRNA_valid([r[0],r['cut']],[self.chr,self.mutation_pos],r[5],self.target_pos,len(self.mutation_ref)) for i,r in df.iterrows()]


				## gRNA validation given target mutation
				if debug > 5:
					print ("total sgRNA found (contain invalid sgRNAs): %s"%(df.shape[0]))
					df.to_csv("%s/%s.init.all_sgRNAs.bed"%(self.debug_folder,self.name),sep="\t",header=False,index=False)

				self.valid_init_sgRNA = df[df.target_distance.between(1,max_target_to_sgRNA)][[0,1,2,3,4,5,'cut']]
				current_max_target_to_sgRNA = max_target_to_sgRNA+5

				while self.valid_init_sgRNA.shape[0] == 0:
					if debug>=10:
						print ("increasing max_target_to_sgRNA to:", current_max_target_to_sgRNA)
					if current_max_target_to_sgRNA > max_max_target_to_sgRNA:
						break
					self.valid_init_sgRNA = df[df.target_distance.between(1,current_max_target_to_sgRNA)][[0,1,2,3,4,5,'cut']]
					if self.valid_init_sgRNA.shape[0] > 0:
						print ("max_target_to_sgRNA increased from %s to %s"%(max_target_to_sgRNA,current_max_target_to_sgRNA))
						break
					current_max_target_to_sgRNA += 5
				## sgRNA features
				self.sgRNA_target_distance_dict = df['target_distance'].to_dict()

				if debug > 5:
					print ("showing sgRNAs between 1 to %s"%(current_max_target_to_sgRNA))
					print (df[df.target_distance.between(1,current_max_target_to_sgRNA)])
				df = df.drop(['target_distance'],axis=1)
				if self.valid_init_sgRNA.shape[0] == 0:
					print ("No sgRNA was found for %s using %s gRNA_search_space"%(self.name,extend))
					continue
				else:
					self.found_PE2 = True
					print ("%s valid sgRNAs found for  %s"%(self.valid_init_sgRNA.shape[0],self.name))
					self.dist_dict = distance_matrix(df.values.tolist())
					self.sgRNA_strand_df['+'] = df[df[5]=="+"][[0,1,2,3,4,5]]
					self.sgRNA_strand_df['-'] = df[df[5]=="-"][[0,1,2,3,4,5]]
					self.all_sgRNA = df.copy()
					# self.sgRNA_target_dPAM_dict = {i: is_dPAM(PAM_seq, RTT, self.offset) for i, r in self.valid_init_sgRNA.iterrows()}
					# self.sgRNA_target_dPAM_dict = {i: is_dPAM(self.PAM, self.target_pos,self.ref,self.alt,r[0:4].tolist()) for i, r in self.valid_init_sgRNA.iterrows()}


					break

			except Exception as e:
				print (e)
				print ("Error or No sgRNA was found for %s using %s gRNA_search_space"%(self.name,extend))

		if debug > 5:
			print ("Target name: ",self.name)
			print (self.valid_init_sgRNA.head().to_string(index=False))


	def search(self,debug=0,scaffold=None,**kwargs):
		#Second step: search for all possible PBS, RTS, pegRNA, nick-gRNA combos
		#Input:length min and max to define search space
		#Output:1. valid sgRNA list
		#     	2. PBS dataframe
		#       3. RTT dataframe
		#       4. ngRNA dataframe

		if not self.found_PE2:
			return 0


		self.sgRNA_list = [sgRNA(
								chr = x[0],
								start = x[1],
								end = x[2],
								seq = x[3],
								sgRNA_name = x[4],
								strand = x[5],
								cut_position = x[6],
								mutation_pos = self.mutation_pos,mutation_ref = self.mutation_ref,mutation_alt = self.mutation_alt,
								user_target_pos = self.target_pos,user_ref = self.ref,user_alt = self.alt,
								offset = self.offset,target_to_sgRNA = self.sgRNA_target_distance_dict[x[4]],
								variant_id = self.name,
								dist_dict = self.dist_dict,
								opposite_strand_sgRNAs = self.sgRNA_strand_df[get_opposite_strand(x[5])],
								all_sgRNA_df = self.all_sgRNA,
								target_fa = self.target_fa,
								scaffold_seq = scaffold,
								PAM = self.PAM,
								DeepSpCas9 = self.DeepSpCas9_dict[x[3]]
								)
						for x in self.valid_init_sgRNA.values.tolist()]

		[run_sgRNA_search(s,**dict(kwargs,debug=debug)) for s in self.sgRNA_list]

		self.rawX = pd.concat([s.rawX for s in self.sgRNA_list])
		if debug>=10:
			print (self.name,"combined rawX:")
			print (self.rawX.head())
		if self.rawX.shape[0]==0:
			self.found_PE2=False
			return 0
		self.X = pd.concat([s.X for s in self.sgRNA_list])
		no_ngRNA = sum([s.no_ngRNA for s in self.sgRNA_list])
		if no_ngRNA == len(self.sgRNA_list):
			print ("%s only PE2 found"%(self.name))
		else:
			self.found_PE3 = True


		self.X['N_insertions'] = self.ref_alt[2]
		self.X['N_subsitution'] = self.ref_alt[1]
		self.X['N_deletion'] = self.ref_alt[3]


		self.found_PE3b = (self.X['is_PE3b']==1).any()
		self.found_dPAM = (self.X['is_dPAM']==1).any()




# After finding all the sequences, we now need to predict their scores
	def predict(self, debug=0, PE2_model=None, PE3_model=None,
		use_transformer=False, transformer_model=None, **kwargs):
		if not self.found_PE2:
			return 0

		if use_transformer:
			import torch
			from tokenizer_order3 import build_input_from_rawX

			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			#model = torch.load(transformer_model, map_location=device)
			model = torch.load(transformer_model, map_location=device, weights_only=False)
			model.to(device)
			model.eval()

			tensors = [[] for _ in range(9)]
			for idx in self.X.index:
				# rawX e self.X t�m o mesmo �ndice � acessa sequ�ncias por idx
				raw_row = self.rawX.loc[idx]
				toks = build_input_from_rawX(raw_row, target_fa=self.target_fa)
				for i in range(9):
					tensors[i].append(toks[i])

			with torch.no_grad():
				batch = [torch.tensor(t, dtype=torch.long, device=device) for t in tensors]
				scores = model(batch).squeeze(-1).cpu().tolist()

			myPred = pd.DataFrame({'predicted_efficiency': scores}, index=self.X.index)

		else:
			with open(PE2_model, 'rb') as f:
				xgb_model_PE2 = pickle.load(f)
			with open(PE3_model, 'rb') as f:
				xgb_model_PE3 = pickle.load(f)

			X_feat = self.X[self.feature_for_prediction].copy()
			X_feat.columns = self.PE3_model_feature_names
			X_PE2 = X_feat[X_feat.nick_to_pegRNA.isnull()]
			X_PE3 = X_feat[~X_feat.nick_to_pegRNA.isnull()]

			pred_y_PE2 = xgb_model_PE2.predict(X_PE2[self.PE2_model_feature_names])
			pred_y_PE3 = xgb_model_PE3.predict(X_PE3)

			myPred = pd.DataFrame()
			myPred['predicted_efficiency'] = pred_y_PE2.tolist() + pred_y_PE3.tolist()
			myPred.index = X_PE2.index.tolist() + X_PE3.index.tolist()

		self.X_p = pd.concat([self.X, myPred], axis=1)
		self.rawX['predicted_efficiency'] = myPred.loc[self.rawX.index]['predicted_efficiency']
		self.X_p = self.X_p.sort_values("predicted_efficiency", ascending=False)
		self.rawX = self.rawX.sort_values("predicted_efficiency", ascending=False)


def run_sgRNA_search(s,**kwargs):
	s.find_RTT(**kwargs)
	s.find_PBS(**kwargs)
	s.find_nick_gRNA(**kwargs)
	s.get_rawX_and_X(**kwargs)