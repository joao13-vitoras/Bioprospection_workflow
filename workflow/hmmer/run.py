import subprocess
import os
import argparse

def run_hmmer(db, search_group_dir):
	# Naming the output paths
	db_name = db.split('/')[-1]
	db_name = db_name.split('.')[0]

	search_group_name = search_group_dir.split('/')[-1]

	search_group_path = f'{search_group_dir}/seqs.fa'

	# Multiple Sequence alignment using MAFFT
	align_file = f'{search_group_dir}/aligned_sequences.fa'

	files_search_group = os.listdir(search_group_dir)
	if not 'aligned_sequences.fa' in files_search_group:
		mafft_command = f'mafft --auto --maxiterate 1000  --thread -1 {search_group_path} > {align_file}'
		subprocess.call(mafft_command, shell=True)

	# Preparing to use HMMER
	output_hmmer_search = f'{search_group_dir}/hmmer'
	os.makedirs(output_hmmer_search, exist_ok=True)

	# Generates the profile of the search group
	search_profile = f'{output_hmmer_search}/profile.hmm'

	files_search_group_hmmer = os.listdir(output_hmmer_search)
	if not 'profile.hmm' in files_search_group_hmmer:
		hmmbuild_command = f'hmmbuild -n {search_group_name} {search_profile} {align_file}'
		subprocess.call(hmmbuild_command, shell=True)

	# Generates the consensus sequence for the search group
	consensus_path = f'{search_group_dir}/consensus.fa'

	if not 'consensus.fa' in files_search_group:
		hmmemit_command = f'hmmemit -o {consensus_path} -c --seed 3 {search_profile}'
		subprocess.call(hmmemit_command, shell=True)

	# Score of the profile against the search group
	profile_vs_self = f'{output_hmmer_search}/profile_vs_self.txt'

	if not 'profile_vs_self.txt' in files_search_group_hmmer:
		hmmsearch_command = f'hmmsearch -o {output_hmmer_search}/profile_vs_self_doms.txt --tblout {profile_vs_self} --noali --nobias --cpu 7 -E 0.001 --domE 0.001 {search_profile} {search_group_path}'
		subprocess.call(hmmsearch_command, shell=True)

	# Minimum and maximum score of profile against search group
	seq_score = []
	with open(profile_vs_self, 'r') as file:
		for line in file:
			if '#' not in line:
				l = line.split()
				seq_score.append(float(l[5]))

	min_score = min(seq_score)
	max_score = max(seq_score)

	# Search group profile against the database
	output_db = f'workflow/results/{db_name}/hmmer'
	os.makedirs(output_db, exist_ok=True)

	results_db = f'{output_db}/results_{search_group_name}.txt'

	files_hmmer_db = os.listdir(output_db)
	if not f'results_{search_group_name}.txt' in files_hmmer_db:
		hmmsearch_command = f'hmmsearch -o {output_db}/results_{search_group_name}_doms.txt --tblout {results_db} --noali --nobias --cpu 7 -E 0.001 --domE 0.001 {search_profile} {db}'
		subprocess.call(hmmsearch_command, shell=True)

		with open(results_db, 'a') as file:
			file.write(f'#Minimum score for profile_vs_self: {min_score}\n')
			file.write(f'#Maximum score for profile_vs_self: {max_score}')

	return None

def parse_args():
	ap = argparse.ArgumentParser(description="HMMER program script")
	ap.add_argument("--database", required=True, help="Path to FASTA database file")
	ap.add_argument('--search_group_dir', required=True, help='Path to the directory with the search group FASTA file')
	return ap.parse_args()

if __name__ == '__main__':
	args = parse_args()
	run_hmmer(args.database, args.search_group_dir)
