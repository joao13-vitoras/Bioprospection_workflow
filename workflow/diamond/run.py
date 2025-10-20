import subprocess
import os
import argparse
from Bio import AlignIO
from collections import Counter

def run_diamond(db, search_group_dir):
	# Directory creation
	db_name = db.split('/')[-1]
	db_name = db_name.split('.')[0]

	output_dir = f'workflow/results/{db_name}/diamond'
	os.makedirs(output_dir, exist_ok=True)

	# Make Diamond format dabase file
	db_files = os.listdir(output_dir)
	if not 'database.dmnd' in db_files:
		subprocess.run(['diamond', 'makedb', '--in', db, '-d', f'{output_dir}/database'])

	# Search group paths
	group_name = search_group_dir.split('/')[-1]
	search_group_path = f'{search_group_dir}/seqs.fa'

	# Consensus sequence path for search group
	consensus_path = f'{search_group_dir}/consensus.fa'

	# Comparative bitscore
	os.makedirs(f'{search_group_dir}/diamond', exist_ok=True)
	search_files = os.listdir(f'{search_group_dir}/diamond')

	if not 'search_group.dmnd' in search_files:
		subprocess.run(['diamond', 'makedb', '--in', search_group_path, '-d', f'{search_group_dir}/diamond/search_group'])

	if not 'consensus_vs_self.tsv' in search_files:
		subprocess.run(['diamond', 'blastp',
		'-q', consensus_path,
		'-d', f'{search_group_dir}/diamond/search_group.dmnd',
		'-o', f'{search_group_dir}/diamond/consensus_vs_self.tsv',
		'--very-sensitive',
		'--outfmt', '6', 'qseqid', 'sseqid', 'evalue', 'bitscore',
		'--header',
		'-k0',
		'-e', '0.001'])

	# Minimum and maximum bit score values for consensus_vs_self = reference value
	bit_score = []
	with open(f'{search_group_dir}/diamond/consensus_vs_self.tsv', 'r') as file:
		for line in file:
			if '#' not in line:
				temp = list(line.split('\t'))
				bit_score.append(float(temp[-1].replace('\n', '')))

	min_bit_score =  min(bit_score)
	max_bit_score = max(bit_score)

	# Run protein search
	if not f'blastp_{group_name}.tsv' in db_files:
		subprocess.run(['diamond', 'blastp',
		'-q', consensus_path,
		'-d', f'{output_dir}/database.dmnd',
		'-o', f'{output_dir}/blastp_{group_name}.tsv',
		'--very-sensitive',
		'--outfmt', '6', 'qseqid', 'sseqid', 'evalue', 'bitscore',
		'--header',
		'-k0',
		'-e', '0.001'])

		with open(f'{output_dir}/blastp_{group_name}.tsv', 'a') as file:
			file.write(f'#Minimum bit score for consensus_vs_self: {min_bit_score}\n')
			file.write(f'#Maximum bit score for consensus_vs_self: {max_bit_score}')

	return None

def parse_args():
	ap = argparse.ArgumentParser(description="DIAMOND program script")
	ap.add_argument("--database", required=True, help="Path to FASTA database file")
	ap.add_argument('--search_group_dir', required=True, help='Path to the directory with the search group FASTA file')
	return ap.parse_args()

if __name__ == '__main__':
	args = parse_args()
	run_diamond(args.database, args.search_group_dir)
