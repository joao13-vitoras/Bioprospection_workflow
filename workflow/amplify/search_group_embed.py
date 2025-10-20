import os
import timeit
import argparse
import shutil
from pathlib import Path

import sys
import signal

import torch
import multiprocessing as mp

from scripts.sort_search_group import sort_in_chunks
from scripts.descriptive_info_search_group import descriptive_info_seq
from scripts.embed_search_group import extract_embeddings_batch
from scripts.embed_statistics import embeds_stats



def signal_handler(sig, frame):
	"""Handle interrupt signals gracefully"""
	print("Received interrupt signal, cleaning up ...")
	torch.cuda.empty_cache()
	sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process(EC_number, sorted_seqs_dir, total_seqs, num_gpu):
	batch = int(input('Batch size:  '))

	# Single GPU
	if num_gpu == 1:
		seq_file = os.path.join(sorted_seqs_dir, 'sorted_0.fa')
		extract_embeddings_batch(seq_file, EC_number, total_seqs, batch, 0)

	# Multi GPU
	elif num_gpu > 1:
		# Starting GPU processes simultaneously
		processes = []
		if mp.get_start_method(allow_none=True) is None:
			mp.set_start_method('spawn')
		for idx in range(num_gpu):
			seq_file = os.path.join(sorted_seqs_dir, f'sorted_{idx}.fa')
			p = mp.Process(target=extract_embeddings_batch,
			args=(seq_file, EC_number, total_seqs, batch, idx),
			name=f"GPU_{idx}_Process")
			p.start()
			processes.append(p)

		# Wait for all processes to complete
		for i, p in enumerate(processes):
			p.join()
			status = "successfully" if p.exitcode == 0 else f"with exit code {p.exitcode}"
			print(f"GPU {i} finished {status}")



def search_group_embed(EC_number):
	# Search group folder
	dir = f'workflow/results/imported_seq_pdb/{EC_number}'

	# Number of GPUs
	num_gpu = torch.cuda.device_count()

	# Sort the search group file
	search_group_seq_file = f'{dir}/seqs.fa'
	sorted_seq_dir = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/sorted_sequences'

	embed_dir = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/embedding'

	if not (os.path.isdir(sorted_seq_dir) or os.path.isdir(embed_dir)):
		sort_in_chunks(search_group_seq_file, EC_number, num_gpu)

	# Statistics from the aa search group
	if not os.path.isdir(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/info'):
		descriptive_info_seq(search_group_seq_file, EC_number)

	with open(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/info/aa_database.txt', 'r') as f:
		for line in f:
			if "Total sequences" in line:
				total_seq = int(line.split(':')[-1]) + 1

	# Start
	start = timeit.default_timer()

	# Extract embeddings
	if os.path.isdir(embed_dir):
		file_count = 0
		path = Path(f'{embed_dir}/b_prot')
		for item in path.rglob('*.pt'):
			file_count += 1

		# Count the consensus file
		list_EC_dir = os.listdir(dir)
		if 'consensus_prot.pt' in list_EC_dir:
			file_count += 1

		if file_count < total_seq:
			process(EC_number, sorted_seq_dir, total_seq, num_gpu)

	else:
		process(EC_number, sorted_seq_dir, total_seq, num_gpu)

	# End
	end = timeit.default_timer()
	t_min, t_sec = divmod(end-start,60)
	t_hour,t_min = divmod(t_min,60)
	print(f'Embeddings extraction duration = {int(t_hour)} hour:{int(t_min)} min:{int(t_sec)} sec')

	# Embeds statistics
	info_files = os.listdir(f"workflow/results/imported_seq_pdb/{EC_number}/amplify/info")
	if not 'embeddings_resi_stats.txt' in info_files:
		embeds_stats(embed_dir, EC_number)

	# Delete the sorted file
	if os.path.isdir(sorted_seq_dir):
		shutil.rmtree(sorted_seq_dir)

	return None



def parse_args():
	ap = argparse.ArgumentParser(description="AMPLIFY search group embed extraction")
	ap.add_argument("--search_group", required=True, help="Path to search group directory")
	return ap.parse_args()

if __name__ == '__main__':
	args = parse_args()
	EC_number = (args.search_group).split('/')[-1]

	search_group_embed(EC_number)
