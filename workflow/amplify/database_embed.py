import argparse
import timeit
import os
from pathlib import Path

import sys
import signal
import torch
import multiprocessing as mp

from scripts.split_sort import sort_in_chunks
from scripts.descriptive_info_database import descriptive_info_seq
from scripts.embed_database import extract_embeddings_batch
from scripts.measure_max_batch import measure_max_batch_size

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("Received interrupt signal, cleaning up ...")
    torch.cuda.empty_cache()
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process(db, db_name, total_seqs, sorted_seqs_dir, end_len, num_gpu):
	run_batch_test = input('Run max batch size test (y/n)?  ')
	if run_batch_test == 'y':
		# Batch size
		max_batch, sizes, results = measure_max_batch_size(end_len)

		# Print summary
		print("\nTest Summary:")
		for size, success in zip(sizes, results):
			print(f"Batch size {size}: {'Success' if success else 'Failed'}")

	batch = int(input('Batch size:  '))

	# Single GPU
	if num_gpu == 1:
		seq_file = os.path.join(sorted_seqs_dir, 'sorted_0.fa')
		extract_embeddings_batch(seq_file, db_name, total_seqs, batch, 0)

	# Multi GPU
	elif num_gpu > 1:
		# Starting GPU processes simultaneously
		processes = []
		if mp.get_start_method(allow_none=True) is None:
			mp.set_start_method('spawn')
		for idx in range(num_gpu):
			seq_file = os.path.join(sorted_seqs_dir, f'sorted_{idx}.fa')
			p = mp.Process(target=extract_embeddings_batch,
			args=(seq_file, db_name, total_seqs, batch, idx),
			name=f"GPU_{idx}_Process")
			p.start()
			processes.append(p)

		# Wait for all processes to complete
		for i, p in enumerate(processes):
			p.join()
			status = "successfully" if p.exitcode == 0 else f"with exit code {p.exitcode}"
			print(f"GPU {i} finished {status}")

def run_amplify(db, init_len, end_len):
	# Naming output directory
	db_name = db.split('/')[-1]
	db_name = db_name.split('.')[0]

	num_gpu = torch.cuda.device_count()

	# Split and sort tha database
	if not os.path.isdir(f'workflow/results/{db_name}/amplify/sorted_sequences'):
		sorted_seqs_dir = sort_in_chunks(db, db_name, num_gpu, init_len, end_len)
	else:
		sorted_seqs_dir = f'workflow/results/{db_name}/amplify/sorted_sequences'

	# Statistics from the aa sequence database
	if not os.path.isdir(f'workflow/results/{db_name}/amplify/info'):
		descriptive_info_seq(db, db_name, init_len, end_len)

	with open(f'workflow/results/{db_name}/amplify/info/aa_database.txt', 'r') as f:
		for line in f:
			if "Total sequences" in line:
				total_seq = int(line.split(':')[-1])
	# Start
	start = timeit.default_timer()

	# Extract embeddings
	if os.path.isdir(f'workflow/results/{db_name}/amplify/embedding'):
		file_count = 0
		path = Path(f'workflow/results/{db_name}/amplify/embedding/b_prot')
		for item in path.rglob('*.pt'):
			file_count += 1

		if file_count < total_seq:
			process(db, db_name, total_seq, sorted_seqs_dir, end_len, num_gpu)

	elif not os.path.isdir(f'workflow/results/{db_name}/amplify/embedding'):
		process(db, db_name, total_seq, sorted_seqs_dir, end_len, num_gpu)

	# End
	end = timeit.default_timer()
	t_min, t_sec = divmod(end-start,60)
	t_hour,t_min = divmod(t_min,60)
	print(f'Embeddings extraction duration = {int(t_hour)} hour:{int(t_min)} min:{int(t_sec)} sec')

	return None

def parse_args():
	ap = argparse.ArgumentParser(description="AMPLIFY GPU program script")
	ap.add_argument("--database", required=True, help="Path to FASTA database file")
	ap.add_argument('--init_len', type=int, default=1, help='Minimum length to start the analysis')
	ap.add_argument('--end_len', type=int, default=2046, help='Maximum length to start the analysis')
	return ap.parse_args()

if __name__ == '__main__':
        args = parse_args()
        run_amplify(args.database, args.init_len, args.end_len)
