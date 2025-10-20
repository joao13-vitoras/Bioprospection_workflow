import argparse
import subprocess
import torch
import os

from scripts.split_sort import split_fasta_sorted_chunks
from scripts.descriptive_inf_seq_data import descriptive_info_seq
from scripts.pt_to_txt_multi import torch_to_txt
from scripts.embeddings_statistics import embeddings_stats
from scripts.measure_max_batch import measure_max_batch_size

def run_amplify(db, init_len, end_len):
	# Naming output directory
	db_name = db.split('/')[-1]
	db_name = db_name.split('.')[0]

	# Split and sort tha database
	if not os.path.isdir(f'workflow/results/{db_name}/amplify/sorted_seqs'):
		sorted_seqs = split_fasta_sorted_chunks(db, db_name, 10000)
	else:
		sorted_seqs = f'workflow/results/{db_name}/amplify/sorted_sequences.fa'

	# Statistics from the aa sequence database
	if not os.path.isdir(f'workflow/results/{db_name}/amplify/info'):
		descriptive_info_seq(db, db_name, init_len, end_len)

	# Extract embeddings
	files_amplify = os.listdir(f'workflow/results/{db_name}/amplify')
	if not (os.path.isdir(f'workflow/results/{db_name}/amplify/embedding') or 'embed_prot.txt' in files_amplify):

		run_batch_test = input('Run max batch size test (y/n)?	')
		if run_batch_test == 'y':
			# Batch size
			max_batch, sizes, results = measure_max_batch_size(end_len)

			# Print summary
			print("\nTest Summary:")
			for size, success in zip(sizes, results):
				print(f"Batch size {size}: {'Success' if success else 'Failed'}")

		batch = int(input('Batch size:	'))
		num_gpus = torch.cuda.device_count()

		# Fragmentation problems
		os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:4'

		amplify_command = f'torchrun --standalone --nproc_per_node={num_gpus} ~/workflow/amplify/scripts/batch_multi.py --database {sorted_seqs} --batch_size {batch} --init_len {init_len} --end_len {end_len}'
		subprocess.call(amplify_command, shell=True)

	# Torch to txt format
	if not 'embed_prot.txt' in files_amplify:
		torch_to_txt(f'workflow/results/{db_name}/amplify/embedding', db_name)

	# Embeddings statistics
	files_db_info = os.listdir(f'workflow/results/{db_name}/amplify/info')
	if not 'embeddings_prot_stats.txt' in files_db_info:
		embeddings_stats(f'workflow/results/{db_name}/amplify', db_name)

	return None

def parse_args():
	ap = argparse.ArgumentParser(description="AMPLIFY multi  gpu program script")
	ap.add_argument("--database", required=True, help="Path to FASTA database file")
	ap.add_argument('--init_len', type=int, default=1,  help='Minimum length to start the analysis')
	ap.add_argument('--end_len', type=int, default=2046, help='Maximum length to start the analysis')
	return ap.parse_args()

if __name__ == '__main__':
	args = parse_args()
	run_amplify(args.database, args.init_len, args.end_len)
