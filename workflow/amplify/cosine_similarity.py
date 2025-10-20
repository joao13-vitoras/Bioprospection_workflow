import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import faiss

def set_file_path(dir, processed_set, chunk = 10000):
	path = Path(dir)

	files_path = []
	for file in path.rglob('*.pt'):
		if file.stem not in processed_set:
			files_path.append(file)

			if len(files_path) >= chunk:
				yield files_path
				files_path = []

	if files_path:
		yield files_path


def load_pt_file(file_path):
	id = file_path.stem

	with open(file_path, 'rb') as f:
		embedding = torch.load(f)

	embedding = embedding.numpy()

	return id, embedding

def cosine_similarity_database(db_name, database_dir, EC_number, embed_type, query, processed_ids, min_cosine_score):
	# Set directory
	out_file = f'workflow/results/{db_name}/amplify/{EC_number}_cosine_similarity_{embed_type}.tsv'
	record_ids = f'workflow/results/{db_name}/amplify/seq_id_searched_{embed_type}.txt'

	if not f'{EC_number}_cosine_similarity_{embed_type}.tsv' in os.listdir(f'workflow/results/{db_name}/amplify'):
		with open(out_file, 'a') as file:
			file.write("id\tcosine_similarity\n")

	matches = 0
	counter = tqdm(desc=f"Sequences processed", position=1, leave=False, dynamic_ncols=True)

	# Load chunk of the files
	for files_path in set_file_path(database_dir, processed_ids):
		ids = []
		ids_write = []
		embeddings = []

		for path in files_path:
#			time.sleep(0.001)
			id, embedding = load_pt_file(path)

			ids.append(id)
			ids_write.append(id + '\n')
			embeddings.append(embedding)

		# Use faiss to take the cosine similarity
		dim = 960

		# Format this chunk of embeddings
		embeddings_np = np.array(embeddings)
		faiss.normalize_L2(embeddings_np)

		# Construct the index
		index = faiss.IndexFlatIP(dim)
		index.add(embeddings_np)
		num_embeddings = index.ntotal

		# Format query embedding == Centroid sequence embedding
		faiss.normalize_L2(query)

		# Search
		cosine_similarity, indices = index.search(query, num_embeddings)

		# Write results - Cosine similarity greater than min_score
		with open(out_file, 'a') as file:
			for i in range(len(cosine_similarity[0])):
				if cosine_similarity[0][i] >= min_cosine_score:
					file.write(f"{ids[indices[0][i]]}\t{cosine_similarity[0][i]:.6f}\n")
					matches += 1

		with open(record_ids, 'a') as file:
			file.writelines(ids_write)

		# Print progress
		counter.update(num_embeddings)

		# Timeout I/O operations
#		time.sleep(1)

	return None

def search_group_against_centroid(EC_number, query, search_ids, search_embeddings, embed_type):
	# Format the search_embeddings
	faiss.normalize_L2(search_embeddings)
	dim = 960

	index = faiss.IndexFlatIP(dim)
	index.add(search_embeddings)
	num_embeddings = index.ntotal

	# Format query embedding - Centroid
	faiss.normalize_L2(query)

	# Search
	cosine_similarity, indices = index.search(query, num_embeddings)

	# Write results - EC_number_agaist_self
	with open(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/centroid_{embed_type}_vs_self.tsv', 'w') as file:
		file.write("#id\tcosine_similarity\n")
		for i in range(len(cosine_similarity[0])):
			file.write(f"{search_ids[indices[0][i]]}\t{cosine_similarity[0][i]:.6f}\n")

	return np.mean(cosine_similarity[0]), np.std(cosine_similarity[0])

def run(EC_number, db_name):
	print('Running: cosine_similarity')

	# Set paths
	prot_resi = ['prot', 'resi']
	for embed_type in prot_resi:
		search_group_dir = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/embedding/b_{embed_type}'
		database_dir = f'workflow/results/{db_name}/amplify/embedding/b_{embed_type}'
		record_ids = f'workflow/results/{db_name}/amplify/seq_id_searched_{embed_type}.txt'

		# Set search_group .pt files paths
		search_group_path = Path(search_group_dir)
		search_pt_files = list(search_group_path.rglob('*.pt'))

		# Load search_group embeddings
		search_ids = []
		search_embeddings = []

		for file in search_pt_files:
			id, embedding = load_pt_file(file)

			search_ids.append(id)
			search_embeddings.append(embedding)

		embeddings_np = np.array(search_embeddings)

		# Centroid
		centroid = embeddings_np.mean(axis=0).reshape(1, -1)

		# Search_group vs centroid
		if not f'centroid_{embed_type}_vs_self.tsv' in os.listdir(f'workflow/results/imported_seq_pdb/{EC_number}/amplify'):
			mean_cosine_score, std_cosine_score = search_group_against_centroid(EC_number, centroid, search_ids, embeddings_np, embed_type)

		else:
			search_cosine_score = []
			with open(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/centroid_{embed_type}_vs_self.tsv', 'r') as file:
				count_line = 0
				for line in file:
					if count_line != 0:
						score = line.split('\t')[-1]
						search_cosine_score.append(float(score))
					count_line += 1

			# Reference value for search in database
			mean_cosine_score = np.mean(search_cosine_score)
			std_cosine_score = np.std(search_cosine_score)

		# Set of processed ids
		processed_ids = set()
		if f'seq_id_searched_{embed_type}.txt' in os.listdir(f'workflow/results/{db_name}/amplify'):
			with open(record_ids, 'r') as file:
				for lines in file:
					processed_ids.add(lines.split()[0])

		# Database search
		if not (f'{EC_number}_cosine_similarity_{embed_type}.tsv' in os.listdir(f'workflow/results/{db_name}/amplify') and len(processed_ids) >= 6552584):
			min_cosine_score = mean_cosine_score - std_cosine_score
			cosine_similarity_database(db_name, database_dir, EC_number, embed_type, centroid, processed_ids, min_cosine_score)

	return None

def parse_args():
	ap = argparse.ArgumentParser(description="FAISS cosine similarity search")
	ap.add_argument("--search_group_dir", required=True, help="Path to embeded search_group directory")
	ap.add_argument("--database", required=True, help="Path to database fasta file")
	return ap.parse_args()

if __name__ == '__main__':
	args = parse_args()

	EC_number = args.search_group_dir.split('/')[-1]

	db_name = args.database.split('/')[-1]
	db_name = db_name.split('.')[0]

	run(EC_number, db_name)
