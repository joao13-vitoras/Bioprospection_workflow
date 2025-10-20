import os
import numpy as np
from pathlib import Path
from Bio import SeqIO

# Inputs
path_seq = input('Path to seq file:		')
name_db = path_seq.split('/')[-1]
name_db = name_db.split('.')[0]

dir_pt = f'workflow/results/{name_db}/amplify/embedding/b_prot'

min_len = int(input('Minimal sequence length for embedding:	'))
max_len = int(input('Maximum sequence length for embedding:	'))

# .pt files in dir_pt
pt_dir_path = Path(dir_pt)

num_pt_files = 0
existing_embeddings = set()
for item in pt_dir_path.rglob('*.pt'):
	num_pt_files += 1
	# Mark existing sequences embedded
	if item not in existing_embeddings:
		existing_embeddings.add(item.stem)

# Total of sequences to be extracted
count = 0
seq_id = []
for record in SeqIO.parse(path_seq, "fasta"):
	if len(record.seq) >= min_len and len(record.seq) <= max_len:
		count += 1
		seq_id.append(record.id)

# Not embedded sequences
unique_pt = len(existing_embeddings)
no_embed = count - unique_pt
percentage = no_embed/count
print(f"The number of sequences that is not embedded yet is {no_embed} of {count} = {percentage:.2}")

# Duplicates
duplicates = num_pt_files - unique_pt
if duplicates > 0:
	print(f'There is {duplicates} duplicates.')

# Check id match
id_match = 0
no_embed_id = 0
for id in seq_id:
	if id in existing_embeddings:
		id_match += 1
	else:
		no_embed_id += 1
print(f"There are {id_match} sequences that have matches and {no_embed_id} that don't have.")

# Redistribute no embedded sequences
redistribute = input('Redistribute sequences not embedded? (y/n) 	')

if redistribute == 'y':
	# Decreasing order database in blocks
	file_sort = input('Is the database file sorted? (y/n)	')
	if file_sort == 'n':
		temp = f'{path_seq}.temp'
		open_temp = open(temp, 'w')

		block = []
		for record in SeqIO.parse(path_seq, 'fasta'):
			block.append([len(record.seq), record.id, record.seq])

			if len(block) >= 1000000:
				block.sort(reverse=True, key=lambda x: x[0])
				for i in block:
					open_temp.write(f'>{i[1]}\n{i[2]}\n')
				block = []

		block.sort(reverse=True, key=lambda x: x[0])
		for i in block:
			open_temp.write(f'>{i[1]}\n{i[2]}\n')

		open_temp.close()

		# Replace original with sorted version
		os.replace(temp, path_seq)

	# Original folder
	original_folder = f'workflow/results/{name_db}/amplify/sorted_sequences'
	seq_files = os.listdir(original_folder)

	# Control redistribution
	print(seq_files)
	files_exceptions = input('Indexes to exclude files from the redistribution:  	')
	files_exceptions = [int(idx) for idx in files_exceptions.split()]

	exceptions = []
	for idx in files_exceptions:
		exceptions.append(seq_files[idx])

	# Open files
	open_files = []
	aa_count = []
	for i in seq_files:
		if i not in exceptions:
			print(f'Rewriting file {i} to redistribute the sequences')
			open_files.append(open(f'{original_folder}/{i}', 'w'))
			aa_count.append(0)

	# Define sequences to each file
	for record in SeqIO.parse(path_seq, "fasta"):
		if record.id in existing_embeddings:
			continue

		elif len(record.seq) >= min_len and len(record.seq) <= max_len:
			idx = np.argmin(aa_count)
			aa_count[idx] += len(record.seq)
			open_files[idx].write(f">{record.id}\n{str(record.seq)}\n")

	# Close all output files
	for f in open_files:
		f.close()

	print("Redistribution completed successfully!")
