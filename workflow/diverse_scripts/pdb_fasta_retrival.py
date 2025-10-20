import os
import requests
from tqdm import tqdm

# Inputs and directory name
pdb_ids = list(input('PDB ids:	').split(','))
num_ids = len(pdb_ids)
print(f'The input has {num_ids} PDB ids')

group_name = input('Name of group: ')

output_dir = f'workflow/results/imported_seq_pdb/{group_name}'
os.makedirs(output_dir, exist_ok=True)

p_bar = tqdm(total= num_ids, desc=f"Requesting aa sequence")

seq_mer = []
with open(f'{output_dir}/seqs.fa', 'w') as file:
	for id in pdb_ids:
		p_bar.set_postfix({'Id:': f'{id}'})
		url = f"https://www.rcsb.org/fasta/entry/{id}"
		fasta = requests.get(url).text
		file.write(fasta)
		chain_count = fasta.split('|')[1].count(',') + 1
		seq_mer.append([fasta,chain_count])
		p_bar.update(1)

p_bar.close()

# Sorting mer
print('Sorting mer: Monomer, Dimer ...')

os.makedirs(f'{output_dir}/multimers', exist_ok=True)

mer_classification = {1: 'monomer', 2: 'dimer', 3: 'trimer', 4: 'tetramer', 5: 'pentamer', 6: 'hexamer', 7: 'heptamer', 8: 'octamer', 9: 'nonamer', 10: 'decamer'}
mer = []
for i in seq_mer:
	if i[-1] not in mer:
		mer.append(i[-1])

files = []
for num in mer:
	name_mer = mer_classification[num]
	files.append([f'{name_mer}.fa', num])

for i in files:
	with open(f'{output_dir}/multimers/{i[0]}', 'w') as file:
		for seq in seq_mer:
			count = 0
			if seq[-1] == i[-1]:
				fasta_pdb = seq[0]
				file.write(fasta_pdb)
				count += 1
