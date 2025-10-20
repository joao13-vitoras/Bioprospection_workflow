from Bio import SeqIO

count = [0, 0, 0, 0]

id_diamond = []
with open('workflow/results/protein_catalogue_50_sort/diamond/blastp_3_1_8_1_checked.tsv' ,'r') as file:
	for line in file:
		if not line.startswith('#'):
			id = line.split()[1]
			id_diamond.append(id)

id_hmmer = []
with open('workflow/results/protein_catalogue_50_sort/hmmer/results_3_1_8_1_checked.txt' ,'r') as file:
	for line in file:
		if not line.startswith('#'):
			id = line.split()[0]
			id_hmmer.append(id)

id_amplify, cosine_score = [], []
with open('workflow/results/protein_catalogue_50_sort/amplify/3_1_8_1_checked_cosine_similarity_prot.tsv' ,'r') as file:
	for line in file:
		if not line.startswith('#'):
			id = line.split()[0]
			score = line.split()[-1]
			id_amplify.append(id)
			cosine_score.append(float(score))

sorted_pairs = sorted(zip(cosine_score, id_amplify))
sorted_cosine, sorted_id_amplify = zip(*sorted_pairs)

best_1000 = sorted_id_amplify[-1000:]

count = [0, 0, 0, 0]
for id in best_1000:
	if id in id_diamond and id in id_hmmer:
		count[2] += 1
	elif id in id_hmmer:
		count[1] += 1
	elif id in id_diamond:
		count[0] += 1
	else:
		count[3] += 1

print(count)

id_seq = dict()
for record in SeqIO.parse('seq_databank/protein_catalogue_50_sort.fa', 'fasta'):
	id_seq[record.id] = record.seq

seqs_best_1000 = []
for id in best_1000:
	seqs_best_1000.append(f'>{id}\n{id_seq[id]}\n')

with open('best_1000.fa', 'w') as file:
	file.writelines(seqs_best_1000)

#best = best_1000[-1]
#idx = 0
#for id in sorted_id_amplify:
#	if id == best:
#		break
#	idx += 1

count_homo = [0, 0]
for id in id_hmmer:
	if id in id_diamond:
		count_homo[0] += 1
	else:
		count_homo[1] += 1

print(count_homo)

