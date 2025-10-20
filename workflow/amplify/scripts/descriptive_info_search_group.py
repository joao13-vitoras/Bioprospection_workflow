import os
import statistics
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter

def descriptive_info_seq(search_group, EC_number):
	print('Running: descriptive_info_seq')

	# Output directories
	os.makedirs(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/info', exist_ok=True)
	os.makedirs(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/graphs', exist_ok=True)

	# Length of the sequence and count a.a.
	seq_length = []
	aa_total_counts = Counter()

	for record in SeqIO.parse(search_group, 'fasta'):
		sequence = str(record.seq)
		seq_length.append(len(sequence))
		aa_total_counts.update(sequence)

	# Convert to dictionary and sort by amino acid
	total_aa = sum(seq_length)
	aa_frequencies = {aa: 100*(count / total_aa) for aa, count in aa_total_counts.items()}
	aa_frequencies_sorted = dict(sorted(aa_frequencies.items(), key=lambda x: x[1], reverse=True))

	plt.figure()
	plt.bar(aa_frequencies_sorted.keys(), aa_frequencies_sorted.values())
	plt.xlabel("Amino Acid")
	plt.ylabel("Amino Acid frequency (%)")
	plt.title(f"Amino Acid Frequency in {EC_number}")

	plt.savefig(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/graphs/aa_frequency.png', bbox_inches='tight')
	plt.close()

	# Compute statistics
	total_seqs = len(seq_length)

	try:
    		mode_len = statistics.mode(seq_length)
	except statistics.StatisticsError:
    		mode_len = "No unique mode"

	mean_len = statistics.mean(seq_length)
	std_dev = statistics.stdev(seq_length)
	cv = std_dev / mean_len

	length_stats = {
		"Total sequences": total_seqs,
		"Total amino acids": total_aa,
		"Min length": min(seq_length),
		"Max length": max(seq_length),
		"Mean length": round(mean_len, 4),
		"Median length": round(statistics.median(seq_length), 4),
		"Variance": round(statistics.variance(seq_length), 4),
		"Standard Deviation": round(std_dev, 4),
		"Coefficient of Variation": round(cv, 4)
	}

	# Write to file
	with open(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/info/aa_database.txt', 'w') as f:
    		f.write("=== Sequence Length Statistics ===\n")
    		for k, v in length_stats.items():
        		f.write(f"{k}: {v}\n")

    		f.write("\n=== Amino Acid Frequencies ===\n")
    		for aa, freq in aa_frequencies_sorted.items():
        		f.write(f"{aa}: {freq:.4f}\n")

	return None
