import os
import statistics
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter

def descriptive_info_seq(database, db_name, init_len, end_len):
	print('Running: descriptive_info_seq')

	# Output directories
	os.makedirs(f'workflow/results/{db_name}/amplify/info', exist_ok=True)
	os.makedirs(f'workflow/results/{db_name}/amplify/graphs', exist_ok=True)

	# Length of the sequence and count a.a.
	seq_length = []
	aa_total_counts = Counter()

	for record in SeqIO.parse(database, 'fasta'):
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
	plt.title(f"Amino Acid Frequency in {db_name}")

	plt.savefig(f'workflow/results/{db_name}/amplify/graphs/aa_frequency.png', bbox_inches='tight')
	plt.close()

	# Sequence lentgh pie graph
	plt.figure()

	# Bins for different length of the sequences
	length_lim = [init_len, end_len, 2048]

	# Counting
	count = [0, 0, 0, 0]
	for i in seq_length:
		if i <= length_lim[0]:
			count[0] += 1
		elif length_lim[0] < i <= length_lim[1]:
			count[1] += 1
		elif length_lim[1] < i <= length_lim[2]:
			count[2] += 1
		elif length_lim[2] < i:
			count[3] += 1
	# Percentage
	for i in range(len(count)):
		count[i] = ((count[i]/len(seq_length))*100)

	labels = [f'0-{init_len} a.a.', f'{init_len}-{end_len} a.a.', f'{end_len}-2048 a.a.', '> 2048 a.a.']

	plt.pie(count, autopct='%1.3f%%')
	plt.legend(labels, loc="upper right")

	plt.title('Distributions of the a.a. sequences length')

	plt.savefig(f'workflow/results/{db_name}/amplify/graphs/aa_seq_length_pie.png')
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
	with open(f'workflow/results/{db_name}/amplify/info/aa_database.txt', 'w') as f:
    		f.write("=== Sequence Length Statistics ===\n")
    		for k, v in length_stats.items():
        		f.write(f"{k}: {v}\n")

    		f.write("\n=== Amino Acid Frequencies ===\n")
    		for aa, freq in aa_frequencies_sorted.items():
        		f.write(f"{aa}: {freq:.4f}\n")

	return None
