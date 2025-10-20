import matplotlib.pyplot as plt
import numpy as np

cosine_similarity = []
with open('workflow/results/protein_catalogue_50_sort/amplify/3_1_8_1_checked_cosine_similarity_prot.tsv', 'r') as file:
	idx = 0
	for line in file:
		if idx != 0:
			score = line.split()[-1]
			cosine_similarity.append(float(score))
		idx += 1

bins = np.arange(0.9, 1.001, 0.005)

counts, edges = np.histogram(cosine_similarity, bins=bins)
percentages = (counts / counts.sum()) * 100

centers = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(8, 5))
plt.bar(centers, percentages, width=0.0045)

plt.title("AMPLIFY scores")
plt.xlabel("Cosine Similarity")
plt.ylabel("Percentage (%)")
plt.xticks(np.arange(0.9, 1.001, 0.01))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

