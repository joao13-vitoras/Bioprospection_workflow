import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Open file and extract the results
num_files = int(input('Number of files to plot:	'))

# Empty dictonary
groups = {}

for i in range(0, num_files):
	# File path
	result_path = input('Path to result txt file:   ')
	name = input('What is the name for this cluster?	')
	emb = []
	with open(result_path, 'r') as file:
		for line in file:
			if line.startswith(">") == False:
				vector = list(map(float, (line.split(' '))))
				emb.append(vector)
	groups[name] = emb

# Flatten embeddings and labels
embeddings = []
labels = []
for group_name, emb_list in groups.items():
    embeddings.extend(emb_list)
    labels.extend([group_name] * len(emb_list))

embeddings = np.array(embeddings)

# PCA
n_PC_optimal = int(input('Optimal PC number:	'))
pca = PCA(n_components=n_PC_optimal)
X_pca = pca.fit_transform(embeddings)

# t-SNE
for i in range(10,21):
	tsne = TSNE(n_components=2, random_state=42, perplexity=i)
	X_tsne = tsne.fit_transform(X_pca)

	# Plot
	plt.figure(figsize=(10, 8))
	unique_labels = np.unique(labels)
	for label in unique_labels:
		idx = np.array(labels) == label
		plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label, s=60)

	plt.xlabel(f"t-SNE 1")
	plt.ylabel(f"t-SNE 2")
	plt.title(f"t-SNE of Protein Embeddings by Group {i}")
	plt.legend()

	plt.show()
