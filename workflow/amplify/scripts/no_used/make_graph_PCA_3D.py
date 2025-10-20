import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

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
pca = PCA(n_components=3)
X_pca = pca.fit_transform(embeddings)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

unique_labels = np.unique(labels)
for label in unique_labels:
	idx = np.array(labels) == label
	ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=label, s=60)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}% variance)")
ax.set_title("PCA of Protein Embeddings by Group")
ax.legend()

plt.show()
