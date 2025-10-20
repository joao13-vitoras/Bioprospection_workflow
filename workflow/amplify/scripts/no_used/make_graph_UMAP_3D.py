from sklearn.decomposition import PCA
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Open file and extract the results
num_files = int(input('Number of files to plot: '))

# Empty dictonary
groups = {}

for i in range(0, num_files):
	# File path
	result_path = input('Path to result txt file:   ')
	name = input('What is the name for this cluster?        ')
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
n_PC_optimal = int(input('Optimal PC number:    '))
pca = PCA(n_components=n_PC_optimal)
X_pca = pca.fit_transform(embeddings)

# Apply UMAP (3D)
reducer = umap.UMAP(n_components=3, random_state=42)
X_umap = reducer.fit_transform(X_pca)

# Plotting UMAP results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(labels):
	idx = np.array(labels) == label
	ax.scatter(X_umap[idx, 0], X_umap[idx, 1], X_umap[idx, 2], label=label, s=60)

ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")
ax.set_title("UMAP Projection of Protein Embeddings by Group")
ax.legend()

plt.show()
