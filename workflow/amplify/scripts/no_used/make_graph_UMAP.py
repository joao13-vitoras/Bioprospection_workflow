from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt

def UMAP_graph(database, PC_optimal):
	prot_resi = ''
	for i in range(0, 2):
		if i == 0:
			prot_resi = 'prot'
		elif i == 1:
			prot_resi = 'resi'

		emb = []
		with open(database + f'/{prot_resi}_embeddings.txt', 'r') as file:
			for line in file:
				if line.startswith(">") == False:
					temp = []
					temp.append([float(i) for i in line.split()])
					emb.append(temp[0])



		# PCA
		n_PC_optimal = PC_optimal[i]
		pca = PCA(n_components=n_PC_optimal)
		X_pca = pca.fit_transform(emb)

		# Apply UMAP (2D)
		metric = 'cosine'
		n_neigh = [15, 100, 500, 1000]

		# Create figure
		fig, axes = plt.subplots(1, len(n_neigh), figsize=(15, 10))

		plt.suptitle(f"UMAP Embeddings | metric = {metric}", fontsize=16, y=1.02)

		for j in range(len(n_neigh)):
			reducer = umap.UMAP(n_components=2, metric=metric, n_neighbors=n_neigh[j], init='random', random_state=42)
			X_umap = reducer.fit_transform(X_pca)

			ax = axes[j]

			scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=5)
			ax.set_title(f"Number of neighbors = {n_neigh[j]}", fontsize=10)
			ax.set_xticks([])
			ax.set_yticks([])

			plt.tight_layout()

		plt.savefig(f'{database}/graphs/UMAP_{prot_resi}.png', bbox_inches='tight')

	return None
