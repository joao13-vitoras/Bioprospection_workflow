import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def PCA_optimal(database):
	prot_resi = ''
	PC_optimal = []
	for i in range(0, 2):
		if i == 0:
			prot_resi = 'prot'
		elif i == 1:
			prot_resi = 'resi'
		# Load file
		emb = []

		# File path
		with open(database + f'/{prot_resi}_embeddings.txt', 'r') as file:
			for line in file:
				if line.startswith(">") == False:
					temp = []
					temp.append([float(i) for i in line.split()])
					emb.append(temp[0])

		# List to numpy array
		embeddings = np.array(emb)

		# PCA
		pca = PCA().fit(embeddings)

		# Cumulative Explained Variance
		cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
		n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
		PC_optimal.append(n_components_95)

		# Compute reconstruction error for all possible PCA dimensions
		reconstruction_errors = []
		max_components = embeddings.shape[0]

		for n in range(1, n_components_95 + 1):
			pca = PCA(n_components=n)
			X_reduced = pca.fit_transform(embeddings)
			X_recon = pca.inverse_transform(X_reduced)
			mse = mean_squared_error(embeddings, X_recon)
			reconstruction_errors.append(mse)

		# Plotting
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

		# Plot 1: Cumulative Explained Variance
		ax1.plot(range(1, max_components + 1), cumulative_variance, 'o-', label='Cumulative Explained Variance')
		ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
		ax1.axvline(x=n_components_95, color='g', linestyle=':', label=f'Optimal PCs = {n_components_95}')
		ax1.set_xlabel('Number of Principal Components')
		ax1.set_ylabel('Cumulative Explained Variance')
		ax1.set_title('Cumulative Explained Variance Ratio')
		ax1.legend()
		ax1.grid(True)

		# Plot 2: Reconstruction Error
		ax2.plot(range(1, n_components_95 + 1), reconstruction_errors, 'o-', color='orange')
		ax2.axhline(y=reconstruction_errors[-1], color='r', linestyle='--', label='MSE 95% Variance Threshold')
		ax2.axvline(x=n_components_95, color='g', linestyle=':', label=f'Optimal PCs = {n_components_95}')
		ax2.set_xlabel('Number of Principal Components')
		ax2.set_ylabel('Reconstruction MSE')
		ax2.set_title('Reconstruction Error vs. PCA Components')
		ax2.legend()
		ax2.grid(True)

		plt.tight_layout()
		plt.savefig(f'{database}/graphs/PCA_MSE_{prot_resi}.png', bbox_inches='tight')

	return PC_optimal
