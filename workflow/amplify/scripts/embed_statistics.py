from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

def embeds_stats(dir_path, EC_number):
	print('Running: embeddings_stats')

	prot_files = dir_path+'/b_prot'
	resi_files = dir_path+'/b_resi'

	# Load .pt files
	prot_embeds = []
	path = Path(prot_files)
	for file in path.rglob('*.pt'):
		prot_embeds.append(torch.load(file))

	resi_embeds = []
	path = Path(resi_files)
	for file in path.rglob('*.pt'):
		resi_embeds.append(torch.load(file))

	# Convert to numpy format
	prot_embeds = np.stack([embed.numpy() for embed in prot_embeds])
	resi_embeds = np.stack([embed.numpy() for embed in resi_embeds])

	embeds_list = [prot_embeds, resi_embeds]
	names = ['prot', 'resi']

	for i in range(2):
		emb_list = embeds_list[i]
		prot_resi = names[i]

		# Compute coefficient of variation for each dimension
		means = np.mean(emb_list, axis=0)
		stds = np.std(emb_list, axis=0)
		cvs = np.divide(stds, np.abs(means), where=means!=0)

		# Handle infinite/NaN values that might occur
		cvs = np.nan_to_num(cvs, nan=0, posinf=0, neginf=0).flatten()
		means = means.flatten()
		stds = stds.flatten()

		# Plot bar graph
		plt.figure(figsize=(18, 6))

		dimensions = np.arange(1, 961)
		bars = plt.bar(dimensions, cvs, width=1.0, color='lightgray')

		# Set thresholds
		high_cv_threshold = np.quantile(cvs, 0.95)
		low_cv_threshold = np.quantile(cvs, 0.05)

		# Highlight special bars
		for dim in dimensions:
			idx = dim - 1
			if cvs[idx] > high_cv_threshold:
				bars[idx].set_color('salmon')
		plt.axhline(high_cv_threshold, color='red', linestyle=':', label=f'High CV Threshold ({high_cv_threshold:.2f})')
		plt.axhline(np.median(cvs), color='green', linestyle='--', label=f'Median ({np.median(cvs):.2f})')

		plt.xlabel('Dimension')
		plt.ylabel('CV (σ/μ)')
		plt.xticks(np.arange(0, 961, 50))
		plt.legend()
		plt.grid(axis='y', alpha=0.3)
		plt.tight_layout()

		plt.savefig(f'workflow/results/imported_seq_pdb/{EC_number}/amplify/graphs/cv_dim_embed_{prot_resi}.png', bbox_inches='tight')

		plt.close()

		# Add text annotations for extremes
		high_cv_dims = np.where(cvs > high_cv_threshold)[0] + 1
		low_cv_dims = np.where(cvs < low_cv_threshold)[0] + 1

		# Write to file
		with open(f"workflow/results/imported_seq_pdb/{EC_number}/amplify/info/embeddings_{prot_resi}_stats.txt", "w") as f:
			f.write("=== Most Stable Embeddins Dimensions ===\n")
			low_cv = {}
			for dim in low_cv_dims:
				low_cv[dim] = cvs[dim-1]
			sorted_low_cv = dict(sorted(low_cv.items(), key=lambda item: item[1], reverse=False))
			for k, v in sorted_low_cv.items():
				f.write(f"Dim {k}: CV={v}\n")

			f.write("\n=== Most Variable Dimensions ===\n")
			high_cv = {}
			for dim in high_cv_dims:
				high_cv[dim] = cvs[dim-1]
			sorted_high_cv = dict(sorted(high_cv.items(), key=lambda item: item[1], reverse=True))
			for k, v in sorted_high_cv.items():
				f.write(f"Dim {k}: CV={v}\n")

	return None
