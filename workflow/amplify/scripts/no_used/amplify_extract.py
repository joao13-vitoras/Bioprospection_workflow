import torch
import os
from torch import nn
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm
import timeit

# Input sequences
database = input('Path to Amplify database:	')
sequences = []
for record in SeqIO.parse(database, 'fasta'):
	sequences.append((record.id, str(record.seq)))

# Load AMPLIFY and tokenizer
model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)

# Layer normalization
hidden_size = model.config.hidden_size
layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True).to('cuda')

# Move the model to GPU
model = model.to('cuda')
model.eval()

# Output file
name_db = database.split('/')
name_db = name_db[-1].split('.')

# Create output directory
os.makedirs('results_amplify/{}/embedding/e_prot'.format(name_db[0]), exist_ok=True)
os.makedirs('results_amplify/{}/embedding/e_resi'.format(name_db[0]), exist_ok=True)

# Start
start = timeit.default_timer()

# Embedding extraction
with torch.no_grad():
	for i in tqdm(sequences, desc="Processing sequences", leave=False):
		tokens = tokenizer.encode(i[1], truncation=True, return_tensors='pt').to('cuda')

		output = model(tokens, output_hidden_states= False)

		embeddings = layer_norm(output)

		residues_embeddings = embeddings[1:-1]
		mean_residues_embed = residues_embeddings.mean(dim=0)

		protein_embeddings = embeddings[0]

		# Save protein embeddings
		torch.save(protein_embeddings.cpu(), f"results_amplify/{name_db[0]}/embedding/e_prot/{i[0]}.pt")

		# Save mean residues embeddings
		torch.save(mean_residues_embed.cpu(), f"results_amplify/{name_db[0]}/embedding/e_resi/{i[0]}.pt")

# End
end = timeit.default_timer()
print(end-start)
