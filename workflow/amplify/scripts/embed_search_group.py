import os
import time
from Bio import SeqIO
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def chunked(iterable, batch_size):
	for i in range(0, len(iterable), batch_size):
		yield iterable[i:i + batch_size]

def write_queue_to_disk(queue, out_prot_dir, out_resi_dir, seq_counter):
	"""Helper function to write queued embeddings to disk"""
	for seq_id, prot_embed, resi_embed in queue:
		# Save consensus
		if 'consensus' in seq_id:
			save_dir = out_prot_dir.split('/')
			save_dir = f'{save_dir[0]}/{save_dir[1]}/{save_dir[2]}/{save_dir[3]}'

			torch.save(prot_embed, f'{save_dir}/consensus_prot.pt')
			torch.save(resi_embed, f'{save_dir}/consensus_resi.pt')

		# Other sequences
		else:
			torch.save(prot_embed, f'{out_prot_dir}/{seq_id}.pt')
			torch.save(resi_embed, f'{out_resi_dir}/{seq_id}.pt')

		if seq_counter:
			seq_counter.update(1)

def extract_embeddings_batch(sorted_search_group, EC_number, total_seq, size_batch, gpu_id):
	# Output directories
	out_prot = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/embedding/b_prot/rank{gpu_id}'
	out_resi = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/embedding/b_resi/rank{gpu_id}'

	os.makedirs(out_prot, exist_ok=True)
	os.makedirs(out_resi, exist_ok=True)

	# Check for existing embeddings to resume
	existing_embeddings = set()
	if os.path.exists(out_prot):
		existing_embeddings = {f.split('.')[0] for f in os.listdir(out_prot) if f.endswith('.pt')}

	# Device and avoid memory errors
	device = torch.device(f'cuda:{gpu_id}')
	torch.cuda.set_per_process_memory_fraction(0.9, device=device)

	try:
		# Load model & tokenizer
		tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
		model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True).eval().to(device)

		hidden_size = model.config.hidden_size
		layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True).to(device)

		zero_tensor = torch.tensor(0.0, device=device)
		neg_inf = torch.tensor(float('-inf'), device=device)

		# Create tqdm progress bar for output files
		if gpu_id == 0:
			seq_counter = tqdm(total= (total_seq), desc=f"Sequences processed", unit="seq", position=1, leave=False, dynamic_ncols=True)
			seq_counter.update(len(existing_embeddings))

		elif gpu_id != 0:
			seq_counter = None

		sequences = []
		if os.path.exists(sorted_search_group):
			for record in SeqIO.parse(sorted_search_group, 'fasta'):
				if record.id in existing_embeddings:
					continue

				else:
					sequences.append((record.id, str(record.seq)))

		# Include consensus sequence
		if gpu_id == 0:
			for record in SeqIO.parse(f'workflow/results/imported_seq_pdb/{EC_number}/consensus.fa', 'fasta'):
				sequences.append((record.id, str(record.seq)))

		# Process
		process_sequences(sequences, model, tokenizer, device, layer_norm,
					zero_tensor, neg_inf, size_batch, out_prot, out_resi, seq_counter)

	finally:
		# Always cleanup
		torch.cuda.empty_cache()
		if gpu_id == 0:
			seq_counter.close()

	return None

@torch.inference_mode()
def process_sequences(sequences, model, tokenizer, device, layer_norm,
				zero_tensor, neg_inf, batch_size, out_prot, out_resi, seq_counter):

	# Initialize write buffer
	write_queue = []

	# GPU temperature control
	resume_temp = 70
	cooldown_delay = 10
	cooldown = False

	for batch in chunked(sequences, batch_size):
		# Temperature control
		temp = torch.cuda.temperature(device)
		if  temp >= 90:
			cooldown = True
		while cooldown:
			time.sleep(cooldown_delay)
			temp = torch.cuda.temperature(device)
			if temp <= resume_temp:
				cooldown = False
				break

		batch_ids, batch_seqs = zip(*batch)

       		# Tokenize entire batch
		encoded = tokenizer(batch_seqs,	return_tensors='pt', padding=True, truncation=True, return_attention_mask=True)

       		# Move to GPU
		input_ids = encoded['input_ids'].to(device, non_blocking=True)
		binary_attention_mask = encoded['attention_mask'].to(device, non_blocking=True)
		attention_mask = torch.where(binary_attention_mask.bool(), zero_tensor, neg_inf)

		output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

		embeddings = layer_norm(output)

		# Synchronize after model forward pass
		torch.cuda.synchronize(device)

		for idx in range(len(batch)):
			seq_id = batch_ids[idx]
			seq_len = int(binary_attention_mask[idx].sum().item())

       			# Extract valid token embeddings: remove BOS and EOS
			residue_embeds = embeddings[idx, 1:(seq_len-1)]

			# Save results in buffer
			write_queue.append((seq_id,
				embeddings[idx, 0].detach().clone().to('cpu'),
				residue_embeds.mean(dim=0).detach().clone().to('cpu')))

			# Write queque in disk
			if len(write_queue) >= (10*batch_size):
				write_queue_to_disk(write_queue, out_prot, out_resi, seq_counter)
				write_queue.clear()

		if write_queue:
			write_queue_to_disk(write_queue, out_prot, out_resi, seq_counter)

	return None
