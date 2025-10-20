import os
import torch
import shutil
from tqdm import tqdm

def torch_to_txt(input_dir, name_db):
    prot_torch_dir = input_dir + '/b_prot'
    resi_torch_dir = input_dir + '/b_resi'

    prot_txt = f"workflow/results/{name_db}/amplify/embed_prot.txt"
    resi_txt = f"workflow/results/{name_db}/amplify/embed_resi.txt"

    # Count total files
    prot_count = len([f for f in os.listdir(prot_torch_dir) if f.endswith(".pt")])
    resi_count = len([f for f in os.listdir(resi_torch_dir) if f.endswith(".pt")])
    total_files = prot_count + resi_count

    pbar = tqdm(total=total_files, desc="Converting embeddings")

    with open(prot_txt, 'w') as prot_file:
        for filename in sorted(os.listdir(prot_torch_dir)):
            if filename.endswith(".pt"):
                seq_id = os.path.splitext(filename)[0]
                embedding = torch.load(os.path.join(prot_torch_dir, filename))
                emb_str = ' '.join(f'{x:.30}' for x in embedding.tolist())
                prot_file.write(f">{seq_id}\n{emb_str}\n")
                pbar.update(1)
                pbar.set_postfix({"current": f"prot/{filename}"})

    with open(resi_txt, 'w') as resi_file:
        for filename in sorted(os.listdir(resi_torch_dir)):
            if filename.endswith(".pt"):
                seq_id = os.path.splitext(filename)[0]
                embedding = torch.load(os.path.join(resi_torch_dir, filename))
                emb_str = ' '.join(f"{x:.30}" for x in embedding.tolist())
                resi_file.write(f">{seq_id}\n{emb_str}\n")
                pbar.update(1)
                pbar.set_postfix({"current": f"resi/{filename}"})

    pbar.close()

    shutil.rmtree(input_dir)

    return None
