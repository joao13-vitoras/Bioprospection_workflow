import signal
import sys

import os
import gc
import torch
#import shutil
import argparse
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived shutdown signal ({signum}), initiating graceful shutdown...")
        self.shutdown_requested = True

def remove_duplicates(torch_dir):
    duplicates = []
    non_duplicates = []

    processed = set()
    for item in torch_dir.rglob('*.pt'):
        id = item.stem

        if id in processed:
            duplicates.append(item)

        else:
            processed.add(id)
            non_duplicates.append(item)

    return duplicates, non_duplicates

def process_file(file_path):
    seq_id = file_path.stem
    embedding = torch.load(file_path)
    emb_str = ' '.join(f'{x:.30}' for x in embedding.tolist())

    return f">{seq_id}\n{emb_str}\n"

def torch_to_txt(name_db):
    # If there is a need shutdown
    shutdown = GracefulShutdown()

    batch_size= 10000
    pt_dir = f'workflow/results/{name_db}/amplify/embedding'

    prot_torch_dir = Path(f'{pt_dir}/b_prot')
    resi_torch_dir = Path(f'{pt_dir}/b_resi')

    prot_txt = f"workflow/results/{name_db}/amplify/embed_prot.txt"
    resi_txt = f"workflow/results/{name_db}/amplify/embed_resi.txt"

    # Counting and checking for duplicates
    print('Counting files and checking for duplicates ...')
    duplicates_prot, non_duplicate_prot = remove_duplicates(prot_torch_dir)
    duplicates_resi, non_duplicate_resi = remove_duplicates(resi_torch_dir)

    duplicates = len(duplicates_prot) + len(duplicates_resi)

    if duplicates > 0:
        print(f'There is a total of {duplicates} duplicates.')

	# Clean the duplicates
        delete_duplicates = input('Delete duplicates? (y/n)	')
        if delete_duplicates == 'y':
            for file in duplicates_prot:
                file.unlink()

            for file in duplicates_resi:
                file.unlink()

            print(f'All duplicate files has been deleted')

    # Free RAM memory
    del duplicates_prot
    del duplicates_resi
    gc.collect()

    total_files = len(non_duplicate_prot) + len(non_duplicate_resi)

    # Process files
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        bar = tqdm(total=total_files, desc="Converting embeddings")

        # Sequences converted to txt
        print('Marking files that were converted before ...')
        files_amplify = os.listdir(f'workflow/results/{name_db}/amplify')
        if 'embed_prot.txt' in files_amplify:
            seq_id_prot = set()
            for record in SeqIO.parse(f'workflow/results/{name_db}/amplify/embed_prot.txt', "fasta"):
                seq_id_prot.add(record.id)

            if 'embed_resi.txt' in files_amplify:
                seq_id_resi = set()
                for record in SeqIO.parse(f'workflow/results/{name_db}/amplify/embed_resi.txt', "fasta"):
                    seq_id_resi.add(record.id)

        total_converted_files = len(seq_id_prot) + len(seq_id_resi)
        bar.update(total_converted_files)

        # Protein files
        with open(prot_txt, 'a', buffering=8192*16) as prot_file:
            # Filter files: only process if file stem (ID) not in seq_id_prot
            print("Creating a list of the sequences that weren't converted ...")
            unconverted_prot_files = [f for f in non_duplicate_prot if f.stem not in seq_id_prot]

            del non_duplicate_prot
            del seq_id_prot
            gc.collect()

            print('Running the extrator ...')
            futures = {executor.submit(process_file, f): f for f in unconverted_prot_files}

            batch = []
            for future in concurrent.futures.as_completed(futures):
                batch.append(future.result())

                if len(batch) >= batch_size:
                    prot_file.writelines(batch)
                    bar.update(len(batch))
                    batch = []
                    gc.collect()

            if batch:
                prot_file.writelines(batch)
                bar.update(len(batch))
                del non_duplicate_prot
                gc.collect()

        # Residues files
        with open(resi_txt, 'a', buffering=8192*16) as resi_file:
            print("Creating a list of the sequences that weren't converted ...")
            unconverted_resi_files = [f for f in non_duplicate_resi if f.stem not in seq_id_resi]

            del non_duplicate_resi
            del seq_id_resi
            gc.collect()

            print('Running the extrator ...')
            futures = {executor.submit(process_file, f): f for f in unconverted_resi_files}

            batch = []
            for future in concurrent.futures.as_completed(futures):
                batch.append(future.result())

                if len(batch) >= batch_size:
                    resi_file.writelines(batch)
                    bar.update(len(batch))
                    batch = []
                    gc.collect()

            if batch:
                resi_file.writelines(batch)
                bar.update(len(batch))


        pbar.close()
#    shutil.rmtree(input_dir)

    return None



def parse_args():
    ap = argparse.ArgumentParser(description="AMPLIFY embedded extraction script")
    ap.add_argument("--db_name", required=True, help="Name of the database")
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    torch_to_txt(args.db_name)
