import os
from Bio import SeqIO
import numpy as np
import tempfile

def sort_in_chunks(search_group_path: str, EC_number: str, num_gpus: int):
    """
    Distribute sequences by total amino acid count for better load balancing,
    then sort each file by sequence length in descending order.
    """
    print(f'Running: sort_in_chunks')

    out_dir = f'workflow/results/imported_seq_pdb/{EC_number}/amplify/sorted_sequences'
    os.makedirs(out_dir, exist_ok=True)

    # Create output files and track their total amino acid counts
    output_files = []
    aa_counts = np.zeros(num_gpus, dtype=np.int64)  # Track total AA count per file

    # Open all output files
    for i in range(num_gpus):
        output_path = os.path.join(out_dir, f'unsorted_{i}.fa')
        output_files.append(open(output_path, 'w'))

    # Process sequences one by one and distribute by AA count
    for record in SeqIO.parse(search_group_path, "fasta"):
        seq_len = len(record.seq)

        # Find the file with the smallest total amino acid count
        target_file_idx = np.argmin(aa_counts)

        # Write sequence to the selected file
        output_files[target_file_idx].write(f">{record.id}\n{str(record.seq)}\n")

        # Update the amino acid counter for this file
        aa_counts[target_file_idx] += seq_len

    # Close all output files
    for f in output_files:
        f.close()

    # Sort each file by sequence length in chunks
    sort_files_by_length(out_dir, num_gpus)

    return None

def sort_files_by_length(out_dir: str, num_files: int):
    """Sort files by sequence length in descending order"""

    for i in range(num_files):
        input_path = os.path.join(out_dir, f'unsorted_{i}.fa')
        output_path = os.path.join(out_dir, f'sorted_{i}.fa')

        # Check if file exists and has content
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            print(f"Warning: File {input_path} is empty or doesn't exist")
            continue

        with open(output_path, "w") as fout:
            seq = []
            for record in SeqIO.parse(input_path, "fasta"):
                seq.append([len(record.seq), (record.id).split('|')[0], record.seq])

            # Sort by length in descending order
            seq.sort(key=lambda x: x[0], reverse=True)

            for rec in seq:
                fout.write(f">{rec[1]}\n{str(rec[2])}\n")

        # Remove unsorted files
        os.remove(input_path)

    return None
