import os
import sys
import signal
import argparse
from types import SimpleNamespace
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from functools import partial
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from Bio import SeqIO
from tqdm import tqdm

# ------------------------------
# Helpers
# ------------------------------

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("Received interrupt signal, cleaning up ...")
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def setup_device_from_env():
    """Pick the CUDA device based on torchrun env vars.
       Initialize distributed.
    torchrun / torch.distributed.launch sets the following environment variables for
    each spawned process:
      - LOCAL_RANK: the GPU id local to the node (0..n-1)
      - RANK: global rank across nodes
      - WORLD_SIZE: total number of processes

    This function reads those and sets the active CUDA device accordingly.
    """

    local_rank = get_env_int("LOCAL_RANK", 0)
    world_size = get_env_int("WORLD_SIZE", 1)
    rank = get_env_int("RANK", 0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU extraction.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize distributed if needed
    dist_initialized = False
    if world_size > 1 and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
            )
            dist_initialized = True
        except Exception as e:
            print(f"Distributed initialization failed: {e}")

    return device, local_rank, rank, world_size, dist_initialized


# ------------------------------
# Data pipeline
# ------------------------------


class FASTAStream(IterableDataset):
    """
    Stream records from a FASTA file, filter by length, and shard across ranks.
    """

    def __init__(self, fasta_path: str, min_len: int, max_len: int, world_size: int, rank: int):
        super().__init__()
        self.fasta_path = fasta_path
        self.min_len = min_len
        self.max_len = max_len
        self.world_size = world_size
        self.rank = rank

    def parse(self) -> Iterator[Tuple[str, str]]:
        idx = 0
        for rec in SeqIO.parse(self.fasta_path, "fasta"):
            # Simple deterministic sharding: take every world_size-th record offset by rank
            if (idx % self.world_size) == self.rank:
                L = len(rec.seq)
                if self.min_len <= L <= self.max_len:
                    yield (rec.id, str(rec.seq))
            idx += 1

    def __iter__(self):
        return self.parse()


def collate_and_tokenize(batch, tokenizer, device):
    ids, seqs = zip(*batch)
    encoded = tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    return list(ids), encoded

def collate_with_tokenizer(batch, tokenizer, device):
        return collate_and_tokenize(batch, tokenizer, device)

# ------------------------------
# Core extraction (internal runner)

@torch.inference_mode()
def _run(args):
    try:
        device, local_rank, rank, world_size, dist_initialized = setup_device_from_env()

        mp.set_start_method("spawn")

        # Fragmentation problems
        torch.cuda.set_per_process_memory_fraction(0.9, device=device)

        # Resolve output directories. To avoid write contention, each rank writes to its own subdir.
        root_out = os.path.join("workflow", "results", args.name_db, 'amplify', "embedding")
        out_b_prot = os.path.join(root_out, "b_prot", f"rank{rank}")
        out_b_resi = os.path.join(root_out, "b_resi", f"rank{rank}")
        os.makedirs(out_b_prot, exist_ok=True)
        os.makedirs(out_b_resi, exist_ok=True)

        # Load model & tokenizer (one copy per GPU)
        tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True)
        dtype = torch.float16 if args.fp16 else torch.float32
        model = (AutoModel.from_pretrained("chandar-lab/AMPLIFY_350M", trust_remote_code=True, dtype=dtype).eval().to(device))

        hidden = getattr(model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(model.config, "n_embd", None)
        if hidden is None:
            raise RuntimeError("Could not infer hidden size from model config.")

        layer_norm = nn.LayerNorm(hidden, elementwise_affine=True).to(device)

        zero_tensor = torch.tensor(0.0, device='cuda', dtype=torch.float32)
        neg_inf = torch.tensor(float('-inf'), device='cuda', dtype=torch.float32)

        # Build streaming dataset and dataloader
        ds = FASTAStream(
            fasta_path= args.database,
            min_len=args.init_len,
            max_len=args.end_len,
            world_size=world_size,
            rank=rank,
        )

        collate_fn = partial(collate_with_tokenizer, tokenizer=tokenizer, device=device)

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=bool(args.workers > 0),
        )

        pbar = None
        if rank == 0:
            pbar = tqdm(total=None, dynamic_ncols=True, desc="Extracting embeddings", unit="batch")

        for ids, enc in dl:
	    # Move to GPU
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

            attn_bool = enc["attention_mask"].bool()

            enc["attention_mask"] = torch.where(attn_bool, zero_tensor, neg_inf)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                out = model(**enc, output_hidden_states=False)
                # Access path for hidden states
                if hasattr(out, "last_hidden_state"):
                    reps = out.last_hidden_state
                else:
                    reps = out

                reps = layer_norm(reps)

            input_ids = enc["input_ids"]

            # Derive a mask to drop special tokens (BOS/EOS/CLS/SEP/PAD)
            # Using tokenizer's special token ids.
            special_ids = set(
                tid for tid in [
                    tokenizer.cls_token_id,
                    tokenizer.sep_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.bos_token_id,
                    tokenizer.eos_token_id,
                ]
                if tid is not None)

            specials = torch.isin(input_ids, torch.tensor(list(special_ids), device=input_ids.device)) if special_ids else torch.zeros_like(input_ids, dtype=torch.bool)

            valid_mask = attn_bool & (~specials)  # [B, T]

            # Save one embedding per sequence (e.g., first token) and mean over residue tokens
            for i, seq_id in enumerate(ids):
                seq_len = int(attn_bool[i].sum().item())

                # per-protein embedding (use first token)
                prot_embed = reps[i, 0].detach().to("cpu")

                tokens = reps[i][valid_mask[i]]
                if tokens.size(0) > 0:
                    resi_embed = tokens.mean(dim=0).detach().to("cpu")
                else:
                    resi_embed = reps[i][1:seq_len].mean(dim=0).detach().to("cpu")

                torch.save(prot_embed, os.path.join(out_b_prot, f"{seq_id}.pt"))
                torch.save(resi_embed, os.path.join(out_b_resi, f"{seq_id}.pt"))

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    except Exception as e:
        print(f"Rank {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        torch.cuda.empty_cache()
        if dist_initialized:
            dist.destroy_process_group()
        print(f"Rank {rank} cleanup complete")

    return None

# ------------------------------
# Public wrapper function
# ------------------------------
import timeit

def extract_embeddings_multi(database: str, batch_size: int, init_len: int, end_len: int, workers: int = 1, fp16: bool = True):
    """
    Full multi-GPU pipeline:
    - Sort FASTA sequences
    - Compute database statistics
    - Extract embeddings (multi-GPU with torchrun)
    - Convert torch .pt embeddings to text
    - Compute embeddings statistics
    """

    start = timeit.default_timer()

    n_db = database.split('/')[2]

    args = SimpleNamespace()
    args.database = database
    args.name_db = n_db
    args.batch_size = batch_size
    args.init_len = init_len
    args.end_len = end_len
    args.workers = workers
    args.fp16 = fp16

    rank = int(os.environ.get("RANK", "0"))

    # _run expects the torchrun environment variables to be present. If you call this
    # function without torchrun, it will default to a single-process run on GPU 0.
    _run(args)

    if rank == 0:
        end = timeit.default_timer()

        t_min, t_sec = divmod(end-start,60)
        t_hour,t_min = divmod(t_min,60)

        print(f'Extraction duration = {int(t_hour)} hour:{int(t_min)} min:{int(t_sec)} sec')

    return None

# ------------------------------
# CLI (optional)
# ------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-GPU AMPLIFY embedding extraction (single node)")
    ap.add_argument("--database", required=True, help="Path to FASTA database file")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--init_len", type=int, default=1, help="Minimum sequence length to include")
    ap.add_argument("--end_len", type=int, default=2046, help="Maximum sequence length to include")
    ap.add_argument("--workers", type=int, default=1, help="DataLoader workers (per process)")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 for faster inference")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_embeddings_multi(args.database, args.batch_size, args.init_len, args.end_len, args.workers, args.fp16)
