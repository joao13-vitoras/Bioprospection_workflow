import argparse
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

def measure_max_batch_size(sequence_length):
    """Measure maximum batch size for a given sequence length by incrementally increasing batch size"""

    # Initialize model
    model_name = "chandar-lab/AMPLIFY_350M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to('cuda')

    # Generate test sequence
    test_sequence = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=sequence_length))

    # Testing parameters
    batch_size = 1
    max_tested_batch_size = 0
    success = True
    batch_sizes = []
    results = []

    while success:
        try:
            zero_tensor = torch.tensor(0.0, device='cuda')
            neg_inf = torch.tensor(float('-inf'), device='cuda')
            with torch.no_grad():
                # Create batch
                batch = [test_sequence] * batch_size

                # Tokenize
                encoded = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=False,
                    return_attention_mask=True
                ).to('cuda')

                # Convert attention mask to AMPLIFY format
                input_ids = encoded['input_ids'].to('cuda')
                binary_attention_mask = encoded['attention_mask'].to('cuda')
                attention_mask = torch.where(binary_attention_mask.bool(), zero_tensor, neg_inf)

                _ = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

                # Wait for the result
                torch.cuda.synchronize()

                max_tested_batch_size = batch_size
                batch_sizes.append(batch_size)
                results.append(True)

                # Double the batch size for next test
                batch_size *= 2

                # Clear memory
                del encoded, attention_mask
                torch.cuda.empty_cache()

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"Error: {str(e)}")
            success = False
            results.append(False)
            batch_sizes.append(batch_size)

            # Clear memory
            torch.cuda.empty_cache()

    return max_tested_batch_size, batch_sizes, results

def parse_args():
        ap = argparse.ArgumentParser(description="Test maximum size of batch for the GPU")
        ap.add_argument("--seq_len",type=int, required=True, help="Maximum size of amino acid sequence for analysis in the amplify embedding extraction")
        return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    max_batch, sizes, results = measure_max_batch_size(args.seq_len)

    # Print summary
    print("\nTest Summary:")
    for size, success in zip(sizes, results):
        print(f"Batch size {size}: {'Success' if success else 'Failed'}")
