#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hmmer

read -p 'Path to database:      ' db
read -p 'Path to search group directory:      ' search_group_dir

python3 ~/workflow/hmmer/run.py \
  --database $db \
  --search_group_dir $search_group_dir

conda deactivate


conda activate diamond

python3 ~/workflow/diamond/run.py \
  --database $db \
  --search_group_dir $search_group_dir

conda deactivate


read -p 'Minimum length of amino acid sequence for amplify analysis:      ' init_len
read -p 'Maximum length of amino acid sequence for amplify analysis:      ' end_len

conda activate amplify

gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Number of NVIDIA GPUs: $gpu_count"

# Solve GPU memory Errors
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

if [ $gpu_count -ge 1 ]; then

    python3 ~/workflow/amplify/search_group_embed.py \
      --search_group $search_group_dir

    python3 ~/workflow/amplify/database_embed.py \
      --database $db \
      --init_len $init_len \
      --end_len $end_len

else
    echo "No Nvidia GPU"
fi

conda deactivate


conda activate faiss

python3 ~/workflow/amplify/cosine_similarity.py \
  --search_group_dir $search_group_dir  \
  --database $db

conda deactivate
