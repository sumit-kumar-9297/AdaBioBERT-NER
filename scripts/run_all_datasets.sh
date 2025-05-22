#!/bin/bash

# Run training on all datasets with the specified configuration
#!/bin/bash

# Configuration
DATA_DIR="/path_to_data_dir"
WORD2VEC_PATH="/path_to_word2vec.model"
OUTPUT_DIR="../experiments"

# Common parameters
COMMON_PARAMS="--data_dir $DATA_DIR --word2vec_path $WORD2VEC_PATH --output_dir $OUTPUT_DIR"
TRAINING_PARAMS="--batch_size 32 --eval_batch_size 32 --max_seq_length 64 --num_epochs 20 --gradient_accumulation_steps 32 --use_amp"

# List of datasets
DATASETS=(
    "BC4CHEMD"
    "linnaeus"
    "NCBI-disease"
    "BC5CDR"
    "JNLPBA"
    "AnatEM"
    "BioNLP13GE"
    "s800"
)

# Create log directory
mkdir -p logs

# Run training for each dataset
for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing dataset: $dataset"
    echo "========================================"
    
    # Run training
    python3 ../src/train.py --dataset $dataset $COMMON_PARAMS $TRAINING_PARAMS 2>&1 | tee logs/${dataset}_train.log
    
    # Optional: sleep between datasets to prevent GPU overload
    sleep 10
done

echo "All datasets processed!"
