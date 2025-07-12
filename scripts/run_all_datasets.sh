#!/bin/bash

# Run training on all datasets with the fixed Word2Vec integration
#!/bin/bash

# Configuration
DATA_DIR="path/to/data/ner"
WORD2VEC_PATH="path/to/word2vec.model"
OUTPUT_DIR="../experiments"

# Common parameters
COMMON_PARAMS="--data_dir $DATA_DIR --word2vec_path $WORD2VEC_PATH --output_dir $OUTPUT_DIR"

# Training parameters - optimized for the fixed implementation
TRAINING_PARAMS="--batch_size 16 --eval_batch_size 16 --max_seq_length 64 --num_epochs 20 --gradient_accumulation_steps 32 --use_amp"

# Model parameters - added fine-tuning options
MODEL_PARAMS="--biobert_model dmis-lab/biobert-v1.1 --alpha_init 0.5 --dropout_rate 0.3 --freeze_bert_layers 6 
--word2vec_finetune_epochs 5 --word2vec_min_count 1"

# Learning rate parameters
LR_PARAMS="--learning_rate 2e-5 --alpha_lr 0.01 --classifier_lr 1e-4 --weight_decay 1e-4"

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



# Function to check if required files exist
check_prerequisites() {
    echo "Checking prerequisites..."
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    if [ ! -f "$WORD2VEC_PATH" ]; then
        echo "ERROR: Word2Vec model not found: $WORD2VEC_PATH"
        exit 1
    fi
    
    # Check if Python packages are available
    python3 -c "import torch, transformers, sklearn, gensim, TorchCRF" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Required Python packages not found. Please install:"
        echo "  pip install torch transformers scikit-learn gensim TorchCRF tqdm matplotlib"
        exit 1
    fi
    
    echo "✓ All prerequisites satisfied"
}

# Function to run training for a single dataset
train_dataset() {
    local dataset=$1
    local start_time=$(date +%s)
    
    echo "========================================"
    echo "Processing dataset: $dataset"
    echo "Start time: $(date)"
    echo "========================================"
    
    # Check if dataset directory exists
    if [ ! -d "$DATA_DIR/$dataset" ]; then
        echo "WARNING: Dataset directory not found: $DATA_DIR/$dataset"
        echo "Skipping $dataset"
        return 1
    fi
    
    # Check if required files exist
    required_files=("train.tsv" "devel.tsv" "test.tsv")
    for file in "${required_files[@]}"; do
        if [ ! -f "$DATA_DIR/$dataset/$file" ]; then
            echo "WARNING: Required file not found: $DATA_DIR/$dataset/$file"
            echo "Skipping $dataset"
            return 1
        fi
    done
    
    # Run training with all parameters
    python3 ../src/train.py \
        --dataset $dataset \
        $COMMON_PARAMS \
        $TRAINING_PARAMS \
        $MODEL_PARAMS \
        $LR_PARAMS \
        --seed 42 \
        --log_interval 10 \
        --save_model_every 20 \
        2>&1 | tee logs/${dataset}_train.log
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed $dataset in ${duration}s"
        echo "Results saved to: $OUTPUT_DIR/${dataset}_*"
    else
        echo "✗ Failed to train $dataset (exit code: $exit_code)"
        return $exit_code
    fi
    
    echo ""
}

# Main execution
main() {
    # Check prerequisites
    check_prerequisites
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Track overall progress
    local total_datasets=${#DATASETS[@]}
    local completed=0
    local failed=0
    local overall_start=$(date +%s)
    
    echo "Starting training on $total_datasets datasets..."
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    
    # Run training for each dataset
    for dataset in "${DATASETS[@]}"; do
        train_dataset "$dataset"
        
        if [ $? -eq 0 ]; then
            ((completed++))
        else
            ((failed++))
        fi
        
        # Optional: sleep between datasets to prevent GPU overload
        if [ $completed -lt $total_datasets ]; then
            echo "Waiting 10 seconds before next dataset..."
            sleep 10
        fi
    done
    
    # Final summary
    local overall_end=$(date +%s)
    local total_duration=$((overall_end - overall_start))
    
    echo "========================================"
    echo "TRAINING SUMMARY"
    echo "========================================"
    echo "Total datasets: $total_datasets"
    echo "Completed successfully: $completed"
    echo "Failed: $failed"
    echo "Total time: ${total_duration}s"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    if [ $failed -eq 0 ]; then
        echo "🎉 All datasets processed successfully!"
    else
        echo "⚠️  Some datasets failed. Check logs for details."
        exit 1
    fi
}

# Run main function
main "$@"
