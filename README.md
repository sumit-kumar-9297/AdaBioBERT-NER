## BioBERT Adaptive Loss NER Repository

This repository contains code for Named Entity Recognition (NER) using BioBERT with an Adaptive Token-Sequence Loss function.

### Features

- BioBERT + Word2Vec embeddings for NER in biomedical texts
- Adaptive loss weighting between token-level cross-entropy and sequence-level CRF loss
- Memory-optimized training for large datasets
- Command-line configurable for all parameters

### Repository Structure

```
biobert-adaptive-ner/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── train.py (main script)
│   ├── model.py (model definition)
│   ├── loss.py (adaptive loss implementation)
│   ├── data_utils.py (data processing)
│   └── utils.py (utility functions)
|   └── word2vec_embedder.py
├── scripts/
│   └── run_all_datasets.sh (batch processing script)
└── configs/
    └── default_config.json (default configuration)
```

### Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Run training: `python src/train.py --dataset BC4CHEMD --data_dir /path/to/data --word2vec_path /path/to/model`
3. Process all datasets: `bash scripts/run_all_datasets.sh`

### Command Line Arguments

See `python src/train.py --help` for all available command line options.
