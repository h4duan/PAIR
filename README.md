# PAIR: Protein Annotation-Improved Representations

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-red)](https://www.biorxiv.org/content/10.1101/2024.07.22.604688v2.abstract)


This repository contains the implementation of "Boosting the Predictive Power of Protein Representations with a Corpus of Text Annotations" (2024).

![PAIR](pair.png)


## Highlights

- Fine-tuning framework for protein language models
- Utilizes 19 types of text annotations from UniProt
- Transformer-based encoder-decoder architecture
- Improved performance over BLAST baseline
- Pre-computed vector representations for efficient retrieval

## File structures

### `_config/`
- `fact_types.yml`: Hyperparameters for parsing and loading each fact type
- `paths.yml`: Paths for datasets

### `_evaluation/`
- `get_embeddings.py`: Generate embeddings for each model
- `knn_embeddings.py`: Perform knn on the precomputing embedding

### `_fact/`
- `_/parse.py`: parser code for each fact type
- `_/batch.py`: Batch operation for each fact type

### `_model/`
- `_/model.py`: implementations of the seq2seq architecture
- `_/config.py`: Hyperparameters of the model
- `dataloader_ddp.py`: General implementations of the data loader




