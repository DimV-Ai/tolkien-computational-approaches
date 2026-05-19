# Tolkien Computational Approaches

This repository contains computational notebooks for experiments in literary NLP, focusing on sentence embeddings and sentiment-related evaluation in *The Lord of the Rings*.

At this stage, the repository contains one cleaned evaluation notebook.

## Contents

- `notebooks/01_model_evaluation.ipynb`

## Notebook: `01_model_evaluation.ipynb`

This notebook evaluates whether sentence-transformer models can distinguish between manually selected positive and negative sentiment examples from *The Lord of the Rings*.

The notebook compares:

- a general-purpose MiniLM sentence-transformer baseline;
- a locally stored Tolkien-adapted sentence-transformer model;
- Tolkien-adapted models fine-tuned for different numbers of epochs.

The evaluation uses positive and negative reference sentences, referred to as **anchor sentences**, and a separate set of held-out test sentences. The anchor sentences are used by the scoring procedure, while the test sentences are used only for evaluation.

The notebook includes:

- loading and validating the evaluation CSV;
- separating anchor and held-out test sentences;
- evaluating models with a nearest-centroid anchor-based procedure;
- reporting accuracy, precision, recall, F1, confusion matrices, and classification reports;
- comparing Tolkien-adapted models across epoch settings;
- comparing the baseline and selected Tolkien-adapted model using McNemar's test;
- a qualitative retrieval comparison between the baseline and selected Tolkien-adapted model.

## Data

The full text data and manually selected sentence examples are **not included** in this repository because they are drawn from copyrighted literary text.

To run the notebook locally, provide an evaluation CSV with these columns:

- `sentence_text`: the sentence to evaluate;
- `gold_label`: the manually assigned label, either `positive` or `negative`;
- `split`: the role of the sentence, either `anchor` or `test`.

The `split` column separates:

- `anchor`: reference examples used by the scoring procedure;
- `test`: held-out examples used only for evaluation.

For the retrieval section, the notebook expects a local sentence-per-line corpus file. This file is also not included in the repository.

## Models

The baseline model is loaded from Hugging Face:

- `sentence-transformers/all-MiniLM-L6-v2`

The Tolkien-adapted models are expected to be stored locally. The notebook currently uses paths such as:

- `../models/tolkien_sentence_transformer`
- `../models/tolkien_sentence_transformer_epoch_1`
- `../models/tolkien_sentence_transformer_epoch_2`
- `../models/tolkien_sentence_transformer_epoch_4`
- `../models/tolkien_sentence_transformer_epoch_8`

These paths can be edited inside the notebook.

## Environment

The notebook requires the following main packages:

- `sentence-transformers`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`

## Copyright note

Notebook outputs are cleared before committing because local runs may display copyrighted source text. The repository shares the code and workflow structure, but does not redistribute the underlying literary text.