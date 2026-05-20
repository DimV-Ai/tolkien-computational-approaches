# Tolkien Computational Approaches

This repository contains computational notebooks for experiments in literary NLP, focusing on sentence embeddings, anchor-based sentiment evaluation, and sentiment trajectory analysis in *The Lord of the Rings*.

The repository currently contains two cleaned notebooks:

## Contents

- `notebooks/01_model_evaluation.ipynb`
- `notebooks/02_anchored_sentiment_analysis.ipynb`

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

## Notebook: `02_anchored_sentiment_analysis.ipynb`

This notebook computes and visualises an anchored sentiment trajectory over a literary text.

The notebook includes:

- optional preprocessing templates for TEI/XML and HTML sources;
- a prepared plain-text parser for the local corpus format used in this project;
- dialogue-aware segmentation while preserving chapter boundaries;
- construction of a standard `segments_df` dataframe;
- loading local positive and negative anchor sentences from `data/anchors.csv`;
- computing anchored sentiment scores for each segment;
- smoothing the trajectory with a rolling window;
- visualising the full sentiment trajectory.

The scoring method computes:

`sentiment_score = mean_similarity_to_positive_anchors - mean_similarity_to_negative_anchors`

Positive scores indicate that a segment is closer to the positive anchors, while negative scores indicate that it is closer to the negative anchors.

## Data

The full text data and manually selected sentence examples are **not included** in this repository because they are drawn from copyrighted literary text.

To run `01_model_evaluation.ipynb`, provide an evaluation CSV with these columns:

- `sentence_text`: the sentence to evaluate;
- `gold_label`: the manually assigned label, either `positive` or `negative`;
- `split`: the role of the sentence, either `anchor` or `test`.

To run `02_anchored_sentiment_analysis.ipynb`, provide:

- a local corpus file, such as `data/LotR.txt`;
- a local anchor CSV file, such as `data/anchors.csv`;
- a local sentence-transformer model path.

The anchor CSV should contain:

- `sentence_text`: the anchor sentence;
- `label`: either `positive` or `negative`.

The full corpus text and anchor sentences are not distributed with this repository.

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