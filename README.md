# Tolkien Computational Approaches

This repository contains computational notebooks, reusable scripts, and configuration files for experiments in literary NLP, focusing on sentence embeddings, anchor-based sentiment evaluation, sentiment trajectory analysis, semantic-axis modelling, and topic-distance analysis in *The Lord of the Rings*.

The repository currently contains three cleaned notebooks:

## Contents

- `notebooks/01_model_evaluation.ipynb`  
  Held-out evaluation of MiniLM and Tolkien-adapted sentence-transformer models.

- `notebooks/02_anchored_sentiment_analysis.ipynb`  
  Anchored sentiment trajectory analysis over segmented literary text.
  
- `notebooks/03_semantic_axis_exploration.ipynb`
  Semantic-axis exploration using Light–Dark anchor examples and sentence-transformer embeddings.

- `src/`  
  Reusable Python code for topic-modelling-related experiments, including BERTopic topic-distance analysis.

- `configs/`  
  Configuration files for reproducible experiment settings, including BERTopic/topic-distance parameters.

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

This trajectory notebook uses a default rolling smoothing window of 20 segments. This is a visualisation parameter and should be reported when using the generated figures.

The current evaluation notebook uses sentence-level held-out examples. Segment-level evaluation is planned as a future extension because the downstream trajectory analysis operates on merged text segments.

## Notebook: `03_semantic_axis_exploration.ipynb`

This notebook explores semantic opposition axes in literary text using sentence-transformer embeddings.

The current notebook focuses on a Light–Dark semantic axis. The axis is constructed from two sets of local anchor sentences: one representing the Light pole and one representing the Dark pole. Each pole is represented by a centroid in embedding space. The semantic axis is then defined as the direction between these two centroids, and text segments are projected onto that axis.

This differs from the anchored sentiment scoring procedure in `02_anchored_sentiment_analysis.ipynb`. In notebook 02, the score is computed as the difference between mean similarity to positive and negative sentiment anchors. In notebook 03, the score is computed by projecting each segment embedding onto a constructed semantic axis.

The notebook includes:

- loading local Light and Dark axis-anchor sentences from `data/light_dark_anchors.csv`;
- encoding semantic-axis anchors with a sentence-transformer model;
- computing Light and Dark centroids;
- defining the Light–Dark semantic axis;
- projecting text segments onto the axis;
- inspecting anchor coherence and possible outliers;
- smoothing and visualising the resulting semantic-axis trajectory.

Positive scores indicate movement toward the Light pole, while negative scores indicate movement toward the Dark pole.

## Scoring methods

This repository uses two related but mathematically different scoring methods.

### Anchored sentiment scoring

In `02_anchored_sentiment_analysis.ipynb`, each segment is scored by comparing its mean cosine similarity to positive and negative anchor sentences:

\[
score(x) =
\frac{1}{|P|}\sum_{p \in P}\cos(e_x, e_p)
-
\frac{1}{|N|}\sum_{n \in N}\cos(e_x, e_n)
\]

where \(e_x\) is the embedding of the text segment, \(P\) is the set of positive anchors, and \(N\) is the set of negative anchors.

Positive values indicate greater similarity to the positive anchors; negative values indicate greater similarity to the negative anchors.

### Semantic-axis projection

In `03_semantic_axis_exploration.ipynb`, two semantic poles are first represented as centroids:

\[
c_A = \text{centroid}(A), \quad c_B = \text{centroid}(B)
\]

The semantic axis is then defined as:

\[
u = \frac{c_A - c_B}{\|c_A - c_B\|}
\]

Each segment embedding is centred relative to the midpoint between the two poles:

\[
m = \frac{c_A + c_B}{2}
\]

and projected onto the semantic axis:

\[
score(x) = (e_x - m) \cdot u
\]

Positive values indicate movement toward the first semantic pole, while negative values indicate movement toward the second semantic pole.

## Supporting code: `src/` and `configs/`

In addition to the notebooks, this repository includes reusable code and configuration files for literary NLP experiments, including model evaluation, anchored scoring, semantic-axis analysis, trajectory plotting, and earlier topic-distance work.

The `src/` folder contains Python scripts intended to move repeated logic out of notebooks and into reusable modules. At this stage, this includes code connected to BERTopic topic-distance analysis, where topic representations can be compared using embedding-based similarity.

The `configs/` folder stores experiment settings separately from the code. This makes it easier to rerun or modify experiments without changing the main scripts directly.

These files support the reusable components used across the cleaned notebooks, as well as earlier topic-distance experiments.

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

To run `03_semantic_axis_exploration.ipynb`, provide:

- a local corpus file, such as `data/LotR.txt`;
- a local Light–Dark anchor CSV file, such as `data/light_dark_anchors.csv`;
- a local sentence-transformer model path.

The Light–Dark anchor CSV should contain:

- `sentence_text`: the axis-anchor sentence;
- `axis_label`: either `light` or `dark`.

## Models

The baseline model is loaded from Hugging Face:

- `sentence-transformers/all-MiniLM-L6-v2`

The Tolkien-adapted models are expected to be stored locally. The notebook currently uses paths such as:

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


## Open Research and Copyright Constraints

This repository is designed around a reproducible-research model for copyrighted literary corpora.

Due to copyright restrictions, the original texts used in this project cannot be redistributed. Instead, this repository provides:

- preprocessing pipelines,
- model evaluation procedures,
- evaluation scripts,
- visualization methods,
- configuration settings,
- and documentation

to enable methodological transparency and reproducibility without distributing the original corpus.

Researchers may adapt the pipeline to their own licensed or public-domain corpora.

### Copyright note

Notebook outputs are cleared before committing because local runs may display copyrighted source text. The repository shares the code and workflow structure, but does not redistribute the underlying literary text.