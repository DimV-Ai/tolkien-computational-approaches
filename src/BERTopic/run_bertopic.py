"""
BERTopic pipeline used for the Tolkien computational stylistics project.

This module reproduces the behaviour of the original research notebook that
applied BERTopic to a preprocessed Tolkien corpus. The key assumptions of the
pipeline are:

1. Input documents are NOT raw text. The model expects a tokenized corpus
   produced by an earlier preprocessing and segmentation pipeline.

2. Each document is represented as a list of tokens. Before fitting BERTopic,
   the tokens are re-joined into whitespace-separated strings using:

       " ".join(doc)

3. The segmentation stage (paragraph detection, merging, token filtering, etc.)
   happens earlier in the pipeline and is not implemented in this module.

4. BERTopic is run with custom UMAP and HDBSCAN models defined in the YAML
   configuration file located in `configs/bertopic_config.yaml`.

5. Topic probabilities are intentionally NOT computed (see note below) in
   order to match the behaviour of the original experiment.

This module therefore focuses only on:
- building the BERTopic model from configuration
- converting tokenized documents into BERTopic input format
- running the model
- exporting topics, document-topic assignments, and the trained model
"""

# NOTE: `calculate_probabilities=True`, is intentionally not enabled.
# The original BERTopic notebook used in this project relied on the default
# BERTopic behaviour, which does not compute full document–topic probability
# distributions.The same behaviour is kept here to reproduce the original
# experiment exactly.

from pathlib import Path

import hdbscan
import pandas as pd
import yaml
from bertopic import BERTopic
from umap import UMAP

CONFIG_PATH = Path("configs/bertopic_config.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/bertopic")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load the BERTopic YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_topic_model(config):
    umap_model = UMAP(**config["umap"])
    hdbscan_model = hdbscan.HDBSCAN(**config["hdbscan"])

    topic_model = BERTopic(
        language=config["bertopic"]["language"],
        top_n_words=config["bertopic"]["top_n_words"],
        nr_topics=config["bertopic"]["nr_topics"],
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )

    return topic_model


def prepare_docs(tokenized_corpus):
    """Convert a tokenized corpus (list of token lists) into BERTopic input strings."""
    return [" ".join(doc) for doc in tokenized_corpus]


def save_topic_info(topic_model, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Save BERTopic topic information to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_info_path = output_dir / "topic_info.csv"

    topic_info_df = topic_model.get_topic_info()
    topic_info_df.to_csv(topic_info_path, index=False, encoding="utf-8")
    return topic_info_path



def save_document_topic_assignments(
    docs,
    topics,
    probs=None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_prob_columns: int = 3,
) -> Path:
    """Save document-topic assignments to CSV.

    The output contains the input document text, its assigned topic, and up to
    `max_prob_columns` probability columns when probabilities are available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "document_topic_assignments.csv"

    assignments_df = pd.DataFrame({
        "document": docs,
        "topic": topics,
    })

    if probs is not None:
        try:
            probs_df = pd.DataFrame(probs)
            max_cols = min(max_prob_columns, probs_df.shape[1])
            for i in range(max_cols):
                assignments_df[f"probability_{i}"] = probs_df.iloc[:, i]
        except Exception:
            # Some BERTopic configurations return probabilities in formats that
            # are not straightforward to expand into columns.
            pass

    assignments_df.to_csv(assignments_path, index=False, encoding="utf-8")
    return assignments_path


def save_bertopic_model(topic_model, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Save the BERTopic model to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "bertopic_model"

    topic_model.save(str(model_path))
    return model_path


def run_bertopic(
    tokenized_corpus,
    config_path: str | Path = CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_outputs: bool = True,
    max_prob_columns: int = 3,
):
    """Build and run a BERTopic model on a tokenized corpus.

    Optionally saves:
    - BERTopic topic information as CSV
    - document-topic assignments as CSV
    - the BERTopic model directory
    """
    config = load_config(config_path)
    topic_model = build_topic_model(config)
    docs = prepare_docs(tokenized_corpus)
    topics, probs = topic_model.fit_transform(docs)

    saved_paths = {}
    if save_outputs:
        saved_paths["topic_info_csv"] = save_topic_info(topic_model, output_dir)
        saved_paths["document_topic_assignments_csv"] = save_document_topic_assignments(
            docs=docs,
            topics=topics,
            probs=probs,
            output_dir=output_dir,
            max_prob_columns=max_prob_columns,
        )
        saved_paths["bertopic_model_dir"] = save_bertopic_model(topic_model, output_dir)

    return topic_model, topics, probs, saved_paths


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be imported and used from a notebook or another script. "
        "Call run_bertopic(tokenized_corpus) after loading your tokenized corpus. "
        "The function can also save topic info, document-topic assignments, and the BERTopic model."
    )