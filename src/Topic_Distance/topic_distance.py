"""Topic-distance pipeline for BERTopic topics using a Tolkien Word2Vec model.

This module implements the topic similarity analysis used in the Tolkien
computational stylistics project.

Pipeline overview
-----------------
1. A BERTopic topic table (CSV) is loaded. This table must contain a `Topic`
   column and either a `Representation` or `Words` column containing the top
   words for each topic.

2. For each topic, the top *N* words (defined in the YAML configuration) are
   extracted.

3. Each topic is converted into a vector representation using a pretrained
   Word2Vec model trained on the Tolkien corpus.

4. The topic vector is constructed as the **mean of the Word2Vec vectors of the
   top topic words**:

       topic_vector = mean(Word2Vec(word_i) for word_i in topic_words)

5. Topics that do not contain enough words present in the Word2Vec vocabulary
   are skipped.

6. Pairwise cosine distances are then computed between all topic vectors,
   producing a topic–topic distance matrix.

The resulting matrix can be used to analyse semantic similarity or opposition
between BERTopic topics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
import ast
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_distances


DEFAULT_CONFIG_PATH = Path("configs/topic_distance_config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration for topic-distance analysis."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_word2vec_model(config: dict) -> KeyedVectors:
    """Load a Word2Vec binary/vector model from the configured path."""
    w2v_config = config["word2vec"]
    model_path = Path(w2v_config["model_path"])

    return KeyedVectors.load_word2vec_format(
        str(model_path),
        binary=w2v_config.get("binary", True),
    )


def parse_topic_words(topic_words: str | Iterable[str]) -> list[str]:
    """Normalize topic words into a simple list of strings."""

    if isinstance(topic_words, str):
        topic_words = topic_words.strip()

        # Handle BERTopic list-like strings: "['frodo', 'sam', 'mordor']"
        if topic_words.startswith("[") and topic_words.endswith("]"):
            try:
                topic_words = ast.literal_eval(topic_words)
            except Exception:
                pass

        # Handle comma-separated strings
        if isinstance(topic_words, str):
            return [word.strip() for word in topic_words.split(",") if word.strip()]

    return [str(word).strip() for word in topic_words if str(word).strip()]


def build_topic_vector(
    topic_words: list[str],
    model: KeyedVectors,
    min_required_words: int = 3,
) -> np.ndarray | None:
    """Create a single topic vector by averaging available word vectors.

    Returns ``None`` if too few words from the topic are present in the Word2Vec
    vocabulary.
    """
    valid_vectors = [model[word] for word in topic_words if word in model]

    if len(valid_vectors) < min_required_words:
        return None

    return np.mean(valid_vectors, axis=0)


def load_topic_table(topic_table_path: str | Path) -> pd.DataFrame:
    """Load a topic table exported from BERTopic.

    Expected columns:
    - ``Topic``
    - ``Representation`` or ``Words``

    ``Representation`` may contain comma-separated words.
    """
    topic_table_path = Path(topic_table_path)
    return pd.read_csv(topic_table_path)


def extract_topic_vectors(
    topic_df: pd.DataFrame,
    model: KeyedVectors,
    config: dict,
) -> tuple[list[int], np.ndarray]:
    """Build vectors for each eligible topic in the topic table."""
    input_cfg = config["input"]
    vector_cfg = config["vectorization"]

    top_n_words = input_cfg.get("top_n_words", 15)
    exclude_outlier = input_cfg.get("exclude_outlier_topic", True)
    outlier_label = input_cfg.get("outlier_topic_label", -1)
    min_required_words = vector_cfg.get("require_min_words", 3)

    topic_ids: list[int] = []
    topic_vectors: list[np.ndarray] = []

    representation_column = None
    for candidate in ("Representation", "Words"):
        if candidate in topic_df.columns:
            representation_column = candidate
            break

    if representation_column is None:
        raise ValueError(
            "Topic table must contain either a 'Representation' or 'Words' column."
        )

    for _, row in topic_df.iterrows():
        topic_id = int(row["Topic"])

        if exclude_outlier and topic_id == outlier_label:
            continue

        words = parse_topic_words(row[representation_column])[:top_n_words]
        topic_vector = build_topic_vector(
            topic_words=words,
            model=model,
            min_required_words=min_required_words,
        )

        if topic_vector is None:
            continue

        topic_ids.append(topic_id)
        topic_vectors.append(topic_vector)

    if not topic_vectors:
        raise ValueError(
            "No topic vectors could be constructed. Check the topic words and model vocabulary."
        )

    return topic_ids, np.vstack(topic_vectors)


def compute_distance_matrix(topic_ids: list[int], topic_vectors: np.ndarray) -> pd.DataFrame:
    """Compute a cosine distance matrix between topic vectors."""
    distances = cosine_distances(topic_vectors)
    labels = [f"Topic_{topic_id}" for topic_id in topic_ids]
    return pd.DataFrame(distances, index=labels, columns=labels)


def save_distance_matrix(distance_df: pd.DataFrame, config: dict) -> None:
    """Save the topic distance matrix to CSV if enabled in config."""
    output_cfg = config["output"]

    if not output_cfg.get("save_distance_matrix", True):
        return

    output_path = Path(output_cfg.get("distance_matrix_filename", "topic_distance_matrix.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distance_df.to_csv(output_path, encoding="utf-8")


def main(
    topic_table_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Run the topic-distance pipeline and return the distance matrix."""
    config = load_config(config_path)
    model = load_word2vec_model(config)
    topic_df = load_topic_table(topic_table_path)
    topic_ids, topic_vectors = extract_topic_vectors(topic_df, model, config)
    distance_df = compute_distance_matrix(topic_ids, topic_vectors)
    save_distance_matrix(distance_df, config)
    return distance_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute cosine distances between BERTopic topics using Word2Vec."
    )
    parser.add_argument(
        "topic_table",
        help="Path to a CSV file containing BERTopic topics and their top words.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the topic distance YAML config file.",
    )

    args = parser.parse_args()
    result_df = main(topic_table_path=args.topic_table, config_path=args.config)
    print(result_df.round(4))