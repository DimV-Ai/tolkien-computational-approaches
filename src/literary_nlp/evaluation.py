"""
Evaluation utilities for anchored sentence-transformer experiments.

This module contains reusable functions for evaluating whether a model can
separate held-out positive and negative examples using anchor-based centroids.
It is intended to support the model-evaluation notebook while keeping the
notebook itself readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


LABEL_ORDER = ["positive", "negative"]


@dataclass
class AnchorEvaluationResult:
    """
    Container for anchor-based evaluation outputs.

    Attributes:
        metrics: Dictionary of summary metrics.
        confusion: Confusion matrix as a dataframe.
        predictions: Per-example predictions and similarity scores.
        classification_report_df: Full classification report as a dataframe.
    """

    metrics: dict
    confusion: pd.DataFrame
    predictions: pd.DataFrame
    classification_report_df: pd.DataFrame


def l2_normalise_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def centroid(vectors: np.ndarray) -> np.ndarray:
    """
    Compute and normalise the centroid of a set of vectors.

    The returned shape is (1, embedding_dim), which makes dot-product cosine
    similarity calculations straightforward when embeddings are normalised.
    """
    centre = vectors.mean(axis=0, keepdims=True)
    return l2_normalise_matrix(centre)


def embed_sentences(
    model: SentenceTransformer,
    sentences: list[str],
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode sentences as normalised NumPy embeddings."""
    if not sentences:
        raise ValueError("No sentences were supplied for embedding.")

    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def predict_from_anchor_centroids(
    test_vectors: np.ndarray,
    positive_centroid: np.ndarray,
    negative_centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict labels by comparing similarity to positive and negative centroids.

    With normalised embeddings, cosine similarity is equivalent to the dot
    product. Ties are assigned to the positive label.
    """
    similarity_positive = (test_vectors @ positive_centroid.T).reshape(-1)
    similarity_negative = (test_vectors @ negative_centroid.T).reshape(-1)

    predictions = np.where(
        similarity_positive >= similarity_negative,
        "positive",
        "negative",
    )

    return predictions, similarity_positive, similarity_negative


def validate_anchor_eval_dataframe(
    df: pd.DataFrame,
    required_columns: Iterable[str] = ("sentence_text", "gold_label", "split"),
) -> pd.DataFrame:
    """
    Validate and standardise an anchor-evaluation dataframe.

    Expected columns:
    - sentence_text
    - gold_label: positive or negative
    - split: anchor or test
    """
    df = df.copy()
    required_columns = set(required_columns)
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )

    df["gold_label"] = df["gold_label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    df = df[df["gold_label"].isin(LABEL_ORDER)].copy()

    unexpected_splits = set(df["split"].unique()) - {"anchor", "test"}
    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values: {unexpected_splits}. "
            "Expected only 'anchor' and 'test'."
        )

    if df[df["split"] == "anchor"].empty:
        raise ValueError("No anchor rows found. Check the `split` column.")

    if df[df["split"] == "test"].empty:
        raise ValueError("No test rows found. Check the `split` column.")

    return df


def evaluate_anchor_classifier(
    model: SentenceTransformer,
    anchors: pd.DataFrame,
    test: pd.DataFrame,
    batch_size: int = 64,
) -> AnchorEvaluationResult:
    """
    Evaluate a model using positive and negative anchor centroids.

    The anchor examples define positive and negative centroids. Held-out test
    examples are classified according to which centroid they are closer to.
    """
    positive_anchor_sentences = anchors.loc[
        anchors["gold_label"] == "positive",
        "sentence_text",
    ].tolist()

    negative_anchor_sentences = anchors.loc[
        anchors["gold_label"] == "negative",
        "sentence_text",
    ].tolist()

    if not positive_anchor_sentences:
        raise ValueError("Anchors must include at least one positive sentence.")

    if not negative_anchor_sentences:
        raise ValueError("Anchors must include at least one negative sentence.")

    positive_anchor_vectors = embed_sentences(
        model,
        positive_anchor_sentences,
        batch_size=batch_size,
    )
    negative_anchor_vectors = embed_sentences(
        model,
        negative_anchor_sentences,
        batch_size=batch_size,
    )

    positive_centroid = centroid(positive_anchor_vectors)
    negative_centroid = centroid(negative_anchor_vectors)

    test_sentences = test["sentence_text"].tolist()
    test_vectors = embed_sentences(model, test_sentences, batch_size=batch_size)

    y_true = test["gold_label"].to_numpy()
    y_pred, similarity_positive, similarity_negative = predict_from_anchor_centroids(
        test_vectors,
        positive_centroid,
        negative_centroid,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    confusion = pd.DataFrame(
        cm,
        index=["true_positive", "true_negative"],
        columns=["pred_positive", "pred_negative"],
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_positive": precision_score(
            y_true,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "recall_positive": recall_score(
            y_true,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "f1_positive": f1_score(
            y_true,
            y_pred,
            pos_label="positive",
            zero_division=0,
        ),
        "precision_negative": precision_score(
            y_true,
            y_pred,
            pos_label="negative",
            zero_division=0,
        ),
        "recall_negative": recall_score(
            y_true,
            y_pred,
            pos_label="negative",
            zero_division=0,
        ),
        "f1_negative": f1_score(
            y_true,
            y_pred,
            pos_label="negative",
            zero_division=0,
        ),
        "tp_positive": int(cm[0, 0]),
        "fn_positive": int(cm[0, 1]),
        "fp_positive": int(cm[1, 0]),
        "tn_positive": int(cm[1, 1]),
    }

    predictions = test.copy()
    predictions["pred_label"] = y_pred
    predictions["similarity_positive"] = similarity_positive
    predictions["similarity_negative"] = similarity_negative
    predictions["margin_positive_minus_negative"] = (
        predictions["similarity_positive"] - predictions["similarity_negative"]
    )

    classification_report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    return AnchorEvaluationResult(
        metrics=metrics,
        confusion=confusion,
        predictions=predictions,
        classification_report_df=classification_report_df,
    )


def mcnemar_contingency_table(
    y_true: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    first_name: str = "First model",
    second_name: str = "Second model",
) -> pd.DataFrame:
    """
    Build the paired contingency table used by McNemar's test.

    Rows indicate whether the first model was correct. Columns indicate whether
    the second model was correct.
    """
    y_true = np.asarray(y_true)
    first_predictions = np.asarray(first_predictions)
    second_predictions = np.asarray(second_predictions)

    first_correct = first_predictions == y_true
    second_correct = second_predictions == y_true

    both_correct = int(np.sum(first_correct & second_correct))
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    both_wrong = int(np.sum(~first_correct & ~second_correct))

    return pd.DataFrame(
        [[both_correct, first_only], [second_only, both_wrong]],
        index=[f"{first_name} correct", f"{first_name} wrong"],
        columns=[f"{second_name} correct", f"{second_name} wrong"],
    )