"""
Project one embedding onto the semantic axis.

The embedding is first centred relative to the midpoint between the two
semantic pole centroids. Positive values indicate movement toward the first
pole used to define the axis; negative values indicate movement toward the
second pole.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalised copy of a vector."""
    return vector / np.maximum(np.linalg.norm(vector), 1e-12)

def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute and normalise the centroid of a set of embeddings."""
    return l2_normalise(embeddings.mean(axis=0))

def project_score_from_embedding(
    embedding: np.ndarray,
    midpoint: np.ndarray,
    axis_unit: np.ndarray,
) -> float:
    """
    Project one embedding onto the semantic axis.

    The embedding is first centred relative to the midpoint between the two
    semantic pole centroids. Positive values indicate movement toward the
    Light pole; negative values indicate movement toward the Dark pole.
    """
    centered = embedding - midpoint
    return float(np.dot(centered, axis_unit))

def project_score_from_text(
    text: str,
    model: SentenceTransformer,
    midpoint: np.ndarray,
    axis_unit: np.ndarray,
) -> float:
    """
    Encode one text string and return its semantic-axis score.
    """
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    return project_score_from_embedding(
        embedding=embedding,
        midpoint=midpoint,
        axis_unit=axis_unit,
    )

def project_scores_from_texts(
    texts: list[str],
    model: SentenceTransformer,
    midpoint: np.ndarray,
    axis_unit: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode multiple texts and project them onto the semantic axis.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    centered_embeddings = embeddings - midpoint
    return centered_embeddings @ axis_unit
