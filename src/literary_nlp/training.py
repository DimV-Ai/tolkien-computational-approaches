"""
Training utilities for building local sentence-transformer models.

These functions support unsupervised / weakly supervised literary-domain
sentence-transformer training from user-supplied local corpora. The repository
provides the training procedure, but does not distribute copyrighted training
texts or trained model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader


def load_training_sentences(corpus_path: str | Path) -> list[str]:
    """
    Load non-empty lines from a local training corpus file.

    The expected input is a plain-text file with one sentence or short text unit
    per line. Empty lines are ignored.
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Could not find training corpus: {corpus_path}. "
            "Provide a local plain-text corpus file before running training."
        )

    with open(corpus_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    if len(sentences) < 2:
        raise ValueError(
            "The training corpus must contain at least two non-empty lines "
            "to create sentence pairs."
        )

    return sentences


def build_adjacent_sentence_pairs(
    sentences: list[str],
    step: int = 2,
) -> list[InputExample]:
    """
    Build adjacent-sentence training pairs from a list of sentences.

    By default, pairs are non-overlapping:
    - sentences[0] with sentences[1]
    - sentences[2] with sentences[3]
    - sentences[4] with sentences[5]

    Set `step=1` to create overlapping adjacent pairs instead:
    - sentences[0] with sentences[1]
    - sentences[1] with sentences[2]
    - sentences[2] with sentences[3]

    The resulting InputExample objects are suitable for
    MultipleNegativesRankingLoss.
    """
    if step < 1:
        raise ValueError("step must be at least 1.")

    if len(sentences) < 2:
        raise ValueError("At least two sentences are required to build pairs.")

    return [
        InputExample(texts=[sentences[i], sentences[i + 1]])
        for i in range(0, len(sentences) - 1, step)
    ]


def build_sentence_transformer(
    base_model_name: str = "bert-base-uncased",
    max_seq_length: int = 256,
) -> SentenceTransformer:
    """
    Build a SentenceTransformer from a Transformer backbone and mean pooling.

    This converts a token-level Transformer model, such as BERT, into a
    sentence-embedding model by adding a pooling layer over token embeddings.
    """
    transformer = models.Transformer(
        base_model_name,
        max_seq_length=max_seq_length,
    )

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    return SentenceTransformer(modules=[transformer, pooling])


def build_training_dataloader(
    training_examples: list[InputExample],
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for sentence-transformer training examples.
    """
    if not training_examples:
        raise ValueError("training_examples is empty.")

    return DataLoader(
        training_examples,
        shuffle=shuffle,
        batch_size=batch_size,
    )


def train_with_multiple_negatives_loss(
    model: SentenceTransformer,
    training_examples: list[InputExample],
    output_path: str | Path,
    epochs: int = 1,
    batch_size: int = 16,
    warmup_steps: int = 100,
    show_progress_bar: bool = True,
) -> SentenceTransformer:
    """
    Train a SentenceTransformer with MultipleNegativesRankingLoss.

    This function expects positive sentence pairs, such as adjacent sentences
    from a local literary corpus. Other examples in the same batch are treated
    as implicit negatives by the loss function.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataloader = build_training_dataloader(
        training_examples,
        batch_size=batch_size,
        shuffle=True,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=show_progress_bar,
    )

    return model


def train_epoch_grid(
    training_examples: list[InputExample],
    output_dir: str | Path,
    epoch_values: Iterable[int] = (1, 2, 4, 8),
    base_model_name: str = "bert-base-uncased",
    max_seq_length: int = 256,
    batch_size: int = 16,
    warmup_steps: int = 100,
) -> dict[int, Path]:
    """
    Train separate sentence-transformer models for multiple epoch settings.

    Returns a mapping from epoch count to the saved model directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[int, Path] = {}

    for epochs in epoch_values:
        if epochs < 1:
            raise ValueError(f"Invalid epoch value: {epochs}")

        model = build_sentence_transformer(
            base_model_name=base_model_name,
            max_seq_length=max_seq_length,
        )

        model_output_path = output_dir / f"sentence_transformer_epoch_{epochs}"

        train_with_multiple_negatives_loss(
            model=model,
            training_examples=training_examples,
            output_path=model_output_path,
            epochs=epochs,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
        )

        saved_paths[epochs] = model_output_path

    return saved_paths