"""
Preprocessing utilities for literary text segmentation.

The functions in this module are designed to convert a prepared local text file
into a standard segments dataframe with at least:

- segment_text
- chapter
- segment_type
- token_count

The default plain-text parser assumes chapter markers in the form:

###CHAPTER: Chapter Name

and a local paragraph-start convention based on leading spaces.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


DIALOG_START_CHARS = ('"', "'", "“", "”", "‘", "’", "—", "–", "-", "―")


def is_dialogue_first_line(line: str) -> bool:
    """Return True if a line appears to begin with dialogue punctuation."""
    return line.lstrip().startswith(DIALOG_START_CHARS)


def is_paragraph_lead(line: str) -> bool:
    """Return True if the line follows the local five-space paragraph convention."""
    return line.startswith("     ")


def tokenize_count(text: str) -> int:
    """Count word tokens, treating hyphenated words as single tokens."""
    return len(re.findall(r"\b\w+(?:-\w+)*\b", text))


def build_raw_paragraphs(lines: List[str]) -> Tuple[List[str], List[str], List[bool]]:
    """
    Build raw paragraphs with aligned chapter labels and dialogue flags.

    Returns:
    - paragraphs
    - paragraph chapter labels
    - paragraph dialogue flags
    """
    paragraphs: List[str] = []
    para_chapters: List[str] = []
    para_is_dialogue: List[bool] = []

    current_chapter = "Unknown"
    buf_lines: List[str] = []
    buf_is_dialogue = False

    def flush_paragraph():
        nonlocal buf_lines, buf_is_dialogue

        if buf_lines:
            paragraphs.append(" ".join(ln.strip() for ln in buf_lines).strip())
            para_chapters.append(current_chapter)
            para_is_dialogue.append(buf_is_dialogue)

        buf_lines = []
        buf_is_dialogue = False

    for raw in lines:
        line = raw.rstrip("\r")

        if line.startswith("###CHAPTER:"):
            flush_paragraph()
            current_chapter = line.replace("###CHAPTER:", "").strip()
            continue

        if line.strip() == "":
            flush_paragraph()
            continue

        if is_paragraph_lead(line):
            flush_paragraph()
            buf_lines = [line]
            buf_is_dialogue = is_dialogue_first_line(line)
        else:
            if buf_lines:
                buf_lines.append(line)
            else:
                buf_lines = [line]
                buf_is_dialogue = is_dialogue_first_line(line)

    flush_paragraph()
    return paragraphs, para_chapters, para_is_dialogue


def merge_dialogue_aware(
    paragraphs: List[str],
    para_chapters: List[str],
    para_is_dialogue: List[bool],
    min_tok_narr: int = 80,
    max_tok_narr: int = 300,
    min_tok_dial: int = 60,
    max_tok_dial: int = 180,
    max_dialogue_turns: int = 6,
) -> Tuple[List[str], List[str]]:
    """
    Merge raw paragraphs into dialogue-aware analysis segments.

    Chapter boundaries are preserved: segments are never merged across chapters.
    """
    processed_segments: List[str] = []
    chapter_tags: List[str] = []

    seg_buf: List[str] = []
    seg_tokens = 0
    seg_is_dialogue = None
    seg_dialogue_turns = 0
    seg_chapter = None

    def seg_flush():
        nonlocal seg_buf, seg_tokens, seg_is_dialogue, seg_dialogue_turns, seg_chapter

        if seg_buf:
            processed_segments.append(" ".join(seg_buf).strip())
            chapter_tags.append(seg_chapter)

        seg_buf = []
        seg_tokens = 0
        seg_is_dialogue = None
        seg_dialogue_turns = 0
        seg_chapter = None

    for paragraph, chapter, is_dialogue in zip(
        paragraphs,
        para_chapters,
        para_is_dialogue,
    ):
        paragraph_tokens = tokenize_count(paragraph)

        if paragraph_tokens == 0:
            continue

        if seg_buf and chapter != seg_chapter:
            seg_flush()

        if not seg_buf:
            seg_buf = [paragraph]
            seg_tokens = paragraph_tokens
            seg_is_dialogue = is_dialogue
            seg_dialogue_turns = 1 if is_dialogue else 0
            seg_chapter = chapter
            continue

        max_tokens = max_tok_dial if seg_is_dialogue else max_tok_narr
        min_tokens = min_tok_dial if seg_is_dialogue else min_tok_narr

        type_switch = is_dialogue != seg_is_dialogue
        exceeds_limits = (
            seg_tokens + paragraph_tokens > max_tokens
            or (seg_is_dialogue and seg_dialogue_turns >= max_dialogue_turns)
        )

        if type_switch or exceeds_limits:
            if seg_tokens >= min_tokens:
                seg_flush()

                seg_buf = [paragraph]
                seg_tokens = paragraph_tokens
                seg_is_dialogue = is_dialogue
                seg_dialogue_turns = 1 if is_dialogue else 0
                seg_chapter = chapter
                continue

        seg_buf.append(paragraph)
        seg_tokens += paragraph_tokens

        if is_dialogue and seg_is_dialogue:
            seg_dialogue_turns += 1
        elif type_switch:
            seg_is_dialogue = is_dialogue
            seg_dialogue_turns = 1 if is_dialogue else 0

    seg_flush()
    return processed_segments, chapter_tags


def build_segments_df_from_plaintext(
    corpus_path: str | Path,
    min_tok_narr: int = 80,
    max_tok_narr: int = 300,
    min_tok_dial: int = 60,
    max_tok_dial: int = 180,
    max_dialogue_turns: int = 6,
) -> pd.DataFrame:
    """
    Build a standard segments dataframe from a prepared plain-text corpus file.

    Returns a dataframe with:
    - segment_text
    - chapter
    - segment_type
    - token_count
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Could not find corpus file: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")

    raw_paragraphs, raw_chapter_tags, raw_is_dialogue = build_raw_paragraphs(lines)

    processed_segments, chapter_tags = merge_dialogue_aware(
        raw_paragraphs,
        raw_chapter_tags,
        raw_is_dialogue,
        min_tok_narr=min_tok_narr,
        max_tok_narr=max_tok_narr,
        min_tok_dial=min_tok_dial,
        max_tok_dial=max_tok_dial,
        max_dialogue_turns=max_dialogue_turns,
    )

    segments_df = pd.DataFrame(
        {
            "segment_text": processed_segments,
            "chapter": chapter_tags,
            "segment_type": "mixed_or_unknown",
        }
    )

    segments_df["token_count"] = segments_df["segment_text"].apply(tokenize_count)

    return segments_df


def source_format_diagnostics(
    corpus_path: str | Path,
    segments_df: pd.DataFrame | None = None,
) -> dict:
    """
    Return simple diagnostics for the prepared plain-text source format.

    If a segments dataframe is supplied, include segment-level diagnostics too.
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Could not find corpus file: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")

    raw_paragraphs, _, _ = build_raw_paragraphs(lines)

    diagnostics = {
        "chapter_markers": sum(line.startswith("###CHAPTER:") for line in lines),
        "paragraph_lead_lines": sum(is_paragraph_lead(line) for line in lines),
        "blank_lines": sum(line.strip() == "" for line in lines),
        "raw_paragraphs": len(raw_paragraphs),
    }

    if segments_df is not None:
        diagnostics.update(
            {
                "processed_segments": len(segments_df),
                "unique_chapters": int(segments_df["chapter"].nunique()),
                "min_segment_tokens": int(segments_df["token_count"].min()),
                "max_segment_tokens": int(segments_df["token_count"].max()),
                "mean_segment_tokens": float(segments_df["token_count"].mean()),
                "median_segment_tokens": float(segments_df["token_count"].median()),
            }
        )

    return diagnostics