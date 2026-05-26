import pandas as pd
import matplotlib.pyplot as plt


def add_rolling_score(
    df: pd.DataFrame,
    score_col: str,
    output_col: str,
    window: int = 20,
) -> pd.DataFrame:
    """
    Add a centred rolling-mean score column to a dataframe.
    """
    plot_df = df.copy()
    plot_df["segment_index"] = range(len(plot_df))
    plot_df[output_col] = (
        plot_df[score_col]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )

    return plot_df


def plot_trajectory(
    df: pd.DataFrame,
    score_col: str,
    chapter_col: str = "chapter",
    title: str = "Trajectory",
    ylabel: str = "Score",
    positive_label: str = "positive region",
    negative_label: str = "negative region",
    positive_color: str = "steelblue",
    negative_color: str = "indianred",
    line_color: str = "black",
    figsize: tuple[int, int] = (18, 6),
):
    """
    Plot a trajectory over ordered text segments.

    Positive and negative regions are shaded relative to zero.
    Chapter labels are shown at chapter midpoints.
    """
    required_cols = {"segment_index", score_col, chapter_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    x = df["segment_index"].to_numpy()
    y = df[score_col].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x,
        y,
        color=line_color,
        linewidth=1.8,
        label=score_col,
    )

    ax.axhline(0, linewidth=1, linestyle="--")

    ax.fill_between(
        x,
        y,
        0,
        where=y >= 0,
        color=positive_color,
        alpha=0.25,
        interpolate=True,
        label=positive_label,
    )

    ax.fill_between(
        x,
        y,
        0,
        where=y < 0,
        color=negative_color,
        alpha=0.25,
        interpolate=True,
        label=negative_label,
    )

    chapter_starts = df.groupby(chapter_col)["segment_index"].min()
    chapter_mids = df.groupby(chapter_col)["segment_index"].median()

    for _, start in chapter_starts.items():
        ax.axvline(start, linewidth=0.5, alpha=0.25)

    ax.set_xticks(chapter_mids.values)
    ax.set_xticklabels(chapter_mids.index, rotation=90, fontsize=8)

    ax.set_xlabel("Chapter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    return fig, ax