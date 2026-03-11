

"""Export BERTopic topic relationships to Gephi-ready network files.

This module converts a topic–topic distance matrix into a network representation
that can be opened in Gephi or other graph visualization tools.

Pipeline overview
-----------------
1. Load a topic–topic distance matrix from CSV.
2. Optionally convert distance to similarity using:

       similarity = 1 - distance

3. Filter edges by removing self-links, applying a similarity threshold, and/or
   keeping only the strongest edges per topic.
4. Export an edge list CSV for Gephi.
5. Optionally export a node table enriched with BERTopic topic metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = Path("configs/network_export_config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML configuration for topic-network export."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_distance_matrix(distance_matrix_path: str | Path) -> pd.DataFrame:
    """Load a topic–topic distance matrix from CSV."""
    distance_matrix_path = Path(distance_matrix_path)
    return pd.read_csv(distance_matrix_path, index_col=0)


def convert_to_similarity(matrix_df: pd.DataFrame, convert: bool = True) -> pd.DataFrame:
    """Convert a distance matrix to a similarity matrix if requested."""
    if not convert:
        return matrix_df.copy()
    return 1 - matrix_df


def build_edge_list(similarity_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert a similarity matrix into an edge list suitable for Gephi."""
    filtering_cfg = config["filtering"]
    exclude_self_links = filtering_cfg.get("exclude_self_links", True)
    similarity_threshold = filtering_cfg.get("similarity_threshold", 0.0)
    top_k_edges_per_topic = filtering_cfg.get("top_k_edges_per_topic")

    edges = []

    for source in similarity_df.index:
        row = similarity_df.loc[source]
        candidate_edges = []

        for target, weight in row.items():
            if exclude_self_links and source == target:
                continue

            if pd.isna(weight):
                continue

            if weight < similarity_threshold:
                continue

            candidate_edges.append((source, target, float(weight)))

        if top_k_edges_per_topic is not None:
            candidate_edges = sorted(candidate_edges, key=lambda x: x[2], reverse=True)[:top_k_edges_per_topic]

        edges.extend(candidate_edges)

    edge_df = pd.DataFrame(edges, columns=["source", "target", "weight"])

    if edge_df.empty:
        return edge_df

    # Remove duplicate undirected edges by sorting node pairs.
    edge_df["pair_key"] = edge_df.apply(
        lambda row: tuple(sorted([row["source"], row["target"]])), axis=1
    )
    edge_df = (
        edge_df.sort_values("weight", ascending=False)
        .drop_duplicates(subset="pair_key")
        .drop(columns="pair_key")
        .reset_index(drop=True)
    )

    return edge_df


def load_topic_info(topic_info_path: str | Path) -> pd.DataFrame:
    """Load BERTopic topic metadata from CSV."""
    topic_info_path = Path(topic_info_path)
    return pd.read_csv(topic_info_path)


def build_node_table(similarity_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build a node table for topics, optionally enriched with BERTopic metadata."""
    input_cfg = config["input"]
    nodes_cfg = config["nodes"]

    topic_labels = list(similarity_df.index)
    node_df = pd.DataFrame({
        "id": topic_labels,
        "label": topic_labels,
    })

    topic_info_path = input_cfg.get("topic_info_path")
    if not topic_info_path:
        return node_df

    topic_info_file = Path(topic_info_path)
    if not topic_info_file.exists():
        return node_df

    topic_info_df = load_topic_info(topic_info_file)

    # BERTopic topic labels are typically like "Topic_0" in the similarity matrix.
    # Convert them back to integer topic ids for joining with topic_info.csv.
    node_df["Topic"] = node_df["id"].str.replace("Topic_", "", regex=False).astype(int)

    merge_columns = ["Topic"]
    if nodes_cfg.get("include_topic_size", True) and "Count" in topic_info_df.columns:
        merge_columns.append("Count")
    if nodes_cfg.get("include_representation", True):
        if "Representation" in topic_info_df.columns:
            merge_columns.append("Representation")
        elif "Name" in topic_info_df.columns:
            merge_columns.append("Name")

    node_df = node_df.merge(topic_info_df[merge_columns], on="Topic", how="left")
    return node_df


def save_network_files(edge_df: pd.DataFrame, node_df: pd.DataFrame, config: dict) -> tuple[Path, Path]:
    """Save Gephi-ready edge and node tables to CSV."""
    output_cfg = config["output"]
    output_dir = Path(output_cfg.get("output_dir", "outputs/network"))
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_path = output_dir / output_cfg.get("edges_filename", "topic_edges.csv")
    nodes_path = output_dir / output_cfg.get("nodes_filename", "topic_nodes.csv")

    edge_df.to_csv(edges_path, index=False, encoding="utf-8")
    node_df.to_csv(nodes_path, index=False, encoding="utf-8")
    return edges_path, nodes_path


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the topic-network export pipeline."""
    config = load_config(config_path)
    input_cfg = config["input"]
    conversion_cfg = config["conversion"]

    distance_df = load_distance_matrix(input_cfg["distance_matrix_path"])
    similarity_df = convert_to_similarity(
        distance_df,
        convert=conversion_cfg.get("convert_distance_to_similarity", True),
    )

    edge_df = build_edge_list(similarity_df, config)
    node_df = build_node_table(similarity_df, config)
    save_network_files(edge_df, node_df, config)
    return edge_df, node_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export BERTopic topic relationships to Gephi-ready CSV files."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the network export YAML config file.",
    )

    args = parser.parse_args()
    edges, nodes = main(config_path=args.config)
    print("Exported edge list:")
    print(edges.head())
    print("\nExported node table:")
    print(nodes.head())