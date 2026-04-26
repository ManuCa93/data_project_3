"""Shared helper functions for the notebook splits.

This module centralizes the reusable text, pair, and sampling utilities
that were previously embedded inside the notebook.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def extract_author_names(authors: Any) -> set[str]:
    names: set[str] = set()
    for author in as_list(authors):
        if isinstance(author, dict):
            name = author.get("name")
            if name:
                names.add(normalize_scalar(name))
        else:
            text = normalize_scalar(author)
            if text:
                names.add(text)
    return names


def extract_strings(values: Any) -> set[str]:
    items: set[str] = set()
    for value in as_list(values):
        if isinstance(value, dict):
            for key in ("text", "name", "value", "ref", "id"):
                if key in value and value[key]:
                    text = normalize_scalar(value[key])
                    if text:
                        items.add(text)
                    break
        else:
            text = normalize_scalar(value)
            if text:
                items.add(text)
    return items


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and np.isnan(value):
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def tokenize_text(text: Any) -> set[str]:
    normalized = normalize_scalar(text)
    if not normalized:
        return set()
    return {token for token in re.split(r"\s+", normalized) if token}


def jaccard(left: Any, right: Any) -> float:
    left_set = set(left) if not isinstance(left, set) else left
    right_set = set(right) if not isinstance(right, set) else right
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def build_eda_features(row: pd.Series, id_to_year: Mapping[str, int]) -> pd.Series:
    src_y = id_to_year.get(row["src_id"])
    dst_y = id_to_year.get(row["dst_id"])
    year_gap = src_y - dst_y if src_y is not None and dst_y is not None else None
    return pd.Series({"year_gap": year_gap, "label": row["label"]})


def compute_overlaps(
    row: pd.Series,
    id_to_venue: Mapping[str, Any],
    id_to_authors: Mapping[str, set[str]],
) -> pd.Series:
    src, dst = row["src_id"], row["dst_id"]

    v_src = id_to_venue.get(src)
    v_dst = id_to_venue.get(dst)
    same_venue = 1 if (v_src and v_dst and v_src == v_dst and v_src != "") else 0

    a_src = id_to_authors.get(src, set())
    a_dst = id_to_authors.get(dst, set())
    intersection = len(a_src.intersection(a_dst))
    union = len(a_src.union(a_dst))
    auth_jaccard = intersection / union if union > 0 else 0

    return pd.Series({
        "same_venue": same_venue,
        "author_jaccard": auth_jaccard,
        "label": row["label"],
    })


def get_edge_index(split_df: Any, id_to_idx: Any, max_dst_node: int) -> torch.Tensor:
    edges = (
        split_df.select(["node_idx", "references"])
        .explode("references")
        .drop_nulls()
        .rename({"node_idx": "src", "references": "target_id"})
    )
    edges = edges.join(id_to_idx, left_on="target_id", right_on="id", how="inner")
    edges = edges.rename({"node_idx": "dst"})
    edges = edges.filter(edges["dst"] < max_dst_node)
    src_np = edges["src"].to_numpy()
    dst_np = edges["dst"].to_numpy()
    return torch.tensor(np.vstack([src_np, dst_np]), dtype=torch.long)


def subsample_edges(edge_index: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction >= 1.0:
        return edge_index
    n = max(1, int(edge_index.size(1) * fraction))
    return edge_index[:, torch.randperm(edge_index.size(1))[:n]]


def sample_negatives_hard(
    known_edge_index: torch.Tensor,
    num_nodes: int,
    target_count: int,
    years_array: np.ndarray,
    max_gap: int = 10,
    oversample: float = 5.0,
    seed: int = 42,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    src_k = known_edge_index[0].numpy().astype(np.int64)
    dst_k = known_edge_index[1].numpy().astype(np.int64)
    known_codes = set((src_k * num_nodes + dst_k).tolist())

    collected_src: list[np.ndarray] = []
    collected_dst: list[np.ndarray] = []
    remaining = target_count

    while remaining > 0:
        n_sample = int(remaining * oversample)
        src_rand = rng.integers(0, num_nodes, size=n_sample, dtype=np.int64)
        dst_rand = rng.integers(0, num_nodes, size=n_sample, dtype=np.int64)

        src_years = years_array[src_rand]
        dst_years = years_array[dst_rand]
        valid_time = (dst_years <= src_years) & ((src_years - dst_years) <= max_gap)
        valid = valid_time & (src_rand != dst_rand)
        src_rand, dst_rand = src_rand[valid], dst_rand[valid]

        codes = (src_rand * num_nodes + dst_rand).tolist()
        mask = np.array([c not in known_codes for c in codes], dtype=bool)
        src_rand, dst_rand = src_rand[mask], dst_rand[mask]

        take = min(remaining, len(src_rand))
        collected_src.append(src_rand[:take])
        collected_dst.append(dst_rand[:take])
        remaining -= take

    src_out = np.concatenate(collected_src).astype(np.int64)
    dst_out = np.concatenate(collected_dst).astype(np.int64)
    return torch.tensor(np.vstack([src_out, dst_out]), dtype=torch.long)


def pairs_to_matrix(edge_index: torch.Tensor, node_feats: np.ndarray) -> np.ndarray:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    return np.hstack([node_feats[src], node_feats[dst]])


def create_pair_df_from_edges(
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    split_name: str,
    idx_to_id_dict: Mapping[int, str],
    max_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos_src = pos_edges[0].cpu().numpy()
    pos_dst = pos_edges[1].cpu().numpy()
    neg_src = neg_edges[0].cpu().numpy()
    neg_dst = neg_edges[1].cpu().numpy()

    n_pos = min(len(pos_src), max_samples)
    n_neg = min(len(neg_src), max_samples)

    pos_idx = rng.choice(len(pos_src), n_pos, replace=False)
    neg_idx = rng.choice(len(neg_src), n_neg, replace=False)

    src = np.concatenate([pos_src[pos_idx], neg_src[neg_idx]])
    dst = np.concatenate([pos_dst[pos_idx], neg_dst[neg_idx]])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    return pd.DataFrame({
        "src_id": [idx_to_id_dict[s] for s in src],
        "dst_id": [idx_to_id_dict[d] for d in dst],
        "label": labels,
        "split": split_name,
    })


def build_pair_features(
    src_id: str,
    dst_id: str,
    label: int,
    split: str,
    paper_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    src = paper_lookup[src_id]
    dst = paper_lookup[dst_id]

    src_authors = extract_author_names(src["authors"])
    dst_authors = extract_author_names(dst["authors"])
    src_keywords = extract_strings(src["keywords"])
    dst_keywords = extract_strings(dst["keywords"])
    src_refs = extract_strings(src["refs"])
    dst_refs = extract_strings(dst["refs"])
    src_title = tokenize_text(src["title"])
    dst_title = tokenize_text(dst["title"])
    src_abstract = tokenize_text(src["abstract"])
    dst_abstract = tokenize_text(dst["abstract"])

    src_venue = normalize_scalar(src["venue"])
    dst_venue = normalize_scalar(dst["venue"])
    src_year = int(src["year"])
    dst_year = int(dst["year"])

    author_overlap = len(src_authors & dst_authors)
    keyword_overlap = len(src_keywords & dst_keywords)
    ref_overlap = len(src_refs & dst_refs)

    return {
        "src_id": src_id,
        "dst_id": dst_id,
        "label": label,
        "split": split,
        "year_gap": src_year - dst_year,
        "src_year": src_year,
        "dst_year": dst_year,
        "same_year": int(src_year == dst_year),
        "same_decade": int(src_year // 10 == dst_year // 10),
        "source_not_earlier": int(src_year >= dst_year),
        "same_venue": int(src_venue != "" and src_venue == dst_venue),
        "author_overlap": author_overlap,
        "author_overlap_any": int(author_overlap > 0),
        "author_jaccard": jaccard(src_authors, dst_authors),
        "keyword_overlap": keyword_overlap,
        "keyword_overlap_any": int(keyword_overlap > 0),
        "keyword_jaccard": jaccard(src_keywords, dst_keywords),
        "title_jaccard": jaccard(src_title, dst_title),
        "abstract_jaccard": jaccard(src_abstract, dst_abstract),
        "bibliographic_coupling": ref_overlap,
        "bibliographic_coupling_any": int(ref_overlap > 0),
        "reference_jaccard": jaccard(src_refs, dst_refs),
        "source_n_authors": safe_int(src["n_authors"]),
        "dst_n_authors": safe_int(dst["n_authors"]),
        "source_n_keywords": safe_int(src["n_keywords"]),
        "dst_n_keywords": safe_int(dst["n_keywords"]),
        "source_n_references": safe_int(src["n_references"]),
        "dst_n_references": safe_int(dst["n_references"]),
        "source_title_len": safe_int(src["title_len"]),
        "dst_title_len": safe_int(dst["title_len"]),
        "source_abstract_len": safe_int(src["abstract_len"]),
        "dst_abstract_len": safe_int(dst["abstract_len"]),
    }
