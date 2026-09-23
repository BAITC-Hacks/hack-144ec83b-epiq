from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from .graph_features import assign_clusters, build_graph, calculate_features
from .io_validation import file_hashes, load_data, validate_data
from .scoring import build_cluster_summary, score_nodes


@dataclass
class AnalysisResult:
    nodes: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    edges: pd.DataFrame
    quality_report: dict
    manifest: dict


def analyze(data_dir: Path) -> AnalysisResult:
    started = time.perf_counter()
    nodes, edges, tx = load_data(data_dir)
    quality = validate_data(nodes, edges, tx)
    graph = build_graph(nodes, edges)
    features = calculate_features(graph, nodes, tx)
    features, communities = assign_clusters(graph, features)
    scored = score_nodes(features)
    cluster_summary = build_cluster_summary(scored, communities, edges)

    required_nodes = scored[
        ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    ].copy()
    top = (
        scored.sort_values(["priority_score", "gid"], ascending=[False, True])
        .head(20)
        .reset_index(drop=True)
    )
    top_nodes = top[["gid", "role", "priority_score", "evidence"]].copy()
    top_nodes.insert(0, "rank", range(1, len(top_nodes) + 1))
    top_nodes = top_nodes.rename(columns={"evidence": "why"})

    quality.update(
        {
            "n_components_all_nodes": nx.number_weakly_connected_components(graph),
            "n_isolates": int(scored["is_isolate"].sum()),
            "n_truncated_depth4": int(scored["truncated_by_depth"].sum()),
            "role_counts": {str(k): int(v) for k, v in scored["role"].value_counts().items()},
        }
    )
    manifest = {
        "status": "completed",
        "rules_version": "1.0.0",
        "input_sha256": file_hashes(data_dir),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "networkx": nx.__version__,
    }
    return AnalysisResult(
        nodes=scored,
        clusters=cluster_summary,
        top_nodes=top_nodes,
        edges=edges.copy(),
        quality_report=quality,
        manifest=manifest,
    )


def write_outputs(result: AnalysisResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    required = result.nodes[
        ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    ]
    required.to_csv(out_dir / "nodes_roles.csv", index=False)
    result.clusters.to_csv(out_dir / "clusters.csv", index=False)
    result.top_nodes.to_csv(out_dir / "top_nodes.csv", index=False)
    result.nodes.to_csv(out_dir / "node_features.csv", index=False)
    result.edges.to_csv(out_dir / "edges.csv", index=False)
    (out_dir / "quality_report.json").write_text(
        json.dumps(result.quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

