from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd

from .advanced_analysis import build_resilience_report, build_route_patterns
from .route_evidence import enrich_route_evidence
from .repeated_routes import repeated_routes
from .demo_cases import build_demo_cases
from .graph_features import assign_clusters, build_graph, calculate_features
from .io_validation import file_hashes, load_data, validate_data
from .scoring import build_cluster_summary, score_nodes


REQUIRED_NODE_COLUMNS = [
    "gid", "role", "role_score", "cluster_id", "priority_score", "evidence"
]
ALLOWED_ROLES = {
    "consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"
}


@dataclass
class AnalysisResult:
    nodes: pd.DataFrame
    clusters: pd.DataFrame
    top_nodes: pd.DataFrame
    edges: pd.DataFrame
    quality_report: dict
    manifest: dict
    resilience: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    repeated_routes: pd.DataFrame = field(default_factory=pd.DataFrame)
    route_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)


def analyze(data_dir: Path) -> AnalysisResult:
    started = time.perf_counter()
    nodes, edges, tx = load_data(data_dir)
    quality = validate_data(nodes, edges, tx)
    graph = build_graph(nodes, edges)
    features = calculate_features(graph, nodes, tx)
    features, communities = assign_clusters(graph, features)
    scored = score_nodes(features)
    cluster_summary = build_cluster_summary(scored, communities, edges)
    resilience = build_resilience_report(graph, scored)
    route_patterns = enrich_route_evidence(build_route_patterns(graph, scored), tx)
    repeated, episodes = repeated_routes(tx)

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
            "n_cycle_nodes": int(scored["in_cycle"].sum()),
            "n_rapid_flow_nodes": int((scored["rapid_flow_days"] > 0).sum()),
            "n_high_anomaly_nodes": int((scored["anomaly_score"] >= 0.8).sum()),
            "n_repeated_routes": len(repeated),
        }
    )
    manifest = {
        "status": "completed",
        "rules_version": "1.3.0",
        "input_sha256": file_hashes(data_dir),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "networkx": nx.__version__,
    }
    result = AnalysisResult(
        nodes=scored,
        clusters=cluster_summary,
        top_nodes=top_nodes,
        edges=edges.copy(),
        quality_report=quality,
        manifest=manifest,
        resilience=resilience,
        route_patterns=route_patterns,
        repeated_routes=repeated,
        route_episodes=episodes,
    )
    validate_result(result)
    return result


def validate_result(result: AnalysisResult) -> None:
    nodes = result.nodes
    missing = [column for column in REQUIRED_NODE_COLUMNS if column not in nodes.columns]
    if missing:
        raise ValueError(f"В результате отсутствуют колонки: {missing}")
    if nodes["gid"].duplicated().any():
        raise ValueError("В результате повторяются gid")
    if nodes[REQUIRED_NODE_COLUMNS].isna().any().any():
        raise ValueError("Обязательные поля результата содержат пустые значения")
    unknown_roles = set(nodes["role"]) - ALLOWED_ROLES
    if unknown_roles:
        raise ValueError(f"Неизвестные роли: {sorted(unknown_roles)}")
    for column in ["role_score", "priority_score"]:
        if not nodes[column].between(0, 1).all():
            raise ValueError(f"{column} должен находиться в диапазоне 0–1")
    if (nodes["evidence"].astype(str).str.len() > 200).any():
        raise ValueError("evidence не должен превышать 200 символов")
    if int(result.clusters["n_nodes"].sum()) != len(nodes):
        raise ValueError("Сумма n_nodes кластеров не совпадает с числом узлов")
    if set(result.clusters["cluster_id"]) != set(nodes["cluster_id"]):
        raise ValueError("Набор cluster_id не согласован")

    expected_top_size = min(20, len(nodes))
    if len(result.top_nodes) != expected_top_size:
        raise ValueError(f"Топ должен содержать {expected_top_size} строк")
    if result.top_nodes["gid"].duplicated().any():
        raise ValueError("В топе повторяются gid")
    if not result.top_nodes["priority_score"].is_monotonic_decreasing:
        raise ValueError("Топ не отсортирован по priority_score")
    source = nodes.set_index("gid")
    for row in result.top_nodes.itertuples(index=False):
        if row.gid not in source.index:
            raise ValueError(f"gid {row.gid} из топа отсутствует в узлах")
        node = source.loc[row.gid]
        if row.role != node["role"] or abs(row.priority_score - node["priority_score"]) > 1e-9:
            raise ValueError(f"Топ не согласован с результатом для gid {row.gid}")


def write_outputs(result: AnalysisResult, out_dir: Path) -> None:
    validate_result(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = result.nodes[REQUIRED_NODE_COLUMNS]
    required.to_csv(out_dir / "nodes_roles.csv", index=False)
    result.clusters.to_csv(out_dir / "clusters.csv", index=False)
    result.top_nodes.to_csv(out_dir / "top_nodes.csv", index=False)
    result.nodes.to_csv(out_dir / "node_features.csv", index=False)
    result.edges.to_csv(out_dir / "edges.csv", index=False)
    result.resilience.to_csv(out_dir / "resilience.csv", index=False)
    result.route_patterns.to_csv(out_dir / "route_patterns.csv", index=False)
    result.repeated_routes.to_csv(out_dir / "repeated_routes.csv", index=False)
    result.route_episodes.to_csv(out_dir / "route_episodes.csv", index=False)
    (out_dir / "quality_report.json").write_text(
        json.dumps(result.quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cases, demo_text = build_demo_cases(result.nodes, result.edges, result.repeated_routes,
                                       result.route_episodes, result.manifest)
    (out_dir / "demo_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "demo_cases.md").write_text(demo_text, encoding="utf-8")

