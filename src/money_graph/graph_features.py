from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in nodes.sort_values("gid").itertuples(index=False):
        graph.add_node(int(row.gid), depth=int(row.depth), is_seed=bool(row.is_seed))
    for row in edges.sort_values(["src", "dst"]).itertuples(index=False):
        graph.add_edge(
            int(row.src),
            int(row.dst),
            sum_kzt=float(row.sum_kzt),
            n_tx=int(row.n_tx),
            depth=int(row.depth),
        )
    return graph


def _normalized_entropy(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    shares = [value / total for value in values if value > 0]
    return float(-sum(p * math.log(p) for p in shares) / math.log(len(values)))


def _seed_reach(graph: nx.DiGraph, seeds: list[int]) -> dict[int, int]:
    counts = {int(node): 0 for node in graph.nodes}
    for seed in sorted(seeds):
        distances = nx.single_source_shortest_path_length(graph, seed, cutoff=4)
        for node, distance in distances.items():
            if distance > 0:
                counts[int(node)] += 1
    return counts


def calculate_features(
    graph: nx.DiGraph, nodes: pd.DataFrame, tx: pd.DataFrame
) -> pd.DataFrame:
    gids = [int(gid) for gid in nodes.sort_values("gid")["gid"]]
    seeds = [int(gid) for gid in nodes.loc[nodes["is_seed"], "gid"]]

    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    in_kzt = dict(graph.in_degree(weight="sum_kzt"))
    out_kzt = dict(graph.out_degree(weight="sum_kzt"))
    in_tx = dict(graph.in_degree(weight="n_tx"))
    out_tx = dict(graph.out_degree(weight="n_tx"))
    seed_reach = _seed_reach(graph, seeds)

    try:
        pagerank = nx.pagerank(
            graph, alpha=0.85, max_iter=200, tol=1e-6, weight="sum_kzt"
        )
    except nx.PowerIterationFailedConvergence:
        pagerank = {node: np.nan for node in graph}

    k = min(128, len(graph))
    betweenness = nx.betweenness_centrality(
        graph, k=k if k < len(graph) else None, normalized=True, weight=None, seed=42
    )

    incoming_dates = tx.groupby("dst")["date"].agg(["min", "max"])
    outgoing_dates = tx.groupby("src")["date"].agg(["min", "max"])
    period_end = tx["date"].max()

    result = nodes[["gid", "depth", "is_seed"]].copy().sort_values("gid")
    result["in_deg"] = result["gid"].map(in_deg).fillna(0).astype(int)
    result["out_deg"] = result["gid"].map(out_deg).fillna(0).astype(int)
    result["in_kzt"] = result["gid"].map(in_kzt).fillna(0.0)
    result["out_kzt"] = result["gid"].map(out_kzt).fillna(0.0)
    result["in_tx"] = result["gid"].map(in_tx).fillna(0).astype(int)
    result["out_tx"] = result["gid"].map(out_tx).fillna(0).astype(int)
    result["pass_through"] = np.where(
        result["in_kzt"] > 0, result["out_kzt"] / result["in_kzt"], np.nan
    )
    result["seed_reach"] = result["gid"].map(seed_reach).fillna(0).astype(int)
    result["pagerank"] = result["gid"].map(pagerank)
    result["betweenness"] = result["gid"].map(betweenness).fillna(0.0)
    result["out_entropy"] = result["gid"].map(
        lambda gid: _normalized_entropy(
            [data["sum_kzt"] for _, _, data in graph.out_edges(int(gid), data=True)]
        )
    )
    result["first_in"] = result["gid"].map(incoming_dates["min"])
    result["last_in"] = result["gid"].map(incoming_dates["max"])
    result["first_out"] = result["gid"].map(outgoing_dates["min"])
    result["last_out"] = result["gid"].map(outgoing_dates["max"])
    result["days_after_last_in"] = (period_end - result["last_in"]).dt.days
    result["truncated_by_depth"] = (result["depth"] == 4) & (result["out_deg"] == 0)
    result["is_isolate"] = (result["in_deg"] == 0) & (result["out_deg"] == 0)
    return result.reset_index(drop=True)


def assign_clusters(
    graph: nx.DiGraph, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        value = float(data["sum_kzt"])
        if weighted.has_edge(source, target):
            weighted[source][target]["weight"] += value
        else:
            weighted.add_edge(source, target, weight=value)

    components = sorted(
        nx.weakly_connected_components(graph), key=lambda group: (-len(group), min(group))
    )
    component_id: dict[int, int] = {}
    communities: list[set[int]] = []
    for cid, component in enumerate(components):
        for gid in component:
            component_id[int(gid)] = cid
        if len(component) == 1:
            communities.append(set(component))
            continue
        subgraph = weighted.subgraph(component).copy()
        detected = nx.community.louvain_communities(
            subgraph, weight="weight", resolution=1.0, seed=42
        )
        communities.extend(detected)

    communities = sorted(communities, key=lambda group: min(group))
    cluster_id = {
        int(gid): index for index, community in enumerate(communities) for gid in community
    }
    enriched = features.copy()
    enriched["component_id"] = enriched["gid"].map(component_id).astype(int)
    enriched["cluster_id"] = enriched["gid"].map(cluster_id).astype(int)

    external_clusters = {}
    for gid in graph.nodes:
        neighbors = set(graph.predecessors(gid)) | set(graph.successors(gid))
        external_clusters[int(gid)] = len(
            {cluster_id[int(other)] for other in neighbors if cluster_id[int(other)] != cluster_id[int(gid)]}
        )
    enriched["external_clusters"] = enriched["gid"].map(external_clusters).astype(int)
    return enriched, pd.DataFrame(
        [{"cluster_id": index, "members": sorted(int(x) for x in group)} for index, group in enumerate(communities)]
    )

