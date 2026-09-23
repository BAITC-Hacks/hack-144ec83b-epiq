"""Bounded, explainable investigations over the observed transaction graph."""
from __future__ import annotations

from collections import deque

import networkx as nx
import pandas as pd


def seed_paths(graph: nx.DiGraph, seed: int, max_hops: int = 4) -> dict[int, list[int]]:
    """Deterministic shortest directed paths, limited to the observed horizon."""
    paths = {seed: [seed]}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        if len(paths[current]) - 1 >= max_hops:
            continue
        for target in sorted(graph.successors(current)):
            if target not in paths:
                paths[target] = paths[current] + [int(target)]
                queue.append(int(target))
    return paths


def common_recipients(nodes: pd.DataFrame, edges: pd.DataFrame, seeds: list[int],
                      max_hops: int = 4) -> tuple[pd.DataFrame, dict]:
    seeds = sorted(set(int(seed) for seed in seeds))
    valid = set(nodes.loc[nodes["is_seed"].astype(bool), "gid"].astype(int))
    if not 2 <= len(seeds) <= 5 or not set(seeds) <= valid:
        raise ValueError("Выберите от двух до пяти различных seed из данных")
    if not 1 <= max_hops <= 4:
        raise ValueError("Глубина должна быть от 1 до 4")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gid"].astype(int))
    graph.add_edges_from((int(r.src), int(r.dst)) for r in edges.itertuples(index=False))
    paths = {seed: seed_paths(graph, seed, max_hops) for seed in seeds}
    shared = set.intersection(*(set(p) for p in paths.values())) - set(seeds)
    result = nodes[nodes["gid"].isin(shared)].copy()
    result["selected_seed_count"] = len(seeds)
    result["max_hops"] = [max(len(paths[s][int(gid)]) - 1 for s in seeds)
                          for gid in result["gid"]]
    result = result.sort_values(["priority_score", "gid"], ascending=[False, True])
    return result, {int(gid): {s: paths[s][int(gid)] for s in seeds}
                    for gid in result["gid"]}
