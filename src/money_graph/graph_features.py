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
    incoming_days = tx.groupby("dst")["date"].apply(
        lambda values: sorted(set(pd.to_datetime(values).dt.normalize()))
    ).to_dict()
    outgoing_days = tx.groupby("src")["date"].apply(
        lambda values: sorted(set(pd.to_datetime(values).dt.normalize()))
    ).to_dict()
    incoming_daily = (
        tx.assign(day=tx["date"].dt.normalize())
        .groupby(["dst", "day"])
        .agg(payers=("src", "nunique"), amount=("sum_kzt", "sum"))
        .reset_index()
    )
    max_same_day_payers = incoming_daily.groupby("dst")["payers"].max().to_dict()
    max_daily_in = incoming_daily.groupby("dst")["amount"].max().to_dict()
    median_daily_in = incoming_daily.groupby("dst")["amount"].median().to_dict()

    same_day_flow: dict[int, int] = {}
    rapid_flow: dict[int, int] = {}
    min_forward_delay: dict[int, float] = {}
    for gid in gids:
        in_days = incoming_days.get(gid, [])
        out_days = outgoing_days.get(gid, [])
        same_day_flow[gid] = len(set(in_days) & set(out_days))
        delays = [
            int((out_day - in_day).days)
            for in_day in in_days
            for out_day in out_days
            if out_day >= in_day
        ]
        rapid_flow[gid] = sum(
            1 for in_day in in_days if any(0 <= (out_day - in_day).days <= 2 for out_day in out_days)
        )
        min_forward_delay[gid] = float(min(delays)) if delays else np.nan

    cycle_size = {gid: 0 for gid in gids}
    for component in nx.strongly_connected_components(graph):
        is_cycle = len(component) > 1 or any(graph.has_edge(node, node) for node in component)
        if is_cycle:
            for gid in component:
                cycle_size[int(gid)] = len(component)

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
    result["same_day_flow_days"] = result["gid"].map(same_day_flow).fillna(0).astype(int)
    result["rapid_flow_days"] = result["gid"].map(rapid_flow).fillna(0).astype(int)
    result["min_forward_delay_days"] = result["gid"].map(min_forward_delay)
    result["max_same_day_payers"] = (
        result["gid"].map(max_same_day_payers).fillna(0).astype(int)
    )
    result["max_daily_in_kzt"] = result["gid"].map(max_daily_in).fillna(0.0)
    daily_median = result["gid"].map(median_daily_in).fillna(0.0)
    result["incoming_spike_ratio"] = np.where(
        daily_median > 0, result["max_daily_in_kzt"] / daily_median, 0.0
    )
    result["cycle_size"] = result["gid"].map(cycle_size).fillna(0).astype(int)
    result["in_cycle"] = result["cycle_size"] > 0
    result["truncated_by_depth"] = (result["depth"] == 4) & (result["out_deg"] == 0)
    result["is_isolate"] = (result["in_deg"] == 0) & (result["out_deg"] == 0)
    result["data_gap"] = "Критичных пробелов по наблюдаемому контуру не выявлено"
    result["next_request"] = "Сверить операции и контрагентов по первичным банковским данным"
    depth_mask = result["truncated_by_depth"]
    result.loc[depth_mask, "data_gap"] = "Исходящий поток после четвёртого колена неизвестен"
    result.loc[depth_mask, "next_request"] = (
        "Запросить исходящие переводы узла за тот же период и следующий месяц"
    )
    seed_mask = result["is_seed"].astype(bool)
    result.loc[seed_mask, "data_gap"] = "Входящий поток seed до начала наблюдаемой цепочки неполон"
    result.loc[seed_mask, "next_request"] = (
        "Запросить входящие переводы seed до первой наблюдаемой операции"
    )
    missing_inbound = (~seed_mask) & (result["in_kzt"] > 0) & (result["pass_through"] > 1.5)
    result.loc[missing_inbound, "data_gap"] = (
        "Исходящий объём заметно выше наблюдаемого входящего"
    )
    result.loc[missing_inbound, "next_request"] = (
        "Запросить полный входящий оборот и остаток на начало периода"
    )
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

