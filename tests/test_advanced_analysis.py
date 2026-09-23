from __future__ import annotations

import networkx as nx
import pandas as pd

from money_graph.advanced_analysis import build_resilience_report, build_route_patterns
from money_graph.graph_features import build_graph, calculate_features


def test_temporal_and_cycle_features_are_explainable():
    nodes = pd.DataFrame(
        {
            "gid": [1, 2],
            "depth": [0, 1],
            "is_seed": [True, False],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [1, 2],
            "dst": [2, 1],
            "sum_kzt": [100.0, 80.0],
            "n_tx": [1, 1],
            "depth": [1, 1],
        }
    )
    tx = pd.DataFrame(
        {
            "src": [1, 2],
            "dst": [2, 1],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "sum_kzt": [100.0, 80.0],
        }
    )

    features = calculate_features(build_graph(nodes, edges), nodes, tx).set_index("gid")

    assert features.loc[1, "cycle_size"] == 2
    assert features.loc[2, "cycle_size"] == 2
    assert features.loc[2, "same_day_flow_days"] == 1
    assert features.loc[2, "rapid_flow_days"] == 1


def test_resilience_and_two_hop_routes():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, sum_kzt=100.0, n_tx=1)
    graph.add_edge(2, 3, sum_kzt=60.0, n_tx=1)
    scored = pd.DataFrame(
        {
            "gid": [1, 2, 3],
            "role": ["peripheral", "transit", "terminal"],
            "priority_score": [0.2, 0.9, 0.3],
            "rapid_flow_days": [0, 1, 0],
            "cycle_size": [0, 0, 0],
        }
    )

    routes = build_route_patterns(graph, scored)
    resilience = build_resilience_report(graph, scored)

    assert routes.iloc[0][["src", "via", "dst"]].tolist() == [1, 2, 3]
    assert routes.iloc[0]["bottleneck_kzt"] == 60.0
    assert resilience.loc[resilience["n_removed"] == 1, "n_components"].iloc[0] == 2
