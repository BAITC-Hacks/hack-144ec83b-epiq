from __future__ import annotations

import pandas as pd

from money_graph.scoring import score_nodes


def test_depth_four_sink_is_not_terminal():
    row = _base_row(gid=1, depth=4, in_deg=4, in_kzt=1_000_000, in_tx=4)
    result = score_nodes(pd.DataFrame([row])).iloc[0]
    assert result["role"] != "terminal"
    assert result["role"] == "consolidator"


def test_seed_ratio_does_not_create_transit_role():
    row = _base_row(
        gid=2,
        depth=0,
        is_seed=True,
        in_deg=1,
        out_deg=1,
        in_kzt=100_000,
        out_kzt=100_000,
        in_tx=2,
        out_tx=2,
        pass_through=1.0,
    )
    result = score_nodes(pd.DataFrame([row])).iloc[0]
    assert result["role"] == "peripheral"


def _base_row(**updates):
    row = {
        "gid": 0,
        "depth": 1,
        "is_seed": False,
        "in_deg": 0,
        "out_deg": 0,
        "in_kzt": 0.0,
        "out_kzt": 0.0,
        "in_tx": 0,
        "out_tx": 0,
        "pass_through": float("nan"),
        "seed_reach": 0,
        "pagerank": 0.0,
        "betweenness": 0.0,
        "out_entropy": 0.0,
        "days_after_last_in": float("nan"),
        "truncated_by_depth": False,
        "is_isolate": True,
        "component_id": 0,
        "cluster_id": 0,
        "external_clusters": 0,
    }
    row.update(updates)
    row["is_isolate"] = row["in_deg"] == 0 and row["out_deg"] == 0
    if row["in_kzt"] > 0 and "pass_through" not in updates:
        row["pass_through"] = row["out_kzt"] / row["in_kzt"]
    return row

