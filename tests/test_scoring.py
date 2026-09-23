from __future__ import annotations

import pandas as pd

from money_graph.scoring import PRIORITY_LABELS, score_nodes


def test_priority_contributions_sum_to_score_and_isolate_stays_zero():
    rows = [_base_row(gid=1, in_deg=4, in_kzt=1_000_000, in_tx=4, seed_reach=3),
            _base_row(gid=2, out_deg=6, out_kzt=2_000_000, out_tx=6, seed_reach=7),
            _base_row(gid=3)]
    result = score_nodes(pd.DataFrame(rows))
    pd.testing.assert_series_equal(result[list(PRIORITY_LABELS)].sum(axis=1).round(6),
                                   result.priority_score, check_names=False)
    assert result.loc[2, list(PRIORITY_LABELS)].eq(0).all()
    assert result.loc[1, "seed_contribution"] == 0.3


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

