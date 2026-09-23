from __future__ import annotations

import pandas as pd
import pytest

from money_graph.pipeline import AnalysisResult, analyze, validate_result, write_outputs


def test_pipeline_exports_repeat_evidence_and_demo_cases(tmp_path):
    import json

    nodes = pd.DataFrame({"gid": [1, 2, 3, 4], "depth": [0, 1, 4, 0],
                          "is_seed": [True, False, False, True]})
    tx = pd.DataFrame([(1, 2, "2026-07-01", 100_000), (2, 3, "2026-07-02", 95_000),
                       (1, 2, "2026-07-08", 100_000), (2, 3, "2026-07-09", 95_000)],
                      columns=["src", "dst", "date", "sum_kzt"])
    edges = tx.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    edges["depth"] = [1, 4]
    for name, frame in [("nodes", nodes), ("edges", edges), ("transactions", tx)]:
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    result = analyze(tmp_path)
    output = tmp_path / "out"
    write_outputs(result, output)
    assert pd.read_csv(output / "repeated_routes.csv").episode_count.tolist() == [2]
    assert len(pd.read_csv(output / "route_episodes.csv")) == 2
    assert len(pd.read_csv(output / "nodes_roles.csv")) == 4
    demo = json.loads((output / "demo_cases.json").read_text(encoding="utf-8"))
    assert demo["boundary"]["gid"] == "3"
    assert demo["repeated_route"]["via"] == "2"
    assert "collector" not in demo  # no invented case where prerequisites fail
    assert result.nodes.loc[result.nodes.gid == 4, "priority_score"].iloc[0] == 0


def test_validation_rejects_empty_evidence():
    result = _valid_result()
    result.nodes.loc[0, "evidence"] = None

    with pytest.raises(ValueError, match="пустые значения"):
        validate_result(result)


def test_validation_rejects_inconsistent_top():
    result = _valid_result()
    result.top_nodes.loc[0, "priority_score"] = 0.1

    with pytest.raises(ValueError, match="не согласован"):
        validate_result(result)


def _valid_result() -> AnalysisResult:
    nodes = pd.DataFrame(
        {
            "gid": [1, 2],
            "role": ["consolidator", "peripheral"],
            "role_score": [0.8, 0.0],
            "cluster_id": [0, 0],
            "priority_score": [0.9, 0.1],
            "evidence": ["3 плательщика", "Признаков роли нет"],
        }
    )
    clusters = pd.DataFrame(
        {
            "cluster_id": [0],
            "n_nodes": [2],
            "n_seed": [1],
            "sum_kzt_internal": [100.0],
            "top_gids": ["[1, 2]"],
            "hypothesis": ["тест"],
        }
    )
    top = pd.DataFrame(
        {
            "rank": [1, 2],
            "gid": [1, 2],
            "role": ["consolidator", "peripheral"],
            "priority_score": [0.9, 0.1],
            "why": ["3 плательщика", "Признаков роли нет"],
        }
    )
    return AnalysisResult(nodes, clusters, top, pd.DataFrame(), {}, {})
