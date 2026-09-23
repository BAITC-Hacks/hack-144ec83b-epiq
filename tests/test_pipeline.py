from __future__ import annotations

import pandas as pd
import pytest

from money_graph.pipeline import AnalysisResult, validate_result


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

