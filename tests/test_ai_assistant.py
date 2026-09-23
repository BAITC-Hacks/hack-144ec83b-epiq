from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from money_graph.ai_assistant import AIConfig, AssistantError, ask_graph, load_config, safe_api_error
from money_graph.assistant_tools import GraphTools
from money_graph.pipeline import analyze
from money_graph.scoring import PRIORITY_LABELS


@pytest.fixture
def graph(tmp_path):
    ids = [100000000000000001 + i for i in range(5)]
    nodes = pd.DataFrame({"gid": ids, "depth": [0, 0, 1, 2, 4], "is_seed": [True, True, False, False, False]})
    tx = pd.DataFrame([(ids[0], ids[2], "2026-07-01", 100000), (ids[2], ids[3], "2026-07-02", 95000),
                       (ids[0], ids[2], "2026-07-08", 200000), (ids[2], ids[3], "2026-07-09", 190000),
                       (ids[1], ids[2], "2026-07-08", 5000), (ids[3], ids[4], "2026-07-10", 6000)],
                      columns=["src", "dst", "date", "sum_kzt"])
    edges = tx.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    edges["depth"] = 1
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    result = analyze(tmp_path)
    return GraphTools(result.nodes, edges, tx, result.repeated_routes, result.route_episodes)


def test_tools_preserve_gids_and_resolve_each_seed(graph):
    seeds = [str(g) for g in graph.nodes.loc[graph.nodes.is_seed, "gid"]]
    result = graph.execute("find_common_recipients", {"seeds": seeds, "max_hops": 1, "limit": 5})
    assert result["matched"] == 1
    assert result["shown"] == 1
    candidate = result["candidates"][0]
    assert candidate["node"]["gid"] == "100000000000000003"
    assert {p["seed"] for p in candidate["paths"]} == set(seeds)
    assert all(p["path"][-1] == candidate["node"]["gid"] for p in candidate["paths"])
    assert "error" in graph.execute("find_common_recipients", {"seeds": [seeds[0]] * 2, "max_hops": 1, "limit": 5})


def test_history_filters_before_aggregation_and_limits_only_sample(graph):
    result = graph.execute("transaction_history", {"gid": "100000000000000003", "direction": "incoming",
                           "since": "2026-07-08", "until": "2026-07-08", "limit": 1})
    assert result["matched"] == 2
    assert result["incoming_kzt"] == 205000
    assert result["outgoing_kzt"] == 0
    assert len(result["transactions"]) == 1


def test_comparison_and_repeated_routes_use_calculated_facts(graph):
    gids = ["100000000000000003", "100000000000000004"]
    profile = graph.execute("node_profile", {"gid": gids[0]})
    assert profile["factors"]["between_contribution"] == "Посредничество"
    assert sum(profile["priority_parts"].values()) == pytest.approx(profile["node"]["priority_score"], abs=1e-6)
    result = graph.execute("compare_nodes", {"gids": gids})
    assert [n["gid"] for n in result["nodes"]] == gids
    for row in result["nodes"]:
        assert abs(sum(row[k] for k in PRIORITY_LABELS) - row["priority_score"]) < 1e-6
    result = graph.execute("find_repeated_routes", {"gid": gids[0], "limit": 2})
    assert result["routes"][0]["episode_count"] == 2
    assert len(result["routes"][0]["examples"]) == 2


def test_combined_filters_and_untrusted_arguments(graph):
    args = {"role": None, "min_seed_count": 2, "min_payers": 2, "max_out_ratio": 1,
            "depth": 1, "sort_by": "priority_score", "limit": 10}
    result = graph.execute("rank_nodes", args)
    assert result["matched"] == 1
    assert result["nodes"][0]["gid"] == "100000000000000003"
    assert "error" in graph.execute("rank_nodes", {**args, "limit": 100000})
    assert "error" in graph.execute("rank_nodes", {**args, "max_out_ratio": float('nan')})
    assert "error" in graph.execute("node_profile", {"gid": "999", "file": ".streamlit/secrets.toml"})
    assert "error" in graph.execute("read_file", {"path": ".streamlit/secrets.toml"})


class FakeClient:
    def __init__(self, responses):
        self.replies = iter(responses)
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def response(output=None, text="", status="completed"):
    return SimpleNamespace(output=output or [], output_text=text, status=status,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def call(name="node_profile", args=None):
    return SimpleNamespace(type="function_call", name=name,
                           arguments=json.dumps(args or {"gid": "100000000000000003"}), call_id="call_test")


def test_tool_loop_retains_reasoning_and_exposes_evidence_without_key(graph):
    reasoning = SimpleNamespace(type="reasoning", encrypted_content="opaque")
    client = FakeClient([response([reasoning, call()]), response(text="Проверьте 100000000000000003 [D1].")])
    result = ask_graph("Объясни выбранного", graph, AIConfig("local-secret"), 100000000000000003, client=client)
    assert result["trace"][0]["result"]["node"]["gid"] == "100000000000000003"
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert all(not c["store"] for c in client.calls)
    assert reasoning in client.calls[1]["input"]
    assert "local-secret" not in str(client.calls)


@pytest.mark.parametrize("answer", ["Клиент 100000000000099999 [D1].", "Клиент 100000000000000003 [D9]."])
def test_invented_gids_and_references_are_rejected(graph, answer):
    client = FakeClient([response([call()]), response(text=answer)])
    with pytest.raises(AssistantError, match="неподтверждённые"):
        ask_graph("Объясни узел", graph, AIConfig("secret"), 100000000000000003, client=client)


def test_api_failure_and_incomplete_output_are_safe(graph):
    client = FakeClient([RuntimeError("secret-value-in-provider-error")])
    with pytest.raises(AssistantError) as error:
        ask_graph("Объясни узел", graph, AIConfig("secret"), 100000000000000003, client=client)
    assert "secret-value" not in str(error.value)
    client = FakeClient([response(status="incomplete")])
    with pytest.raises(AssistantError, match="не завершён"):
        ask_graph("Объясни узел", graph, AIConfig("secret"), 100000000000000003, client=client)
    assert "ключ" in safe_api_error(SimpleNamespace(status_code=401))
    assert "квота" in safe_api_error(SimpleNamespace(status_code=429, code="insufficient_quota"))


def test_secret_config_is_optional_and_repr_does_not_expose_it(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert not load_config(tmp_path).api_key
    monkeypatch.setenv("OPENAI_API_KEY", "test-private-key")
    config = load_config(tmp_path)
    assert config.api_key == "test-private-key"
    assert "test-private-key" not in repr(config)
