"""Explicit opt-in real API check. Never prints credentials or provider error bodies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from money_graph.ai_assistant import AssistantError, ask_graph, load_config
from money_graph.assistant_tools import GraphTools
from money_graph.pipeline import analyze


def synthetic_graph(out):
    """Entirely invented numbers; no records from the user's dataset are read."""
    data = out / "synthetic_ai_data"
    data.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame({"gid": [11, 12, 13, 21, 31], "depth": [0, 0, 0, 1, 4],
                          "is_seed": [True, True, True, False, False]})
    tx = pd.DataFrame([(11, 21, "2026-07-01", 100000), (12, 21, "2026-07-01", 50000),
                       (13, 21, "2026-07-02", 20000), (21, 31, "2026-07-03", 40000)],
                      columns=["src", "dst", "date", "sum_kzt"])
    edges = tx.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    edges["depth"] = 1
    for name, frame in (("nodes", nodes), ("edges", edges), ("transactions", tx)):
        frame.to_parquet(data / f"{name}.parquet", index=False)
    result = analyze(data)
    graph = GraphTools(result.nodes, edges, tx, result.repeated_routes, result.route_episodes)
    return graph, {"gid": "21", "seeds": ["11", "12"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send two real questions to OpenAI; uses API credits")
    parser.add_argument("--synthetic", action="store_true", help="Use only a fully invented five-node dataset")
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit --live is required: this check sends graph facts to OpenAI and uses API credits")
    out = Path(os.getenv("MONEY_GRAPH_OUT", str(ROOT / "out")))
    data = Path(os.getenv("MONEY_GRAPH_DATA", str(ROOT / "data")))
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT)
    if args.synthetic:
        graph, collector = synthetic_graph(out)
    else:
        cases = json.loads((out / "demo_cases.json").read_text(encoding="utf-8"))
        collector = cases["collector"]
        graph = GraphTools(pd.read_csv(out / "node_features.csv"), pd.read_csv(out / "edges.csv"),
                           pd.read_parquet(data / "transactions.parquet"), pd.read_csv(out / "repeated_routes.csv"),
                           pd.read_csv(out / "route_episodes.csv"))
    gid = int(collector["gid"])
    seeds = collector["seeds"][:2]
    questions = [
        ("profile", "Почему выбранный участник в топе? Сколько у него плательщиков и что запросить дальше?", "node_profile"),
        ("common", f"Кто получает деньги от обоих seed {seeds[0]} и {seeds[1]}? "
         "Покажи одного первого кандидата по приоритету и путь от каждого seed, максимум 4 перехода.", "find_common_recipients"),
    ]
    report = {"model": config.model, "synthetic": args.synthetic, "checks": []}
    for name, question, expected_tool in questions:
        result = ask_graph(question, graph, config, gid)
        tools_used = [t["tool"] for t in result["trace"]]
        assert expected_tool in tools_used, f"{name}: expected graph tool was not called"
        assert all("error" not in t["result"] for t in result["trace"]), f"{name}: graph tool returned an error"
        if name == "common":
            call = next(t for t in result["trace"] if t["tool"] == expected_tool)
            assert set(call["arguments"]["seeds"]) == set(seeds)
            assert call["result"]["matched"] > 0
        report["checks"].append({"name": name, "tools": tools_used, "usage": result["usage"],
                                 "answer": result["answer"]})
        print(f"{name}: OK; tools={tools_used}; usage={result['usage']}", flush=True)
    report_path = out / ("ai_smoke_synthetic.json" if args.synthetic else "ai_smoke.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Live checks passed. Answers saved to {report_path.name} (ignored by Git).")


if __name__ == "__main__":
    try:
        main()
    except AssistantError as exc:
        print(str(exc))
        raise SystemExit(1)
