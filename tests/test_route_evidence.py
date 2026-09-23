import pandas as pd

from money_graph.route_evidence import enrich_route_evidence, graph_edge_signals


def test_exact_counterparty_dates_and_reverse_time():
    routes = pd.DataFrame({"src": [1, 4, 7], "via": [2, 2, 2], "dst": [3, 5, 8]})
    tx = pd.DataFrame([
        (1, 2, "2026-07-01", 100), (2, 3, "2026-07-02", 90),
        (4, 2, "2026-07-09", 100), (2, 5, "2026-07-08", 90),
        (7, 2, "2026-07-12", 100), (2, 8, "2026-07-12", 10),
    ], columns=["src", "dst", "date", "sum_kzt"])
    result = enrich_route_evidence(routes, tx)
    assert result.rapid_signal.tolist() == [True, False, True]
    assert result.comparable_amounts.tolist() == [True, False, False]
    assert result.iloc[0].amount_ratio == 0.9
    assert result.iloc[0].delay_days == 1
    assert result.iloc[2].temporal_status == "Один день: порядок неизвестен"


def test_empty_routes_keep_schema_and_large_identifiers():
    gid = 100000008710791100
    routes = pd.DataFrame({"src": [gid], "via": [gid + 1], "dst": [gid + 2]})
    tx = pd.DataFrame(columns=["src", "dst", "date", "sum_kzt"])
    result = enrich_route_evidence(routes, tx)
    assert result.src.iloc[0] == gid
    assert not result.rapid_signal.iloc[0]
    assert "temporal_status" in enrich_route_evidence(routes.iloc[:0], tx)


def test_graph_signals_do_not_spread_to_unrelated_edges():
    edges = pd.DataFrame([(1, 2), (2, 3), (2, 4)], columns=["src", "dst"])
    tx = pd.DataFrame([(1, 2, "2026-07-02", 100),
                       (2, 3, "2026-07-03", 95),
                       (2, 4, "2026-07-01", 95)],
                      columns=["src", "dst", "date", "sum_kzt"])
    signals = graph_edge_signals(edges, tx)
    assert set(signals) == {(1, 2), (2, 3)}
    assert signals[(2, 3)]["level"] == 2
    assert "95" in signals[(2, 3)]["text"]
    tx.loc[1, "date"] = "2026-07-02"
    signals = graph_edge_signals(edges, tx)
    assert signals[(2, 3)]["level"] == 1
    assert "порядок неизвестен" in signals[(2, 3)]["text"]
