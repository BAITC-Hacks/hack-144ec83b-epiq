"""Original pair-level temporal checks; see docs/open-source-review.md."""
from __future__ import annotations

import pandas as pd


def enrich_route_evidence(routes: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Find a dated witness on the exact two edges, without attributing funds."""
    result = routes.copy()
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    pairs = {
        (int(src), int(dst)): list(group[["date", "sum_kzt"]].itertuples(index=False, name=None))
        for (src, dst), group in tx.groupby(["src", "dst"])
    }
    evidence = []
    for row in routes.itertuples(index=False):
        witnesses = []
        for in_date, in_amount in pairs.get((int(row.src), int(row.via)), []):
            for out_date, out_amount in pairs.get((int(row.via), int(row.dst)), []):
                delay = (out_date - in_date).days
                if 0 <= delay <= 2 and in_amount > 0:
                    ratio = float(out_amount / in_amount)
                    witnesses.append((delay, abs(1 - ratio), in_date, out_date,
                                      float(in_amount), float(out_amount), ratio))
        # Prefer known day order and comparable amounts; choose deterministically.
        witnesses.sort(key=lambda w: (not (w[0] > 0 and 0.8 <= w[6] <= 1.2),
                                      not (w[0] > 0), w[1], w[0], w[2], w[4], w[5]))
        if witnesses:
            delay, _, in_date, out_date, incoming, outgoing, ratio = witnesses[0]
            status = "Порядок по дням соблюдён" if delay > 0 else "Один день: порядок неизвестен"
            evidence.append((True, status, delay, in_date, out_date, incoming, outgoing,
                             round(ratio, 4), 0.8 <= ratio <= 1.2))
        else:
            evidence.append((False, "Нет пары в окне 0–2 дня", None, None, None,
                             None, None, None, False))
    columns = ["rapid_signal", "temporal_status", "delay_days", "incoming_date",
               "outgoing_date", "example_in_kzt", "example_out_kzt", "amount_ratio",
               "comparable_amounts"]
    details = pd.DataFrame(evidence, columns=columns, index=result.index)
    for column in columns:
        result[column] = details[column]
    return result
