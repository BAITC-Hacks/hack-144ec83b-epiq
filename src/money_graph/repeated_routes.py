"""Conservative repeated A→B→C episodes with independent date windows."""
from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = ["src", "via", "dst", "episode_count", "first_in", "last_out"]
EPISODE_COLUMNS = ["src", "via", "dst", "episode", "incoming_date", "outgoing_date",
                   "incoming_kzt", "outgoing_kzt", "delay_days", "amount_ratio"]


def repeated_routes(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate exact-edge pairs. Count >=2 disjoint 1–2-day windows per route.

    Amount ratio 0.8–1.2 is a hypothesis filter, not evidence of funds identity.
    Duplicated day/amount rows never create additional episodes. A payment may
    support different alternative routes; episode sums must not be added across routes.
    """
    tx = transactions[["src", "dst", "date", "sum_kzt"]].copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    tx = tx.drop_duplicates().sort_values(["date", "src", "dst", "sum_kzt"])
    incoming, outgoing = {}, {}
    for row in tx.itertuples(index=False):
        incoming.setdefault(int(row.dst), []).append(row)
        outgoing.setdefault(int(row.src), []).append(row)
    candidates = {}
    for via in sorted(set(incoming) & set(outgoing)):
        by_day = {}
        for out in outgoing[via]:
            by_day.setdefault(out.date, []).append(out)
        for entry in incoming[via]:
            if entry.sum_kzt <= 0:
                continue
            for delay in (1, 2):
                for exit_tx in by_day.get(entry.date + pd.Timedelta(days=delay), []):
                    key = (int(entry.src), via, int(exit_tx.dst))
                    ratio = float(exit_tx.sum_kzt / entry.sum_kzt)
                    if len(set(key)) == 3 and 0.8 <= ratio <= 1.2:
                        candidates.setdefault(key, []).append(
                            (exit_tx.date, entry.date, float(entry.sum_kzt),
                             float(exit_tx.sum_kzt), delay, ratio))
    summaries, episodes = [], []
    for key in sorted(candidates):
        chosen, last_exit = [], None
        # Earliest finishing windows; no reused operation/date window within a route.
        for event in sorted(set(candidates[key])):
            out_day, in_day, in_sum, out_sum, delay, ratio = event
            if last_exit is None or in_day > last_exit:
                chosen.append((*key, len(chosen) + 1, in_day, out_day,
                               in_sum, out_sum, delay, round(ratio, 6)))
                last_exit = out_day
        if len(chosen) >= 2:
            summaries.append((*key, len(chosen), chosen[0][4], chosen[-1][5]))
            episodes.extend(chosen)
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["episode_count", "src", "via", "dst"],
                                      ascending=[False, True, True, True]).reset_index(drop=True)
    return summary, pd.DataFrame(episodes, columns=EPISODE_COLUMNS)
