from __future__ import annotations

import json

import numpy as np
import pandas as pd


ROLE_ORDER = ["coordinator", "consolidator", "distributor", "transit", "terminal"]


def _sat(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _positive_percentile(series: pd.Series) -> pd.Series:
    positive = series[series > 0]
    result = pd.Series(0.0, index=series.index)
    if not positive.empty:
        result.loc[positive.index] = positive.rank(method="average", pct=True)
    return result


def score_nodes(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["p_in"] = _positive_percentile(df["in_kzt"])
    df["p_out"] = _positive_percentile(df["out_kzt"])
    df["p_turnover"] = _positive_percentile(df[["in_kzt", "out_kzt"]].max(axis=1))
    df["p_between"] = _positive_percentile(df["betweenness"])
    df["p_degree"] = _positive_percentile(df["in_deg"] + df["out_deg"])

    roles: list[str] = []
    role_scores: list[float] = []
    ambiguous: list[bool] = []
    evidence: list[str] = []

    for row in df.itertuples(index=False):
        candidates: dict[str, float] = {}
        if row.in_deg >= 3:
            candidates["consolidator"] = (
                0.50 * _sat(row.in_deg / 8)
                + 0.30 * row.p_in
                + 0.20 * _sat(row.seed_reach / 3)
            )
        if row.out_deg >= 5:
            candidates["distributor"] = (
                0.60 * _sat(row.out_deg / 10)
                + 0.25 * row.p_out
                + 0.15 * row.out_entropy
            )
        if (
            row.in_kzt > 0
            and row.out_kzt > 0
            and 0.8 <= row.pass_through <= 1.2
            and not row.is_seed
            and row.depth < 4
        ):
            candidates["transit"] = 0.70 * _sat(
                1 - abs(row.pass_through - 1) / 0.2
            ) + 0.30 * _sat(min(row.in_tx, row.out_tx) / 3)
        if (
            row.in_kzt > 0
            and row.pass_through <= 0.1
            and not row.is_seed
            and row.depth < 4
            and row.days_after_last_in >= 2
        ):
            candidates["terminal"] = (
                0.50 * _sat(1 - row.pass_through / 0.1)
                + 0.30 * row.p_in
                + 0.20 * _sat(row.in_deg / 3)
            )
        if (
            row.seed_reach >= 2
            and row.betweenness > 0
            and row.p_between >= 0.9
            and row.external_clusters >= 2
            and row.in_deg > 0
            and row.out_deg > 0
        ):
            candidates["coordinator"] = (
                0.50 * row.p_between
                + 0.30 * _sat(row.seed_reach / 5)
                + 0.20 * _sat(row.external_clusters / 3)
            )

        if not candidates:
            roles.append("peripheral")
            role_scores.append(0.0)
            ambiguous.append(False)
            if row.is_isolate:
                text = f"Связей: 0; seed={bool(row.is_seed)}, depth={row.depth}. Данных для роли недостаточно"
            elif row.truncated_by_depth:
                text = f"Граница depth=4; входов {row.in_deg}, {row.in_kzt:,.0f} KZT. Дальнейший поток неизвестен"
            else:
                text = f"Входов {row.in_deg}, выходов {row.out_deg}; выраженных признаков роли нет"
            evidence.append(text[:200])
            continue

        ordered = sorted(
            candidates.items(), key=lambda item: (-item[1], ROLE_ORDER.index(item[0]))
        )
        role, raw_score = ordered[0]
        is_ambiguous = len(ordered) > 1 and raw_score - ordered[1][1] < 0.10
        cap = 0.60 if role in {"coordinator", "terminal"} else 0.85
        if row.depth == 4:
            cap = min(cap, 0.70)
        if is_ambiguous:
            cap = min(cap, 0.60)
        score = min(raw_score, cap)

        if role == "consolidator":
            text = f"Сбор: {row.in_deg} плательщиков, вход {row.in_kzt:,.0f} KZT, достижим из {row.seed_reach} seed"
        elif role == "distributor":
            text = f"Распределение: {row.out_deg} получателей, выход {row.out_kzt:,.0f} KZT, операций {row.out_tx}"
        elif role == "transit":
            text = f"Транзит: выход/вход {row.pass_through:.2f}, входящих операций {row.in_tx}, исходящих {row.out_tx}"
        elif role == "terminal":
            text = f"Кандидат в получатели: выход/вход {row.pass_through:.2f}, вход {row.in_kzt:,.0f} KZT"
        else:
            text = f"Связующий узел: {row.seed_reach} seed, {row.external_clusters} соседних кластеров, P(B)={row.p_between:.2f}"
        if is_ambiguous:
            text += f"; также {ordered[1][0]}"
        roles.append(role)
        role_scores.append(round(float(score), 6))
        ambiguous.append(is_ambiguous)
        evidence.append(text[:200])

    df["role"] = roles
    df["role_score"] = role_scores
    df["role_ambiguous"] = ambiguous
    df["evidence"] = evidence
    df["priority_score"] = (
        0.30 * (df["seed_reach"] / 5).clip(0, 1)
        + 0.25 * df["p_turnover"]
        + 0.20 * df["p_between"]
        + 0.15 * df["p_degree"]
        + 0.10 * df["role_score"]
    ).clip(0, 1).round(6)
    df.loc[df["is_isolate"], "priority_score"] = 0.0

    turnover = df[["in_kzt", "out_kzt"]].max(axis=1)
    degree = df["in_deg"] + df["out_deg"]
    df["depth_turnover_pct"] = turnover.groupby(df["depth"]).rank(method="average", pct=True)
    df["depth_degree_pct"] = degree.groupby(df["depth"]).rank(method="average", pct=True)
    rapid_days = df.get("rapid_flow_days", pd.Series(0, index=df.index))
    cycle_sizes = df.get("cycle_size", pd.Series(0, index=df.index))
    rapid_signal = (rapid_days > 0).astype(float)
    cycle_signal = (cycle_sizes > 0).astype(float)
    df["anomaly_score"] = (
        0.45 * df["depth_turnover_pct"]
        + 0.25 * df["depth_degree_pct"]
        + 0.15 * df["p_between"]
        + 0.10 * rapid_signal
        + 0.05 * cycle_signal
    ).clip(0, 1).round(6)

    anomaly_flags = []
    for row in df.itertuples(index=False):
        flags = []
        if row.depth_turnover_pct >= 0.95:
            flags.append("оборот выше 95% узлов своего колена")
        if row.depth_degree_pct >= 0.95:
            flags.append("связность выше 95% узлов своего колена")
        if getattr(row, "rapid_flow_days", 0) > 0:
            flags.append("входящие и исходящие операции в окне 0–2 дня")
        if getattr(row, "cycle_size", 0) > 0:
            flags.append(f"участник цикла из {int(row.cycle_size)} узлов")
        if getattr(row, "max_same_day_payers", 0) >= 3:
            flags.append(
                f"до {int(row.max_same_day_payers)} плательщиков переводили в один день"
            )
        if getattr(row, "incoming_spike_ratio", 0) >= 3:
            flags.append(
                f"дневной входящий всплеск x{float(row.incoming_spike_ratio):.1f} к медиане"
            )
        anomaly_flags.append("; ".join(flags) if flags else "выраженных аномалий не выявлено")
    df["anomaly_flags"] = anomaly_flags
    return df


def build_cluster_summary(
    scored: pd.DataFrame, clusters: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    cluster_by_gid = scored.set_index("gid")["cluster_id"].to_dict()
    edge_clusters = edges.copy()
    edge_clusters["src_cluster"] = edge_clusters["src"].map(cluster_by_gid)
    edge_clusters["dst_cluster"] = edge_clusters["dst"].map(cluster_by_gid)
    internal = edge_clusters[edge_clusters["src_cluster"] == edge_clusters["dst_cluster"]]
    internal_sum = internal.groupby("src_cluster")["sum_kzt"].sum().to_dict()

    rows = []
    for cluster in clusters.itertuples(index=False):
        members = set(cluster.members)
        part = scored[scored["gid"].isin(members)].sort_values(
            ["priority_score", "gid"], ascending=[False, True]
        )
        role_counts = part[part["role"] != "peripheral"]["role"].value_counts()
        if role_counts.empty:
            hypothesis = "Связи без выраженной функциональной роли"
        else:
            labels = {
                "consolidator": "сбор средств",
                "distributor": "распределение",
                "transit": "транзит",
                "terminal": "конечные получатели",
                "coordinator": "структурная координация",
            }
            leading = labels[role_counts.index[0]]
            hypothesis = f"Наблюдаются признаки функции: {leading}"
        rows.append(
            {
                "cluster_id": int(cluster.cluster_id),
                "n_nodes": int(len(part)),
                "n_seed": int(part["is_seed"].sum()),
                "sum_kzt_internal": round(float(internal_sum.get(cluster.cluster_id, 0.0)), 2),
                "top_gids": json.dumps(part.head(5)["gid"].astype(int).tolist()),
                "hypothesis": hypothesis,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

