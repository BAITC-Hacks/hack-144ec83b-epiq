from __future__ import annotations

import networkx as nx
import pandas as pd


def build_resilience_report(graph: nx.DiGraph, scored: pd.DataFrame) -> pd.DataFrame:
    """Measure fragmentation after removing the highest-priority nodes."""
    ranked = scored.sort_values(["priority_score", "gid"], ascending=[False, True])
    baseline = graph.to_undirected()
    rows = []
    for requested in (0, 1, 5, 10, 20):
        n_removed = min(requested, len(baseline))
        candidate = baseline.copy()
        removed = ranked.head(n_removed)["gid"].astype(int).tolist()
        candidate.remove_nodes_from(removed)
        component_sizes = sorted(
            (len(component) for component in nx.connected_components(candidate)), reverse=True
        )
        remaining = len(candidate)
        largest = component_sizes[0] if component_sizes else 0
        rows.append(
            {
                "n_removed": n_removed,
                "removed_gids": ",".join(str(gid) for gid in removed),
                "remaining_nodes": remaining,
                "n_components": len(component_sizes),
                "largest_component": largest,
                "largest_component_share": round(largest / remaining, 6) if remaining else 0.0,
                "fragmentation": round(1 - largest / remaining, 6) if remaining else 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_route_patterns(
    graph: nx.DiGraph, scored: pd.DataFrame, limit: int = 200
) -> pd.DataFrame:
    """Return the strongest observed two-hop routes A→B→C."""
    node_info = scored.set_index("gid")
    rows = []
    for via in sorted(graph.nodes):
        predecessors = sorted(graph.predecessors(via))
        successors = sorted(graph.successors(via))
        if not predecessors or not successors:
            continue
        for source in predecessors:
            incoming = float(graph[source][via]["sum_kzt"])
            for target in successors:
                if source == target:
                    continue
                outgoing = float(graph[via][target]["sum_kzt"])
                rows.append(
                    {
                        "src": int(source),
                        "via": int(via),
                        "dst": int(target),
                        "bottleneck_kzt": round(min(incoming, outgoing), 2),
                        "incoming_kzt": round(incoming, 2),
                        "outgoing_kzt": round(outgoing, 2),
                        "via_role": str(node_info.loc[via, "role"]),
                        "via_priority": float(node_info.loc[via, "priority_score"]),
                        "rapid_signal": bool(node_info.loc[via].get("rapid_flow_days", 0) > 0),
                        "cycle_signal": bool(node_info.loc[via].get("cycle_size", 0) > 0),
                    }
                )
    columns = [
        "src", "via", "dst", "bottleneck_kzt", "incoming_kzt", "outgoing_kzt",
        "via_role", "via_priority", "rapid_signal", "cycle_signal",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(["bottleneck_kzt", "via_priority"], ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
