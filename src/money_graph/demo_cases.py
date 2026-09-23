"""Select reproducible demonstration cases from calculated facts, never fixed gids."""
from __future__ import annotations

import networkx as nx
from .investigations import seed_paths


def build_demo_cases(nodes, edges, repeats, episodes, manifest):
    graph = nx.DiGraph()
    graph.add_nodes_from(int(g) for g in nodes.gid)
    graph.add_edges_from((int(r.src), int(r.dst)) for r in edges.itertuples(index=False))
    all_paths = {int(s): seed_paths(graph, int(s)) for s in nodes.loc[nodes.is_seed, "gid"]}
    ranked = nodes.sort_values(["priority_score", "gid"], ascending=[False, True])
    ranks = {int(gid): i for i, gid in enumerate(ranked.gid, 1)}
    cases = {}
    lines = ["# Три проверяемых примера для защиты", "",
             "Сформированы автоматически из текущих данных. Идентификаторы не зашиты в алгоритм.",
             "Гипотезы не устанавливают виновность или движение одной и той же суммы.", ""]

    def add_node(kind, title, candidates):
        lines.extend([f"## {title}", ""])
        if candidates.empty:
            lines.extend(["На этом наборе подходящего примера нет.", ""])
            return None
        node = next(candidates.itertuples(index=False))
        gid = int(node.gid)
        paths = {str(s): [str(g) for g in paths[gid]] for s, paths in sorted(all_paths.items())
                 if gid in paths and s != gid}
        cases[kind] = {"gid": str(gid), "seeds": list(paths)[:5], "paths": paths,
                       "role": node.role, "rank": ranks[gid], "priority": node.priority_score}
        lines.extend([f"Найти **`{gid}`** через поле «Найти участника».", "",
                      f"- Роль: `{node.role}`; место {ranks[gid]}; приоритет {node.priority_score:.6f}.",
                      f"- Плательщиков: {node.in_deg}; получателей: {node.out_deg}.",
                      f"- Наблюдаемый вход: {node.in_kzt:,.0f} ₸; выход: {node.out_kzt:,.0f} ₸.",
                      f"- Обоснование: {node.evidence}", ""])
        if paths:
            lines.append("Кратчайшие направленные пути (до четырёх переходов):")
            lines.append("")
            lines.extend(f"- `{' → '.join(path)}`" for path in list(paths.values())[:5])
            lines.append("")
        return node

    collector = ranked[(ranked.role == "consolidator") & ~ranked.is_seed
                       & (ranked.depth < 4) & (ranked.seed_reach >= 2) & (ranked.pass_through <= 0.5)]
    node = add_node("collector", "1. Кто собирает деньги из нескольких цепочек", collector)
    if node is not None:
        lines.extend([f"В наблюдаемом периоде выход составляет {node.pass_through:.1%} входа.",
                      "Раскрыть «Почему такое место в топе» и сравнить вклад пяти факторов с другим кандидатом.",
                      "В «Общих получателях» выбрать первые 2–5 seed из путей выше и этого получателя.",
                      "Проверить каждое ребро и историю операций; суммы пересекающихся путей не складывать.", ""])

    lines.extend(["## 2. Повторение маршрута во времени", ""])
    if repeats.empty:
        lines.extend(["Маршрутов с двумя независимыми эпизодами на этом наборе нет.", ""])
    else:
        flow_nodes = set(nodes.loc[nodes.role.isin(["transit", "distributor"]), "gid"])
        flow_routes = repeats[repeats.via.isin(flow_nodes)]
        route = next((flow_routes if not flow_routes.empty else repeats).itertuples(index=False))
        chosen = episodes[(episodes.src == route.src) & (episodes.via == route.via) & (episodes.dst == route.dst)]
        cases["repeated_route"] = {"src": str(route.src), "via": str(route.via), "dst": str(route.dst),
                                   "episode_count": int(route.episode_count)}
        via = nodes[nodes.gid == route.via].iloc[0]
        lines.extend([f"В «Маршрутах» выбрать `{route.src} → {route.via} → {route.dst}`.",
                      f"Посредник имеет основную роль `{via['role']}`. Сигнал маршрута не подменяет основную роль.",
                      f"Независимых эпизодов: **{route.episode_count}**.", "",
                      "| Вход | Выход | Вход, ₸ | Выход, ₸ | Задержка, дней | Выход / вход |",
                      "|---|---|---:|---:|---:|---:|"])
        for row in chosen.itertuples(index=False):
            lines.append(f"| {str(row.incoming_date)[:10]} | {str(row.outgoing_date)[:10]} | "
                         f"{row.incoming_kzt:,.0f} | {row.outgoing_kzt:,.0f} | {row.delay_days} | {row.amount_ratio:.3f} |")
        lines.extend(["", "Открыть «Операции» для посредника и сверить переводы с исходной историей.",
                      "Правило: 1–2 дня, суммы 80–120%, окна не пересекаются. Это проверяемая гипотеза транзита.", ""])

    boundary = ranked[(ranked.depth == 4) & (ranked.out_deg == 0) & (ranked.in_kzt > 0)]
    node = add_node("boundary", "3. Где след обрывается и что запросить", boundary)
    if node is not None:
        lines.extend(["Показать глубину 4 и предупреждение об обрыве наблюдения.",
                      f"При нулевом наблюдаемом выходе узлу назначена роль `{node.role}`, а не `terminal`.",
                      f"Следующий запрос: {node.next_request}",
                      "Добавить клиента в перечень, указать решение и выгрузить CSV.", ""])
    lines.extend(["## Привязка к расчёту", "", f"Версия правил: `{manifest['rules_version']}`.", ""])
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in manifest["input_sha256"].items())
    return cases, "\n".join(lines) + "\n"
