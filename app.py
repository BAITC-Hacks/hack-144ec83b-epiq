from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from money_graph.investigation_ui import show_common_recipients, show_repeated_routes
from money_graph.scoring import PRIORITY_LABELS, priority_components
from money_graph.assistant_tools import GraphTools
from money_graph.assistant_ui import render_assistant

ROLE_LABELS = {
    "coordinator": "Координатор",
    "consolidator": "Аккумулятор",
    "distributor": "Распределитель",
    "transit": "Транзит",
    "terminal": "Конечный узел",
    "peripheral": "Периферия",
}
ROLE_COLORS = {
    "coordinator": "violet",
    "consolidator": "blue",
    "distributor": "orange",
    "transit": "blue",
    "terminal": "green",
    "peripheral": "gray",
}
GRAPH_COLORS = {
    "coordinator": "#B9A2F4",
    "consolidator": "#8CABD9",
    "distributor": "#D9AD89",
    "transit": "#83BDB7",
    "terminal": "#A6BC94",
    "peripheral": "#9295A5",
}


@st.cache_data(max_entries=4)
def load_results(out_dir: str, data_dir: str, version: int = 0) -> tuple[pd.DataFrame, ...]:
    root = Path(out_dir)
    transactions_path = Path(data_dir) / "transactions.parquet"
    transactions = (
        pd.read_parquet(transactions_path)
        if transactions_path.exists()
        else pd.DataFrame(columns=["src", "dst", "date", "sum_kzt"])
    )
    if not transactions.empty:
        transactions["date"] = pd.to_datetime(transactions["date"])
    resilience_path = root / "resilience.csv"
    route_patterns_path = root / "route_patterns.csv"
    return (
        pd.read_csv(root / "node_features.csv"),
        pd.read_csv(root / "edges.csv"),
        pd.read_csv(root / "clusters.csv"),
        pd.read_csv(root / "top_nodes.csv"),
        transactions,
        pd.read_csv(resilience_path) if resilience_path.exists() else pd.DataFrame(),
        pd.read_csv(route_patterns_path) if route_patterns_path.exists() else pd.DataFrame(),
        pd.read_csv(root / "repeated_routes.csv") if (root / "repeated_routes.csv").exists() else pd.DataFrame(),
        pd.read_csv(root / "route_episodes.csv") if (root / "route_episodes.csv").exists() else pd.DataFrame(),
    )


def compact_number(value: float, suffix: str = "") -> str:
    amount = float(value)
    for divider, marker in ((1_000_000_000, "млрд"), (1_000_000, "млн"), (1_000, "тыс")):
        if abs(amount) >= divider:
            return f"{amount / divider:.1f} {marker} {suffix}".strip()
    return f"{amount:,.0f} {suffix}".replace(",", " ").strip()


def node_label(gid: int, selected_gid: int) -> str:
    value = str(int(gid))
    return value if int(gid) == selected_gid else f"…{value[-6:]}"


def analyst_rationale(row: pd.Series) -> str:
    payers = int(row["in_deg"])
    recipients = int(row["out_deg"])
    incoming = float(row["in_kzt"])
    outgoing = float(row["out_kzt"])

    if incoming > 0 and outgoing > 0:
        forwarded_share = outgoing / incoming * 100
        if forwarded_share > 150:
            outgoing_multiple = outgoing / incoming
            return (
                f"Получает от {payers} плательщиков и переводит {recipients} получателям. "
                f"Исходящий объём в {outgoing_multiple:.1f} раза выше наблюдаемого входящего; "
                "входящий контур может быть неполным."
            )
        return (
            f"Получает от {payers} плательщиков и переводит {forwarded_share:.0f}% "
            f"входящего объёма {recipients} получателям."
        )
    if incoming > 0:
        return f"Получает от {payers} плательщиков; исходящие переводы в выгрузке не наблюдаются."
    if outgoing > 0:
        return (
            f"Переводит деньги {recipients} получателям; входящий поток отсутствует "
            "или не попал в наблюдаемый фрагмент."
        )
    return "Наблюдаемых входящих и исходящих переводов нет."


def risk_signals(row: pd.Series, turnover_cutoff: float) -> list[str]:
    signals = []
    if int(row["in_deg"]) >= 5:
        signals.append("Сбор средств")
    if int(row["out_deg"]) >= 10:
        signals.append("Веерные переводы")
    if 0.8 <= float(row["pass_through"]) <= 1.2 and int(row["in_tx"]) > 0:
        signals.append("Транзит")
    if int(row["seed_reach"]) >= 2:
        signals.append("Несколько seed")
    if float(row["in_kzt"]) + float(row["out_kzt"]) >= turnover_cutoff:
        signals.append("Высокий оборот")
    if float(row.get("anomaly_score", 0)) >= 0.8:
        signals.append("Аномалия колена")
    if int(row.get("rapid_flow_days", 0)) > 0:
        signals.append("Быстрый транзит")
    if int(row.get("cycle_size", 0)) > 0:
        signals.append("Цикл")
    if int(row.get("max_same_day_payers", 0)) >= 3:
        signals.append("Синхронные плательщики")
    if float(row.get("incoming_spike_ratio", 0)) >= 3:
        signals.append("Всплеск активности")
    if bool(row["truncated_by_depth"]):
        signals.append("Обрыв depth=4")
    return signals


def trace_from_seed(nodes: pd.DataFrame, edges: pd.DataFrame, target_gid: int) -> list[int]:
    seeds = set(nodes.loc[nodes["is_seed"].astype(bool), "gid"].astype(int))
    if target_gid in seeds:
        return [target_gid]

    incoming: dict[int, list[tuple[int, float]]] = {}
    for edge in edges.itertuples(index=False):
        incoming.setdefault(int(edge.dst), []).append((int(edge.src), float(edge.sum_kzt)))
    for predecessors in incoming.values():
        predecessors.sort(key=lambda item: item[1], reverse=True)

    queue = deque([target_gid])
    next_node: dict[int, int | None] = {target_gid: None}
    found_seed = None
    while queue and found_seed is None:
        current = queue.popleft()
        for predecessor, _ in incoming.get(current, []):
            if predecessor in next_node:
                continue
            next_node[predecessor] = current
            if predecessor in seeds:
                found_seed = predecessor
                break
            queue.append(predecessor)

    if found_seed is None:
        return []
    path = [found_seed]
    while path[-1] != target_gid:
        successor = next_node[path[-1]]
        if successor is None:
            break
        path.append(successor)
    return path


def answer_analyst_query(query: str, nodes: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Interpret a small set of investigation questions without an external API."""
    normalized = query.lower().strip()
    gid_match = re.search(r"\d{12,}", normalized)
    if gid_match:
        gid = int(gid_match.group())
        result = nodes[nodes["gid"] == gid].copy()
        return (
            ("Участник найден. Откройте его карточку через поиск по gid."
             if not result.empty else "Такой gid отсутствует в текущем графе."),
            result,
        )

    if any(word in normalized for word in ("собира", "аккум", "консолид")):
        result = nodes[nodes["role"] == "consolidator"].copy()
        answer = "Показываю приоритетные точки сбора средств."
    elif any(word in normalized for word in ("распредел", "веер", "получател")):
        result = nodes[nodes["role"] == "distributor"].copy()
        answer = "Показываю узлы с веерным распределением средств."
    elif any(word in normalized for word in ("транзит", "быстро", "день")):
        rapid = nodes.get("rapid_flow_days", pd.Series(0, index=nodes.index))
        result = nodes[(nodes["role"] == "transit") | (rapid > 0)].copy()
        answer = "Показываю транзитные узлы и операции в окне до двух дней."
    elif any(word in normalized for word in ("аномал", "необыч", "подозр")):
        result = nodes.copy()
        answer = "Показываю узлы с наибольшей аномальностью относительно своего колена."
    elif any(word in normalized for word in ("координ", "организ", "выше")):
        result = nodes[nodes["role"] == "coordinator"].copy()
        answer = "Показываю структурных координаторов, связывающих seed и кластеры."
    else:
        result = nodes.copy()
        answer = "Запрос не распознан точно. Показываю общий приоритет проверки."

    sort_column = "anomaly_score" if "аномал" in normalized and "anomaly_score" in result else "priority_score"
    return answer, result.sort_values(sort_column, ascending=False).head(10)


def select_graph_neighborhood(
    edges: pd.DataFrame,
    selected_gid: int,
    radius: int,
    edge_limit: int,
) -> pd.DataFrame:
    """Select the strongest connected edges hop by hop around one participant."""
    frontier = {selected_gid}
    visited_nodes = {selected_gid}
    selected_indices: list[int] = []

    for hop in range(radius):
        candidates = edges[
            edges["src"].isin(frontier) | edges["dst"].isin(frontier)
        ].drop(index=selected_indices, errors="ignore")
        if candidates.empty:
            break
        remaining = edge_limit - len(selected_indices)
        if remaining <= 0:
            break
        hops_left = radius - hop
        quota = remaining if hops_left == 1 else max(1, math.ceil(remaining / hops_left))
        chosen = candidates.sort_values("sum_kzt", ascending=False).head(quota)
        selected_indices.extend(chosen.index.tolist())
        reached = set(chosen["src"].astype(int)) | set(chosen["dst"].astype(int))
        frontier = reached - visited_nodes
        visited_nodes |= reached
        if not frontier:
            break

    return edges.loc[selected_indices].copy()


def render_neighborhood(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    selected_gid: int,
    radius: int = 1,
    edge_limit: int = 12,
    transactions: pd.DataFrame | None = None,
    glow: bool = True,
) -> None:
    from pyvis.network import Network
    from money_graph.route_evidence import graph_edge_signals
    import networkx as nx

    neighborhood = select_graph_neighborhood(edges, selected_gid, radius, edge_limit)
    signals = graph_edge_signals(neighborhood, transactions) if transactions is not None else {}
    visible_gids = {selected_gid} | set(neighborhood["src"].astype(int)) | set(
        neighborhood["dst"].astype(int)
    )
    visible_nodes = nodes[nodes["gid"].isin(visible_gids)]
    background = "#19191F"
    font_color = "#B8B7C6"
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(sorted(visible_gids))
    layout_graph.add_edges_from(
        (int(row.src), int(row.dst)) for row in neighborhood.itertuples(index=False)
    )
    positions = nx.spring_layout(layout_graph, seed=42, iterations=100, scale=300)

    network = Network(
        height="570px",
        width="100%",
        directed=True,
        bgcolor=background,
        font_color=font_color,
        cdn_resources="in_line",
    )
    for row in visible_nodes.itertuples(index=False):
        is_selected = int(row.gid) == selected_gid
        role_name = ROLE_LABELS.get(row.role, row.role)
        node_color = GRAPH_COLORS.get(row.role, "#64748B")
        gid_text = str(int(row.gid))
        compact_label = gid_text if is_selected else f"…{gid_text[-6:]}"
        network.add_node(
            gid_text,
            label=f"{role_name}\n{compact_label}" if is_selected else compact_label,
            title=(
                f"gid: {int(row.gid)}\nРоль: {role_name}\n"
                f"Колено: {int(row.depth)}\nПриоритет: {row.priority_score:.2f}\n"
                f"Входящий поток: {float(row.in_kzt):,.0f} ₸\n"
                f"Исходящий поток: {float(row.out_kzt):,.0f} ₸\n{row.evidence}"
            ),
            color={
                "background": node_color,
                "border": "#EEE8FF" if is_selected else background,
                "highlight": {"background": "#E1D5FF", "border": "#F6F2FF"},
                "hover": {"background": node_color, "border": "#E1D5FF"},
            },
            size=15 if is_selected else 5 + 7 * float(row.priority_score),
            shape="dot",
            borderWidth=2 if is_selected or bool(row.is_seed) else 0,
            x=float(positions[int(row.gid)][0]),
            y=float(positions[int(row.gid)][1]),
            shadow={"enabled": is_selected, "color": "rgba(185,162,244,0.25)",
                    "size": 22, "x": 0, "y": 0},
        )
    max_amount = max(float(neighborhood["sum_kzt"].max()), 1.0)
    for row in neighborhood.itertuples(index=False):
        signal = signals.get((int(row.src), int(row.dst)))
        edge_color = "#626072" if selected_gid in (int(row.src), int(row.dst)) else "#42424E"
        if signal:
            edge_color = "#FF8D9B" if signal["level"] == 2 else "#EBC17B"
        relative_width = math.log10(float(row.sum_kzt) + 1) / math.log10(max_amount + 1)
        network.add_edge(
            str(int(row.src)),
            str(int(row.dst)),
            width=(1.4 if signal else 0.4) + 1.1 * relative_width,
            title=(f"Всего по связи: {float(row.sum_kzt):,.0f} ₸ · {int(row.n_tx)} операций\n\n"
                   + (signal["text"] if signal else "На видимых двухшаговых маршрутах сигнал не найден. Это не оценка безопасности.")),
            arrows="to",
            color={"color": edge_color, "highlight": edge_color if signal else "#C4B1EF",
                   "hover": "#FFE9ED" if signal else "#C4B1EF", "opacity": 1 if signal else 0.55},
            shadow={"enabled": bool(signal) and glow, "color": edge_color,
                    "size": 16, "x": 0, "y": 0},
        )
    network.set_options(json.dumps({
        "layout": {"improvedLayout": False},
        "interaction": {
            "hover": True,
            "navigationButtons": False,
            "keyboard": True,
            "tooltipDelay": 120,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True,
        },
        "physics": {"enabled": False},
        "edges": {
            "arrowStrikethrough": False,
            "smooth": {"enabled": True, "type": "continuous", "roundness": 0.12},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.28}},
        },
        "nodes": {
            "font": {
                "size": 11,
                "face": "Inter, Arial",
                "color": font_color,
                "strokeWidth": 2,
                "strokeColor": background,
            }
        },
    }))
    graph_html = network.generate_html().replace(
        "</head>",
        "<style>html,body{margin:0;background:#19191F;}"
        ".card{border:0!important;background:transparent!important;}"
        "#mynetwork{border:0!important;border-radius:14px;}"
        "div.vis-tooltip{background:#292832;color:#EEEAF6;border:1px solid #514C63;"
        "border-radius:8px;padding:12px;font:12px sans-serif;max-width:360px;white-space:pre-line;}"
        "</style></head>",
    )
    st.iframe(graph_html, width="stretch", height=590)


def render_role_badge(role: str) -> None:
    st.badge(ROLE_LABELS.get(role, role), color=ROLE_COLORS.get(role, "gray"))


st.set_page_config(
    page_title="Граф денег",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("### :green[:material/account_tree:] Граф денег")
    st.caption("Кейс Freedom · прототип HackAlem")
    st.divider()
    gid_value = st.text_input(
        "Найти участника",
        placeholder="Введите gid",
        icon=":material/search:",
        type="search",
    )
    with st.expander("Источник данных", icon=":material/database:"):
        out_dir = st.text_input(
            "Папка результатов",
            os.getenv("MONEY_GRAPH_OUT", "out"),
            help="Каталог с результатами расчётного пайплайна.",
        )
        data_dir = st.text_input(
            "Папка исходных данных",
            os.getenv("MONEY_GRAPH_DATA", "data"),
            help="Каталог с transactions.parquet для просмотра истории операций.",
        )

try:
    nodes, edges, clusters, top, transactions, resilience, route_patterns, repeats, episodes = load_results(
        out_dir, data_dir,
        version=(Path(out_dir) / "run_manifest.json").stat().st_mtime_ns
    )
except FileNotFoundError:
    st.error("Результаты анализа не найдены", icon=":material/folder_off:")
    st.info("Запустите расчётный пайплайн, затем укажите папку результатов в боковой панели.")
    st.code("python -m money_graph.cli --data data --out out", language="powershell")
    st.stop()

role_options = sorted(nodes["role"].dropna().unique())
investigation_scenarios = {
    "Общий приоритет": ([], "Все кандидаты с учётом выбранных фильтров."),
    "Кто собирает деньги": (["Сбор средств", "Несколько seed"],
                           "Ищем точки сбора и участников, связанных с несколькими исходными клиентами."),
    "Куда деньги уходят дальше": (["Быстрый транзит", "Веерные переводы"],
                                 "Ищем быстрый выход и распределение; проверяем даты в разделе «Маршруты»."),
    "Где обрывается след": (["Обрыв depth=4"],
                           "Выбираем пограничные узлы, для которых нужен следующий запрос данных."),
}
with st.sidebar:
    scenario = st.selectbox("Сценарий проверки", list(investigation_scenarios))
    scenario_signals, scenario_description = investigation_scenarios[scenario]
    st.caption(scenario_description)
    selected_roles = st.pills(
        "Роли",
        role_options,
        default=role_options,
        format_func=lambda role: ROLE_LABELS.get(role, role),
        selection_mode="multi",
        wrap=True,
    )
    min_priority = st.slider(
        "Минимальный приоритет",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )
    selected_signals = st.pills(
        "Подозрительная активность",
        [
            "Сбор средств", "Веерные переводы", "Транзит", "Несколько seed",
            "Высокий оборот", "Аномалия колена", "Быстрый транзит", "Цикл",
            "Синхронные плательщики", "Всплеск активности",
        ],
        selection_mode="multi",
        wrap=True,
        help="При нескольких вариантах достаточно совпадения хотя бы с одним сигналом.",
    )
    st.divider()
    st.caption("Система предлагает гипотезы. Решение принимает аналитик.")

header = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
with header:
    with st.container():
        st.caption("FREEDOM · КЕЙС HACKALEM · AML-АНАЛИТИКА")
        st.title("От перевода — к цепочке")
        st.caption("Найдите точку сбора, проверьте путь денег и подготовьте следующий запрос.")
    st.badge("4 колена · локальный анализ", icon=":material/account_tree:", color="green")

metrics = st.container(border=True, horizontal=True, horizontal_alignment="distribute")
with metrics:
    st.metric("Участники", f"{len(nodes):,}".replace(",", " "), border=False)
    st.metric("Связи", f"{len(edges):,}".replace(",", " "), border=False)
    st.metric("Кластеры", f"{len(clusters):,}".replace(",", " "), border=False)
    st.metric("Оборот", compact_number(edges["sum_kzt"].sum(), "₸"), border=False)

active_roles = selected_roles or role_options
turnover_cutoff = float((nodes["in_kzt"] + nodes["out_kzt"]).quantile(0.9))
ranked_nodes = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()
ranked_nodes["rank"] = range(1, len(ranked_nodes) + 1)
ranked_nodes["signal_list"] = ranked_nodes.apply(
    lambda row: risk_signals(row, turnover_cutoff), axis=1
)
priority = ranked_nodes[
    ranked_nodes["role"].isin(active_roles)
    & (ranked_nodes["priority_score"] >= min_priority)
].copy()
if selected_signals:
    priority = priority[
        priority["signal_list"].apply(
            lambda values: any(signal in values for signal in selected_signals)
        )
    ]
if scenario_signals:
    priority = priority[
        priority["signal_list"].apply(
            lambda values: any(signal in values for signal in scenario_signals)
        )
    ]
matched_count = len(priority)
priority = priority.head(200)
priority["gid_display"] = priority["gid"].astype(str)
priority["role_display"] = priority["role"].map(ROLE_LABELS).fillna(priority["role"])
priority["signals_display"] = priority["signal_list"].apply(lambda values: " · ".join(values))
rationale_by_gid = {
    int(row["gid"]): analyst_rationale(row) for _, row in nodes.iterrows()
}
priority["rationale"] = priority["gid"].map(rationale_by_gid)

selected_gid = st.session_state.get("selected_gid")
query_error = None
if gid_value.strip():
    try:
        query_gid = int(gid_value.strip())
        if query_gid in set(nodes["gid"]):
            selected_gid = query_gid
        else:
            query_error = "Участник с таким gid не найден"
    except ValueError:
        query_error = "gid должен состоять только из цифр"

if selected_gid not in set(nodes["gid"]):
    selected_gid = int(priority.iloc[0]["gid"]) if not priority.empty else int(nodes.iloc[0]["gid"])

st.subheader("Приоритет проверки")
if query_error:
    st.warning(query_error, icon=":material/search_off:")

table_col, card_col = st.columns([1.75, 1], gap="large")
with table_col:
    with st.container(border=True):
        st.caption(f"Найдено: {matched_count} · показано: {len(priority)} · выберите участника")
        if priority.empty:
            st.info("Нет участников по этим условиям. Измените сценарий, роли или минимальный приоритет.")
            table_event = None
        else:
            table_event = st.dataframe(
                priority[["gid_display", "rationale", "signals_display", "priority_score"]],
                width="stretch",
                height=420,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="priority_table",
                column_config={
                    "gid_display": st.column_config.TextColumn("gid", width="medium", pinned=True),
                    "rationale": st.column_config.TextColumn("Обоснование", width="large"),
                    "signals_display": st.column_config.TextColumn("Сигналы", width="medium"),
                    "priority_score": st.column_config.ProgressColumn(
                        "Приоритет", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )
            if table_event.selection.rows:
                selected_gid = int(priority.iloc[table_event.selection.rows[0]]["gid"])

st.session_state["selected_gid"] = selected_gid
node = nodes[nodes["gid"] == selected_gid].iloc[0]
node_rationale = analyst_rationale(node)
node_signals = risk_signals(node, turnover_cutoff)
stored_review_cases = st.session_state.setdefault("review_cases", {})
valid_gids = set(nodes["gid"].astype(int))
review_cases = {
    int(review_gid): review_route
    for review_gid, review_route in stored_review_cases.items()
    if int(review_gid) in valid_gids
}
st.session_state["review_cases"] = review_cases
review_routes = [
    "Углублённая проверка",
    "Запрос в правоохранительные органы",
    "Проверка и запрос",
]

with card_col:
    with st.container(border=True):
        title_row = st.container(horizontal=True, horizontal_alignment="distribute")
        with title_row:
            st.subheader(str(selected_gid))
            render_role_badge(str(node["role"]))
        st.caption(f"Кластер {int(node['cluster_id'])}")
        scores = st.container(horizontal=True, horizontal_alignment="distribute")
        with scores:
            st.metric("Сила роли", f"{float(node['role_score']):.2f}")
            st.metric("Приоритет", f"{float(node['priority_score']):.2f}")
            st.metric("Глубина", int(node["depth"]))
        st.markdown("**Основание для проверки**")
        st.write(node_rationale)
        st.caption("Сигналы: " + (" · ".join(node_signals) if node_signals else "не выявлены"))
        st.caption(node["evidence"])
        if "data_gap" in node.index and "next_request" in node.index:
            with st.expander("Какой запрос сделать следующим", icon=":material/outgoing_mail:"):
                st.write(node["data_gap"])
                st.markdown(f"**Рекомендуемое действие:** {node['next_request']}")
        if bool(node["truncated_by_depth"]):
            st.warning("Данные заканчиваются на глубине 4. Следующие переводы не видны.")
        current_route = review_cases.get(selected_gid, review_routes[0])
        route = st.selectbox(
            "Решение аналитика",
            review_routes,
            index=review_routes.index(current_route),
            key=f"review_route_{selected_gid}",
        )
        if st.button(
            "Обновить решение" if selected_gid in review_cases else "Добавить в перечень",
            icon=":material/playlist_add_check:",
            type="primary",
            width="stretch",
        ):
            review_cases[selected_gid] = route
            st.session_state["review_cases"] = review_cases
            st.toast("Решение сохранено", icon=":material/check_circle:")

with st.expander("Почему такое место в топе", icon=":material/leaderboard:"):
    current_parts = priority_components(nodes[nodes["gid"] == selected_gid]).iloc[0]
    rank = int(ranked_nodes.loc[ranked_nodes["gid"] == selected_gid, "rank"].iloc[0])
    st.write(f"Место {rank} из {len(nodes)} · итог {float(node['priority_score']):.6f}")
    compare_options = [str(int(g)) for g in ranked_nodes.head(20)["gid"] if int(g) != selected_gid]
    compare_gid = st.selectbox("Сравнить с участником из топ-20", compare_options,
                               key=f"compare_{selected_gid}")
    other = nodes[nodes["gid"] == int(compare_gid)] if compare_gid else None
    other_parts = priority_components(other).iloc[0] if other is not None else None
    rows = []
    for field, (label, weight) in PRIORITY_LABELS.items():
        part = {"Фактор": label, "Вклад": float(current_parts[field]), "Предел": weight}
        if other_parts is not None:
            part["У сравниваемого"] = float(other_parts[field])
            part["Разница"] = float(current_parts[field] - other_parts[field])
        rows.append(part)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Вклад": st.column_config.NumberColumn(format="%.4f"),
        "Предел": st.column_config.NumberColumn(format="%.2f"),
        "У сравниваемого": st.column_config.NumberColumn(format="%.4f"),
        "Разница": st.column_config.NumberColumn(format="%.4f"),
    })
    if other is not None:
        difference = float(node["priority_score"] - other["priority_score"].iloc[0])
        st.caption(f"Разница итогового приоритета: {difference:+.6f}.")
    st.caption("Вклады суммируются до итогового приоритета с округлением до 6 знаков. "
               "Охват насыщается на 5 seed; оборот, посредничество и связность — "
               "перцентиль среди положительных значений. При равенстве итогов порядок по gid.")

st.subheader("Перечень для дальнейших действий")
if not review_cases:
    st.info("Выберите участника, укажите решение и добавьте его в перечень.")
else:
    review_rows = []
    for review_gid, review_route in review_cases.items():
        review_node = nodes[nodes["gid"] == int(review_gid)].iloc[0]
        review_rows.append(
            {
                "gid": str(int(review_gid)),
                "role": ROLE_LABELS.get(str(review_node["role"]), str(review_node["role"])),
                "priority": float(review_node["priority_score"]),
                "route": review_route,
                "rationale": analyst_rationale(review_node),
                "cluster": int(review_node["cluster_id"]),
            }
        )
    review_df = pd.DataFrame(review_rows).sort_values("priority", ascending=False)
    st.dataframe(
        review_df[["gid", "rationale", "route", "priority", "role", "cluster"]],
        width="stretch",
        hide_index=True,
        column_config={
            "gid": st.column_config.TextColumn("gid", pinned=True),
            "role": st.column_config.TextColumn("Роль"),
            "priority": st.column_config.ProgressColumn(
                "Приоритет", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "route": st.column_config.TextColumn("Решение", width="medium"),
            "rationale": st.column_config.TextColumn("Обоснование", width="large"),
            "cluster": st.column_config.NumberColumn("Кластер", format="%d"),
        },
    )
    actions = st.container(horizontal=True, vertical_alignment="bottom")
    with actions:
        remove_gids = st.multiselect(
            "Исключить из перечня",
            review_df["gid"].tolist(),
            placeholder="Выберите gid",
        )
        if st.button("Исключить", icon=":material/remove_circle:", disabled=not remove_gids):
            for remove_gid in remove_gids:
                review_cases.pop(int(remove_gid), None)
            st.session_state["review_cases"] = review_cases
            st.rerun()
        st.download_button(
            "Скачать перечень CSV",
            data=review_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="review_candidates.csv",
            mime="text/csv",
            icon=":material/download:",
        )

render_assistant(
    ROOT, GraphTools(nodes, edges, transactions, repeats, episodes), selected_gid,
    (str(Path(out_dir).resolve()), (Path(out_dir) / "run_manifest.json").stat().st_mtime_ns),
    ROLE_LABELS, answer_analyst_query, analyst_rationale,
)

st.subheader("Анализ участника")
view_mode = st.segmented_control(
    "Представление",
    ["network", "trace", "common", "transactions", "patterns", "resilience", "links", "clusters"],
    default="network",
    format_func=lambda value: {
        "network": "Сеть",
        "trace": "След денег",
        "common": "Общие получатели",
        "transactions": "Операции",
        "patterns": "Маршруты",
        "resilience": "Устойчивость",
        "links": "Связи",
        "clusters": "Кластеры",
    }[value],
    label_visibility="collapsed",
)

neighborhood = edges[(edges["src"] == selected_gid) | (edges["dst"] == selected_gid)].copy()

if view_mode == "network":
    with st.container(border=True):
        graph_controls = st.container(horizontal=True, vertical_alignment="bottom")
        with graph_controls:
            graph_radius = st.segmented_control(
                "Окружение",
                [1, 2],
                default=2,
                format_func=lambda value: f"{value} колено" if value == 1 else f"{value} колена",
                help="Второе колено раскрывает контрагентов соседних узлов.",
            )
            graph_edge_limit = st.select_slider(
                "Максимум связей",
                options=[12, 20, 40, 60],
                value=40,
                help="Показываются крупнейшие связи на каждом шаге от выбранного узла.",
            )
        st.caption(
            "Цвет — роль · размер — приоритет · светлый контур — выбранный участник. "
            "Стрелки показывают направление денег. Нажмите на узел для выделения связей; "
            "колесо — масштаб, перетаскивание — перемещение."
        )
        legend = st.container(horizontal=True, vertical_alignment="center")
        with legend:
            st.html("<div style='display:flex;flex-wrap:wrap;gap:16px;background:#19191F;"
                    "color:#D2CEDD;padding:12px 16px;border-radius:10px;font-size:12px'>"
                    + "".join(f"<span><span style='color:{color}'>●</span> {ROLE_LABELS[role]}</span>"
                              for role, color in GRAPH_COLORS.items()) + "</div>")
        graph_glow = st.toggle("Свечение связей с сигналами", value=True)
        st.caption("Связи: янтарный — операции в окне 0–2 дня; коралловый — также известен "
                   "порядок по дням и сопоставимы суммы. Наведите на линию для дат и сумм. "
                   "Проверяются только маршруты в показанном окружении; серый не означает безопасный.")
        if neighborhood.empty:
            st.info("У участника нет наблюдаемых связей в выгрузке.")
        else:
            render_neighborhood(
                nodes,
                edges,
                selected_gid,
                radius=int(graph_radius or 1),
                edge_limit=int(graph_edge_limit),
                transactions=transactions,
                glow=graph_glow,
            )

elif view_mode == "common":
    show_common_recipients(nodes, edges, ROLE_LABELS)

elif view_mode == "trace":
    trace_path = trace_from_seed(nodes, edges, selected_gid)
    if not trace_path:
        st.info("Путь от известного seed до этого участника не найден.")
    elif len(trace_path) == 1:
        st.info("Выбранный участник сам является исходным seed.")
    else:
        st.caption(
            f"Кратчайший наблюдаемый путь от seed: {len(trace_path) - 1} переводов. "
            "При равной длине выбран путь с более крупными связями."
        )
        st.markdown(" → ".join(f"`{gid}`" for gid in trace_path))

        path_nodes = nodes.set_index("gid").loc[trace_path].reset_index()
        path_nodes["gid"] = path_nodes["gid"].astype(str)
        path_nodes["role_display"] = path_nodes["role"].map(ROLE_LABELS).fillna(path_nodes["role"])
        path_nodes["rationale"] = path_nodes.apply(analyst_rationale, axis=1)
        path_nodes["step"] = range(len(path_nodes))
        st.markdown("**Узлы цепочки**")
        st.dataframe(
            path_nodes[["step", "gid", "depth", "role_display", "priority_score", "rationale"]],
            width="stretch",
            hide_index=True,
            column_config={
                "step": st.column_config.NumberColumn("Шаг", format="%d", width="small"),
                "gid": st.column_config.TextColumn("gid", pinned=True),
                "depth": st.column_config.NumberColumn("Колено", format="%d", width="small"),
                "role_display": st.column_config.TextColumn("Роль"),
                "priority_score": st.column_config.ProgressColumn(
                    "Приоритет", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "rationale": st.column_config.TextColumn("Обоснование", width="large"),
            },
        )

        path_edges = []
        for step, (source, target) in enumerate(zip(trace_path, trace_path[1:]), start=1):
            edge = edges[(edges["src"] == source) & (edges["dst"] == target)].iloc[0]
            path_edges.append(
                {
                    "step": step,
                    "src": str(source),
                    "dst": str(target),
                    "sum_kzt": float(edge["sum_kzt"]),
                    "n_tx": int(edge["n_tx"]),
                }
            )
        st.markdown("**Переводы по пути**")
        st.dataframe(
            pd.DataFrame(path_edges),
            width="stretch",
            hide_index=True,
            column_config={
                "step": st.column_config.NumberColumn("Шаг", format="%d", width="small"),
                "src": st.column_config.TextColumn("Отправитель"),
                "dst": st.column_config.TextColumn("Получатель"),
                "sum_kzt": st.column_config.NumberColumn("Сумма", format="localized"),
                "n_tx": st.column_config.NumberColumn("Операций", format="%d"),
            },
        )

elif view_mode == "transactions":
    history = transactions[
        (transactions["src"] == selected_gid) | (transactions["dst"] == selected_gid)
    ].copy()
    if history.empty:
        st.info("История операций недоступна или для участника нет транзакций.")
    else:
        history["direction"] = history["src"].apply(
            lambda source: "Исходящий" if int(source) == selected_gid else "Входящий"
        )
        history["counterparty"] = history.apply(
            lambda row: str(int(row["dst"]))
            if int(row["src"]) == selected_gid
            else str(int(row["src"])),
            axis=1,
        )
        incoming_history = history[history["direction"] == "Входящий"]
        outgoing_history = history[history["direction"] == "Исходящий"]
        history_metrics = st.container(horizontal=True, horizontal_alignment="distribute")
        with history_metrics:
            st.metric(
                "Входящих",
                f"{len(incoming_history)} / {compact_number(incoming_history['sum_kzt'].sum(), '₸')}",
            )
            st.metric(
                "Исходящих",
                f"{len(outgoing_history)} / {compact_number(outgoing_history['sum_kzt'].sum(), '₸')}",
            )
            st.metric("Контрагентов", history["counterparty"].nunique())

        incoming_dates = set(incoming_history["date"].dt.date)
        outgoing_dates = set(outgoing_history["date"].dt.date)
        same_day_dates = sorted(incoming_dates & outgoing_dates)
        if same_day_dates:
            st.warning(
                f"В {len(same_day_dates)} днях зафиксированы входящие и исходящие операции "
                "в один день. Проверьте возможный сквозной транзит."
            )

        history = history.sort_values(["date", "sum_kzt"], ascending=[False, False])
        st.dataframe(
            history[["date", "direction", "counterparty", "sum_kzt"]],
            width="stretch",
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Дата", format="DD.MM.YYYY"),
                "direction": st.column_config.TextColumn("Направление"),
                "counterparty": st.column_config.TextColumn("Контрагент", pinned=True),
                "sum_kzt": st.column_config.NumberColumn("Сумма", format="localized"),
            },
        )

elif view_mode == "links":
    if neighborhood.empty:
        st.info("У участника нет наблюдаемых связей в выгрузке.")
    else:
        display_edges = neighborhood.sort_values("sum_kzt", ascending=False).copy()
        display_edges["src"] = display_edges["src"].astype(str)
        display_edges["dst"] = display_edges["dst"].astype(str)
        st.dataframe(
            display_edges[["src", "dst", "sum_kzt", "n_tx"]],
            width="stretch",
            hide_index=True,
            column_config={
                "src": st.column_config.TextColumn("Отправитель", pinned=True),
                "dst": st.column_config.TextColumn("Получатель", pinned=True),
                "sum_kzt": st.column_config.NumberColumn("Сумма", format="localized", width="medium"),
                "n_tx": st.column_config.NumberColumn("Операций", format="%d", width="small"),
            },
        )

elif view_mode == "patterns":
    show_repeated_routes(repeats, episodes, selected_gid)
    st.markdown("**Крупнейшие структурные маршруты**")
    if route_patterns.empty:
        st.info("Перезапустите пайплайн, чтобы построить устойчивые маршруты A→B→C.")
    else:
        patterns_view = route_patterns.copy()
        only_selected = st.checkbox("Только маршруты выбранного участника", value=False)
        if only_selected:
            patterns_view = patterns_view[
                patterns_view[["src", "via", "dst"]].eq(selected_gid).any(axis=1)
            ]
        if "temporal_status" in patterns_view:
            dated_only = st.checkbox("Только пары с известным порядком по дням", value=False)
            if dated_only:
                patterns_view = patterns_view[
                    patterns_view["temporal_status"] == "Порядок по дням соблюдён"
                ]
        for column in ("src", "via", "dst"):
            patterns_view[column] = patterns_view[column].astype(str)
        patterns_view["via_role"] = patterns_view["via_role"].map(ROLE_LABELS).fillna(
            patterns_view["via_role"]
        )
        st.caption(
            "Двухшаговые маршруты отсортированы по минимальной сумме на двух связях. "
            "Это кандидаты для проверки, а не доказательство движения одной суммы."
        )
        st.caption(
            "Проверяются до 200 крупнейших структурных маршрутов. Для каждой пары "
            "ищется конкретный вход и выход в окне 0–2 дня. Внутри одного дня порядок "
            "неизвестен. Сопоставимые суммы: выход составляет 80–120% входа; "
            "это настраиваемая в коде эвристика, а не нормативный порог."
        )
        st.dataframe(
            patterns_view,
            width="stretch",
            hide_index=True,
            column_config={
                "src": st.column_config.TextColumn("Источник", pinned=True),
                "via": st.column_config.TextColumn("Через узел"),
                "dst": st.column_config.TextColumn("Получатель"),
                "bottleneck_kzt": st.column_config.NumberColumn("Минимум по пути", format="localized"),
                "incoming_kzt": st.column_config.NumberColumn("Вход", format="localized"),
                "outgoing_kzt": st.column_config.NumberColumn("Выход", format="localized"),
                "via_role": st.column_config.TextColumn("Роль посредника"),
                "via_priority": st.column_config.ProgressColumn(
                    "Приоритет", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "rapid_signal": st.column_config.CheckboxColumn("Быстрый транзит"),
                "cycle_signal": st.column_config.CheckboxColumn("Цикл"),
                "temporal_status": st.column_config.TextColumn("Проверка дат"),
                "delay_days": st.column_config.NumberColumn("Разрыв, дней"),
                "incoming_date": st.column_config.TextColumn("Дата входа"),
                "outgoing_date": st.column_config.TextColumn("Дата выхода"),
                "example_in_kzt": st.column_config.NumberColumn("Пример входа, ₸"),
                "example_out_kzt": st.column_config.NumberColumn("Пример выхода, ₸"),
                "amount_ratio": st.column_config.NumberColumn("Выход / вход", format="%.2f"),
                "comparable_amounts": st.column_config.CheckboxColumn("Суммы сопоставимы"),
            },
        )

elif view_mode == "resilience":
    if resilience.empty:
        st.info("Перезапустите пайплайн, чтобы рассчитать устойчивость сети.")
    else:
        st.caption(
            "Стресс-тест показывает, как меняется связность сети после удаления узлов "
            "с максимальным приоритетом."
        )
        chart_data = resilience.set_index("n_removed")[["largest_component_share", "fragmentation"]]
        st.line_chart(chart_data, x_label="Удалено узлов", y_label="Доля")
        st.dataframe(
            resilience,
            width="stretch",
            hide_index=True,
            column_config={
                "n_removed": st.column_config.NumberColumn("Удалено", format="%d"),
                "removed_gids": st.column_config.TextColumn("Удалённые gid", width="large"),
                "remaining_nodes": st.column_config.NumberColumn("Осталось узлов", format="%d"),
                "n_components": st.column_config.NumberColumn("Компоненты", format="%d"),
                "largest_component": st.column_config.NumberColumn("Крупнейшая компонента", format="%d"),
                "largest_component_share": st.column_config.NumberColumn("Доля крупнейшей", format="%.2f"),
                "fragmentation": st.column_config.ProgressColumn(
                    "Фрагментация", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

else:
    cluster_view = clusters.sort_values(["n_seed", "sum_kzt_internal"], ascending=False).copy()
    cluster_view["top_gids"] = cluster_view["top_gids"].fillna("").astype(str)
    st.dataframe(
        cluster_view[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids"]],
        width="stretch",
        hide_index=True,
        column_config={
            "cluster_id": st.column_config.NumberColumn("Кластер", format="%d", pinned=True),
            "n_nodes": st.column_config.NumberColumn("Участники", format="%d"),
            "n_seed": st.column_config.NumberColumn("Seed", format="%d"),
            "sum_kzt_internal": st.column_config.NumberColumn("Внутренний оборот", format="localized"),
            "top_gids": st.column_config.TextColumn("Ключевые gid", width="large"),
        },
    )

with st.expander("Как читать результаты", icon=":material/info:"):
    st.markdown(
        """
        **Роль** описывает основной паттерн движения денег. **Сила роли** показывает,
        насколько признаки узла соответствуют этому паттерну. **Приоритет** задаёт очередь
        ручной проверки с учётом роли, положения в сети и близости к seed-узлам.

        Граф показывает наблюдаемый фрагмент сети. Аналитик проверяет каждую гипотезу
        по первичным данным.
        """
    )
