from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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
    "coordinator": "#7C3AED",
    "consolidator": "#2563EB",
    "distributor": "#EA580C",
    "transit": "#0891B2",
    "terminal": "#16A34A",
    "peripheral": "#64748B",
}


@st.cache_data(max_entries=4)
def load_results(out_dir: str) -> tuple[pd.DataFrame, ...]:
    root = Path(out_dir)
    return (
        pd.read_csv(root / "node_features.csv"),
        pd.read_csv(root / "edges.csv"),
        pd.read_csv(root / "clusters.csv"),
        pd.read_csv(root / "top_nodes.csv"),
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


def render_neighborhood(nodes: pd.DataFrame, edges: pd.DataFrame, selected_gid: int) -> None:
    from pyvis.network import Network

    neighborhood = edges[(edges["src"] == selected_gid) | (edges["dst"] == selected_gid)].copy()
    neighborhood = neighborhood.sort_values("sum_kzt", ascending=False).head(149)
    visible_gids = {selected_gid} | set(neighborhood["src"].astype(int)) | set(
        neighborhood["dst"].astype(int)
    )
    visible_nodes = nodes[nodes["gid"].isin(visible_gids)]
    is_dark = st.context.theme.type == "dark"

    network = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#0B1120" if is_dark else "#FFFFFF",
        font_color="#E5E7EB" if is_dark else "#111827",
        cdn_resources="in_line",
    )
    for row in visible_nodes.itertuples(index=False):
        is_selected = int(row.gid) == selected_gid
        role_name = ROLE_LABELS.get(row.role, row.role)
        network.add_node(
            str(int(row.gid)),
            label=node_label(int(row.gid), selected_gid),
            title=(
                f"gid: {int(row.gid)}<br>Роль: {role_name}<br>"
                f"Приоритет: {row.priority_score:.2f}<br>{row.evidence}"
            ),
            color=GRAPH_COLORS.get(row.role, "#64748B"),
            size=34 if is_selected else 10 + 18 * float(row.priority_score),
            shape="diamond" if bool(row.is_seed) else "dot",
            borderWidth=5 if is_selected else 1,
        )
    for row in neighborhood.itertuples(index=False):
        network.add_edge(
            str(int(row.src)),
            str(int(row.dst)),
            value=max(1.0, math.log10(float(row.sum_kzt) + 1)),
            title=f"{float(row.sum_kzt):,.0f} ₸ · {int(row.n_tx)} операций",
            arrows="to",
        )
    network.set_options(
        '{"interaction":{"hover":true,"navigationButtons":true},'
        '"physics":{"stabilization":{"iterations":160},"barnesHut":{"springLength":125}},'
        '"edges":{"smooth":{"type":"dynamic"},"color":{"color":"#94A3B8"}},'
        '"nodes":{"font":{"size":12,"face":"Inter, Arial"}}}'
    )
    st.iframe(network.generate_html(), width="stretch", height=540)


def render_role_badge(role: str) -> None:
    st.badge(ROLE_LABELS.get(role, role), color=ROLE_COLORS.get(role, "gray"))


st.set_page_config(
    page_title="Граф денег",
    page_icon=":material/account_tree:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Граф денег")
    st.caption("Аналитическое рабочее место")
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

try:
    nodes, edges, clusters, top = load_results(out_dir)
except FileNotFoundError:
    st.error("Результаты анализа не найдены", icon=":material/folder_off:")
    st.info("Запустите расчётный пайплайн, затем укажите папку результатов в боковой панели.")
    st.code("python -m money_graph.pipeline --data-dir data --out-dir out", language="powershell")
    st.stop()

role_options = sorted(nodes["role"].dropna().unique())
with st.sidebar:
    selected_roles = st.pills(
        "Роли",
        role_options,
        default=role_options,
        format_func=lambda role: ROLE_LABELS.get(role, role),
        selection_mode="multi",
        wrap=True,
    )
    st.divider()
    st.caption("Оценки формируют гипотезы для аналитика и не устанавливают виновность.")

header = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
with header:
    with st.container():
        st.title("Транзакционная сеть")
        st.caption("Роли, денежные потоки и связи участников в одном рабочем пространстве")
    st.badge("Данные готовы", icon=":material/check_circle:", color="green")

metrics = st.container(border=True, horizontal=True, horizontal_alignment="distribute")
with metrics:
    st.metric("Участники", f"{len(nodes):,}".replace(",", " "), border=False)
    st.metric("Связи", f"{len(edges):,}".replace(",", " "), border=False)
    st.metric("Кластеры", f"{len(clusters):,}".replace(",", " "), border=False)
    st.metric("Оборот", compact_number(edges["sum_kzt"].sum(), "₸"), border=False)

active_roles = selected_roles or role_options
priority = top[top["role"].isin(active_roles)].copy()
priority["gid_display"] = priority["gid"].astype(str)
priority["role_display"] = priority["role"].map(ROLE_LABELS).fillna(priority["role"])

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

table_col, card_col = st.columns([1.65, 1], gap="large")
with table_col:
    with st.container(border=True):
        st.caption("Выберите строку, чтобы открыть карточку участника")
        if priority.empty:
            st.info("По выбранным ролям нет участников.")
            table_event = None
        else:
            table_event = st.dataframe(
                priority[["rank", "gid_display", "role_display", "priority_score"]],
                width="stretch",
                height=356,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="priority_table",
                column_config={
                    "rank": st.column_config.NumberColumn("№", width="small", format="%d"),
                    "gid_display": st.column_config.TextColumn("gid", width="medium", pinned=True),
                    "role_display": st.column_config.TextColumn("Роль", width="medium"),
                    "priority_score": st.column_config.ProgressColumn(
                        "Приоритет", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )
            if table_event.selection.rows:
                selected_gid = int(priority.iloc[table_event.selection.rows[0]]["gid"])

st.session_state["selected_gid"] = selected_gid
node = nodes[nodes["gid"] == selected_gid].iloc[0]

with card_col:
    with st.container(border=True, height=393):
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
        st.markdown("**Почему узел выделен**")
        st.write(node["evidence"])
        if bool(node["truncated_by_depth"]):
            st.warning("Граница depth=4: дальнейшие исходящие переводы не наблюдаются.")

st.subheader("Анализ участника")
view_mode = st.segmented_control(
    "Представление",
    ["network", "links", "clusters"],
    default="network",
    format_func=lambda value: {
        "network": "Сеть",
        "links": "Связи",
        "clusters": "Кластеры",
    }[value],
    label_visibility="collapsed",
)

neighborhood = edges[(edges["src"] == selected_gid) | (edges["dst"] == selected_gid)].copy()

if view_mode == "network":
    with st.container(border=True):
        legend = st.container(horizontal=True, vertical_alignment="center")
        with legend:
            st.caption("Цвет — роль · размер — приоритет · ромб — seed")
            for role in ("coordinator", "consolidator", "distributor", "transit", "terminal"):
                render_role_badge(role)
        if neighborhood.empty:
            st.info("У участника нет наблюдаемых связей в выгрузке.")
        else:
            render_neighborhood(nodes, edges, selected_gid)

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
        **Роль** описывает наиболее выраженный паттерн движения денег. **Сила роли** показывает,
        насколько признаки узла соответствуют этому паттерну. **Приоритет** помогает выстроить
        очередь ручной проверки и учитывает роль, сетевую позицию и близость к seed-узлам.

        Граф отражает только наблюдаемый фрагмент сети. Все выводы являются аналитическими
        гипотезами и требуют проверки по первичным данным.
        """
    )
