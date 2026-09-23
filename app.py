from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


@st.cache_data
def load_results(out_dir: str):
    root = Path(out_dir)
    return (
        pd.read_csv(root / "node_features.csv"),
        pd.read_csv(root / "edges.csv"),
        pd.read_csv(root / "clusters.csv"),
        pd.read_csv(root / "top_nodes.csv"),
    )


def render_neighborhood(nodes: pd.DataFrame, edges: pd.DataFrame, selected_gid: int) -> None:
    from pyvis.network import Network

    palette = {
        "coordinator": "#7C3AED",
        "consolidator": "#2563EB",
        "distributor": "#EA580C",
        "transit": "#0891B2",
        "terminal": "#16A34A",
        "peripheral": "#64748B",
    }
    neighborhood = edges[(edges["src"] == selected_gid) | (edges["dst"] == selected_gid)].copy()
    neighborhood = neighborhood.sort_values("sum_kzt", ascending=False).head(149)
    visible_gids = {selected_gid} | set(neighborhood["src"].astype(int)) | set(
        neighborhood["dst"].astype(int)
    )
    visible_nodes = nodes[nodes["gid"].isin(visible_gids)]

    network = Network(
        height="620px", width="100%", directed=True, bgcolor="#FFFFFF",
        font_color="#0F172A", cdn_resources="in_line",
    )
    for row in visible_nodes.itertuples(index=False):
        is_selected = int(row.gid) == selected_gid
        network.add_node(
            str(int(row.gid)), label=str(int(row.gid)),
            title=(f"gid: {int(row.gid)}<br>роль: {row.role}<br>"
                   f"приоритет: {row.priority_score:.3f}<br>{row.evidence}"),
            color=palette.get(row.role, "#64748B"),
            size=32 if is_selected else 10 + 20 * float(row.priority_score),
            shape="diamond" if bool(row.is_seed) else "dot",
            borderWidth=4 if is_selected else 1,
        )
    for row in neighborhood.itertuples(index=False):
        network.add_edge(
            str(int(row.src)), str(int(row.dst)),
            value=max(1.0, math.log10(float(row.sum_kzt) + 1)),
            title=f"{float(row.sum_kzt):,.0f} KZT; {int(row.n_tx)} операций",
            arrows="to",
        )
    network.set_options(
        '{"interaction":{"hover":true,"navigationButtons":true},'
        '"physics":{"stabilization":{"iterations":180}},'
        '"edges":{"smooth":{"type":"dynamic"},"color":{"color":"#94A3B8"}},'
        '"nodes":{"font":{"size":12,"face":"Arial"}}}'
    )
    components.html(network.generate_html(), height=640, scrolling=False)


st.set_page_config(page_title="Граф денег", layout="wide")
st.title("Граф денег")
st.caption("Гипотезы о роли узлов для ручной проверки. Оценки не устанавливают виновность.")

out_dir = st.sidebar.text_input("Папка результатов", os.getenv("MONEY_GRAPH_OUT", "out"))
try:
    nodes, edges, clusters, top = load_results(out_dir)
except FileNotFoundError:
    st.info("Сначала запустите расчётный пайплайн и укажите папку результатов.")
    st.stop()

role_options = sorted(nodes["role"].unique())
selected_roles = st.sidebar.multiselect("Роли", role_options, default=role_options)
gid_value = st.sidebar.text_input("Поиск по gid")

col1, col2, col3 = st.columns(3)
col1.metric("Узлов", f"{len(nodes):,}".replace(",", " "))
col2.metric("Кластеров", len(clusters))
col3.metric("Связей", f"{len(edges):,}".replace(",", " "))

st.subheader("Приоритет проверки")
view = top[top["role"].isin(selected_roles)]
st.dataframe(view, use_container_width=True, hide_index=True)

selected_gid = None
if gid_value.strip():
    try:
        selected_gid = int(gid_value.strip())
    except ValueError:
        st.error("gid должен быть целым числом")
    if selected_gid is not None and selected_gid not in set(nodes["gid"]):
        st.warning("gid не найден в текущем запуске")
        selected_gid = None
elif not top.empty:
    selected_gid = int(top.iloc[0]["gid"])

if selected_gid is not None:
    row = nodes[nodes["gid"] == selected_gid].iloc[0]
    st.subheader(f"Карточка узла {selected_gid}")
    a, b, c, d = st.columns(4)
    a.metric("Роль", row["role"])
    b.metric("Сила признаков", f"{row['role_score']:.2f}")
    c.metric("Приоритет", f"{row['priority_score']:.2f}")
    d.metric("Кластер", int(row["cluster_id"]))
    st.write(row["evidence"])
    if bool(row["truncated_by_depth"]):
        st.warning("Узел находится на границе depth=4: дальнейшие исходящие переводы неизвестны.")

    neighborhood = edges[(edges["src"] == selected_gid) | (edges["dst"] == selected_gid)].copy()
    if neighborhood.empty:
        st.info("У узла нет наблюдаемых связей в выгрузке.")
    else:
        st.subheader("Направленное окружение")
        st.caption(
            "Стрелка показывает направление перевода. Размер узла отражает приоритет; "
            "ромб обозначает seed. При большом окружении показаны 149 крупнейших связей."
        )
        render_neighborhood(nodes, edges, selected_gid)
        st.subheader("Наблюдаемые связи")
        st.dataframe(
            neighborhood.sort_values("sum_kzt", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
