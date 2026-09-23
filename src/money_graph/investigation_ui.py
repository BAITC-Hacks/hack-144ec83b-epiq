"""Native Streamlit views for evidence review; calculations live in core modules."""
import pandas as pd
import streamlit as st

from .investigations import common_recipients


@st.cache_data(max_entries=16)
def cached_common(nodes, edges, seeds, max_hops):
    return common_recipients(nodes, edges, seeds, max_hops)


def show_common_recipients(nodes, edges, role_labels):
    seeds = nodes.loc[nodes["is_seed"].astype(bool), "gid"].astype(str).sort_values().tolist()
    selected = st.multiselect("Выберите от 2 до 5 исходных клиентов", seeds,
                              max_selections=5, key="common_seeds")
    depth = st.slider("Максимум переходов от каждого seed", 1, 4, 4)
    st.caption("Ищем участников, достижимых по направлению переводов из КАЖДОГО выбранного seed. "
               "Это общие получатели по структуре; пути не доказывают общий источник денег.")
    if len(selected) < 2:
        st.info("Выберите минимум двух seed для сравнения их цепочек.")
        return
    candidates, paths = cached_common(nodes, edges, [int(s) for s in selected], depth)
    if candidates.empty:
        st.info("Общих получателей в выбранной глубине нет. Это тоже результат проверки.")
        return
    view = candidates[["gid", "role", "priority_score", "selected_seed_count", "max_hops"]].copy()
    view["gid"] = view["gid"].astype(str)
    view["role"] = view["role"].map(role_labels)
    st.write(f"Общих получателей: {len(view)}")
    st.dataframe(view, hide_index=True, column_config={
        "gid": "Получатель", "role": "Роль", "priority_score": "Приоритет",
        "selected_seed_count": "Достижим из seed", "max_hops": "Самый длинный из путей",
    })
    target = st.selectbox("Получатель для разбора путей", view["gid"].tolist(), key="common_target")
    evidence = []
    edge_info = {(int(r.src), int(r.dst)): r for r in edges.itertuples(index=False)}
    for seed, path in paths[int(target)].items():
        st.markdown(f"**Seed {seed}** · {len(path) - 1} переходов")
        st.code(" → ".join(str(gid) for gid in path), language=None)
        for step, (source, destination) in enumerate(zip(path, path[1:]), 1):
            edge = edge_info[(source, destination)]
            evidence.append({"seed": str(seed), "step": step, "src": str(source),
                             "dst": str(destination), "sum_kzt": float(edge.sum_kzt),
                             "n_tx": int(edge.n_tx)})
    details = pd.DataFrame(evidence)
    st.dataframe(details, hide_index=True, column_config={
        "step": "Шаг", "src": "Отправитель", "dst": "Получатель",
        "sum_kzt": "Оборот связи, ₸", "n_tx": "Операций",
    })
    st.caption("Показаны кратчайшие пути до четырёх переходов. Общие участки могут повторяться "
               "в нескольких путях; суммы строк нельзя складывать как независимые потоки.")
    st.download_button("Скачать доказательные пути CSV", details.to_csv(index=False).encode("utf-8-sig"),
                       "common_recipient_paths.csv", "text/csv")


def show_repeated_routes(repeats, episodes, selected_gid):
    st.markdown("**Повторяющиеся цепочки с датированными примерами**")
    st.caption("Один маршрут A→B→C, минимум два непересекающихся окна. "
               "Каждый выход — через 1–2 дня после входа, выход/вход 0,8–1,2. "
               "Операции одного дня и совпадающие строки не увеличивают число повторений.")
    if repeats.empty:
        st.info("Повторяющихся цепочек нет или расчёт ещё не обновлён. Выполните python run.py --no-ui.")
        return
    filtered = repeats.copy()
    if st.checkbox("Только повторяющиеся цепочки выбранного участника", key="repeats_selected"):
        filtered = filtered[filtered[["src", "via", "dst"]].eq(selected_gid).any(axis=1)]
    if filtered.empty:
        st.info("У этого участника повторяющихся цепочек по заданному правилу не найдено.")
        return
    view = filtered.copy()
    for field in ("src", "via", "dst"):
        view[field] = view[field].astype(str)
    st.dataframe(view, hide_index=True, column_config={
        "src": "Источник", "via": "Посредник", "dst": "Получатель",
        "episode_count": "Повторений", "first_in": "Первый вход", "last_out": "Последний выход",
    })
    route_options = [tuple(int(v) for v in row) for row in
                     filtered[["src", "via", "dst"]].itertuples(index=False, name=None)]
    choice = st.selectbox("Цепочка для проверки эпизодов", route_options,
                          format_func=lambda route: " → ".join(str(g) for g in route), key="repeat_route")
    details = episodes[(episodes["src"] == choice[0]) & (episodes["via"] == choice[1])
                       & (episodes["dst"] == choice[2])].copy()
    for field in ("src", "via", "dst"):
        details[field] = details[field].astype(str)
    st.dataframe(details, hide_index=True, column_config={
        "src": "Источник", "via": "Посредник", "dst": "Получатель", "episode": "Эпизод",
        "incoming_date": "Дата входа", "outgoing_date": "Дата выхода",
        "incoming_kzt": "Вход, ₸", "outgoing_kzt": "Выход, ₸", "delay_days": "Дней",
        "amount_ratio": "Выход / вход",
    })
    st.caption("Это повторение наблюдаемой структуры и временных признаков, не доказательство "
               "прохождения тех же денег. Одна операция может поддерживать разные альтернативные маршруты.")
    st.download_button("Скачать эпизоды CSV", details.to_csv(index=False).encode("utf-8-sig"),
                       "route_episodes_selected.csv", "text/csv")
    st.divider()
