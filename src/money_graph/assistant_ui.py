"""Native Streamlit UI; API calls happen only on explicit chat submission."""
from __future__ import annotations

import streamlit as st

from .ai_assistant import AssistantError, ask_graph, load_config
from .assistant_tools import TOOL_LABELS


def show_answer(message):
    st.markdown(message["content"])
    if message.get("trace"):
        with st.expander("Данные, на которых основан ответ", icon=":material/fact_check:"):
            for item in message["trace"]:
                st.markdown(f"**[{item['id']}] {TOOL_LABELS.get(item['tool'], 'Проверка')}**")
                st.json(item["result"], expanded=False)
        st.caption(f"Модель: {message['model']} · токенов в запросах: {message['usage']['input_tokens']} · "
                   f"в ответах: {message['usage']['output_tokens']}. Числа и ссылки нужно сверять с данными.")


def render_assistant(root, graph, selected_gid, dataset_version, role_labels, local_query, rationale):
    with st.expander("Ассистент аналитика", icon=":material/psychology:"):
        config, config_error = None, None
        try:
            config = load_config(root)
        except AssistantError as exc:
            config_error = str(exc)
        mode = st.radio("Режим помощника", ["AI · OpenAI", "Локальные правила"], horizontal=True,
                        index=0 if config and config.api_key else 1, key="assistant_mode")
        if mode == "Локальные правила":
            st.caption("Без интернета и API. Понимает ключевые слова: сбор средств, транзит, координатор, аномалии.")
            question = st.text_input("Вопрос по графу", placeholder="Кто собирает деньги?", key="assistant_query")
            if question.strip():
                answer, candidates = local_query(question, graph.nodes)
                st.write(answer)
                if not candidates.empty:
                    view = candidates.copy()
                    view["gid"] = view.gid.astype(str)
                    view["role"] = view.role.map(role_labels)
                    view["rationale"] = candidates.apply(rationale, axis=1)
                    columns = ["gid", "role", "priority_score"]
                    if "anomaly_score" in view:
                        columns.append("anomaly_score")
                    st.dataframe(view[columns + ["rationale"]], hide_index=True, column_config={
                        "gid": st.column_config.TextColumn("gid", pinned=True), "role": "Роль",
                        "priority_score": st.column_config.ProgressColumn("Приоритет", min_value=0.0, max_value=1.0, format="%.2f"),
                        "anomaly_score": st.column_config.ProgressColumn("Аномальность", min_value=0.0, max_value=1.0, format="%.2f"),
                        "rationale": st.column_config.TextColumn("Обоснование", width="large"),
                    })
            return
        st.caption("По отправке вопроса OpenAI получает текст диалога, выбранный gid и ограниченные "
                   "результаты проверок: идентификаторы, суммы, даты и пути. Полные файлы не загружаются. "
                   "Запросы используют API-кредиты; переключение экранов новых запросов не отправляет.")
        if config_error:
            st.warning(config_error)
            return
        if not config or not config.api_key:
            st.info("Для AI добавьте OPENAI_API_KEY в локальный .streamlit/secrets.toml. Локальные правила уже доступны.")
            return
        st.caption(f"Выбранный участник: {selected_gid} · модель {config.model}. "
                   "Ключ задан; доступ к API проверяется при отправке вопроса.")
        if st.session_state.get("ai_dataset_version") != dataset_version:
            st.session_state["ai_messages"] = []
            st.session_state["ai_dataset_version"] = dataset_version
        messages = st.session_state.setdefault("ai_messages", [])
        if st.button("Очистить диалог", icon=":material/chat_bubble_outline:", key="clear_ai_chat"):
            messages.clear()
        if not messages:
            st.info("Например: «Почему этот участник в топе и что проверить дальше?»; "
                    "«Найди сборщиков с пятью плательщиками, которые отдают не более 10% входа»; "
                    "«Кто получает деньги от seed … и …?»")
        for message in messages:
            with st.chat_message(message["role"]):
                show_answer(message)
        consent = st.checkbox("Разрешаю отправку вопросов и результатов анализа в OpenAI", key="ai_data_consent",
                              help="Включает выбранный gid и ограниченные результаты инструментов: идентификаторы, "
                                   "суммы, даты и пути. Разрешение действует до закрытия сессии или снятия отметки.")
        question = st.chat_input("Задайте вопрос о графе", max_chars=1500, key="ai_prompt", submit_mode="disable",
                                 disabled=not consent)
        if question and consent:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Проверяю граф и готовлю объяснение…"):
                        result = ask_graph(question, graph, config, selected_gid, messages)
                    answer = {"role": "assistant", "content": result["answer"],
                              "trace": result["trace"], "model": result["model"], "usage": result["usage"]}
                    show_answer(answer)
                    messages.extend([{"role": "user", "content": question}, answer])
                    st.session_state["ai_messages"] = messages[-12:]
                except AssistantError as exc:
                    st.error(str(exc))
                    st.caption("Запрос не будет повторён автоматически. Можно продолжить в режиме «Локальные правила».")
