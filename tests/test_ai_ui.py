from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from money_graph.ai_assistant import AIConfig, AssistantError


APP = '''
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from money_graph.assistant_ui import render_assistant
render_assistant(Path('.'), SimpleNamespace(nodes=pd.DataFrame()), 21, ('test', 1), {}, None, None)
'''


def test_chat_requires_consent_and_does_not_resend_on_rerun():
    reply = {"answer": "Три плательщика [D1].", "trace": [
        {"id": "D1", "tool": "node_profile", "result": {"gid": "21", "in_deg": 3}}],
        "model": "test-model", "usage": {"input_tokens": 10, "output_tokens": 5}}
    with patch("money_graph.assistant_ui.load_config", return_value=AIConfig("test-key")), \
         patch("money_graph.assistant_ui.ask_graph", return_value=reply) as api:
        app = AppTest.from_string(APP).run()
        assert not app.exception
        assert app.chat_input[0].disabled
        assert api.call_count == 0
        app.checkbox(key="ai_data_consent").check().run()
        assert not app.chat_input[0].disabled
        app.chat_input[0].set_value("Проверь узел").run()
        assert not app.exception
        assert api.call_count == 1
        assert len(app.session_state["ai_messages"]) == 2
        app.run()
        assert api.call_count == 1
        app.button(key="clear_ai_chat").click().run()
        assert app.session_state["ai_messages"] == []
        assert api.call_count == 1


def test_api_failure_is_visible_without_retry():
    with patch("money_graph.assistant_ui.load_config", return_value=AIConfig("test-key")), \
         patch("money_graph.assistant_ui.ask_graph", side_effect=AssistantError("OpenAI не принял ключ.")) as api:
        app = AppTest.from_string(APP).run()
        app.checkbox(key="ai_data_consent").check().run()
        app.chat_input[0].set_value("Проверь узел").run()
        assert not app.exception
        assert "не принял ключ" in app.error[0].value
        assert app.session_state["ai_messages"] == []
        app.run()
        assert api.call_count == 1
