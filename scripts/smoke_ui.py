"""Exercise all UI views and the generated demo cases after python run.py --no-ui."""
from __future__ import annotations

import json
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
out = Path(os.getenv("MONEY_GRAPH_OUT", "out"))
cases = json.loads((out / "demo_cases.json").read_text(encoding="utf-8"))
app = AppTest.from_file(str(ROOT / "app.py"))


def check(label):
    app.run(timeout=60)
    if app.exception:
        raise RuntimeError(f"{label}: {app.exception[0].value}")
    print(f"{label}: OK", flush=True)


check("initial")
for mode in ("trace", "common", "transactions", "patterns", "resilience", "links", "clusters", "network"):
    app.segmented_control[0].set_value(mode)
    check(mode)

collector = cases.get("collector")
if collector:
    search = next(x for x in app.text_input if x.label == "Найти участника")
    search.set_value(collector["gid"])
    check("collector search")
    comparisons = [d.value for d in app.dataframe if "Фактор" in d.value.columns]
    assert len(comparisons) == 1
    assert abs(comparisons[0]["Вклад"].sum() - collector["priority"]) < 0.000001
    app.segmented_control[0].set_value("common")
    check("common empty selection")
    app.multiselect(key="common_seeds").set_value(collector["seeds"])
    check("common selected seeds")
    assert collector["gid"] in app.selectbox(key="common_target").options
    app.selectbox(key="common_target").set_value(collector["gid"])
    check("common evidence paths")
    assert len(app.code) >= len(collector["seeds"])

repeated = cases.get("repeated_route")
if repeated:
    app.segmented_control[0].set_value("patterns")
    check("repeated routes")
    app.selectbox(key="repeat_route").set_value(tuple(int(repeated[k]) for k in ("src", "via", "dst")))
    check("repeated dated episodes")
    evidence = [d.value for d in app.dataframe if "episode" in d.value.columns and "incoming_date" in d.value.columns]
    assert len(evidence) == 1 and len(evidence[0]) == repeated["episode_count"]

boundary = cases.get("boundary")
if boundary:
    next(x for x in app.text_input if x.label == "Найти участника").set_value(boundary["gid"])
    check("boundary search")
    assert any("глубине 4" in x.value for x in app.warning)
    next(b for b in app.button if b.label == "Добавить в перечень").click()
    check("review decision")
    assert int(boundary["gid"]) in app.session_state["review_cases"]

print("All available demo scenarios passed.")
