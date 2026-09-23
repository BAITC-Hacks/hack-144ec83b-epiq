"""Read-only, bounded graph tools. No model-generated Python, SQL or file access."""
from __future__ import annotations

import json
from dataclasses import dataclass

import jsonschema
import pandas as pd

from .investigations import common_recipients
from .scoring import PRIORITY_LABELS, priority_components


GID = {"type": "string", "pattern": "^[0-9]{1,20}$"}
NULL_GID = {**GID, "type": ["string", "null"]}
DATE = {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 10}
ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


def function(name, description, properties):
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


TOOLS = [
    function("rank_nodes", "Отбор кандидатов: аккумулятор=consolidator. Все фильтры применяются вместе. "
             "null означает отсутствие фильтра; min_seed_count/min_payers=0 без ограничения. "
             "max_out_ratio — максимум наблюдаемого отношения выхода ко входу, например 0.1 для 10%.", {
        "role": {"type": ["string", "null"], "enum": [None, *ROLES]},
        "min_seed_count": {"type": "integer", "minimum": 0, "maximum": 1000000},
        "min_payers": {"type": "integer", "minimum": 0, "maximum": 1000000},
        "max_out_ratio": {"type": ["number", "null"], "minimum": 0, "maximum": 1000000},
        "depth": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
        "sort_by": {"type": "string", "enum": ["priority_score", "anomaly_score"]}, "limit": LIMIT}),
    function("node_profile", "Карточка конкретного gid: роль, приоритет, числовые факты, пробелы и следующий запрос.", {"gid": GID}),
    function("compare_nodes", "Сравнить точные вклады пяти факторов приоритета у 2–5 участников.", {
        "gids": {"type": "array", "items": GID, "minItems": 2, "maxItems": 5}}),
    function("find_common_recipients", "Общие получатели, достижимые из КАЖДОГО из 2–5 seed. "
             "Возвращает структурные пути, без утверждения о хронологии или одних и тех же деньгах.", {
        "seeds": {"type": "array", "items": GID, "minItems": 2, "maxItems": 5},
        "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}, "limit": LIMIT}),
    function("transaction_history", "Операции участника: даты, суммы, контрагенты. Последние строки выбранного "
             "периода; итоги рассчитаны по всему фильтру. since/until включительно; null — весь период.", {
        "gid": GID, "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"]},
        "since": DATE, "until": DATE, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}),
    function("find_repeated_routes", "Повторяющиеся цепочки A→B→C и датированные эпизоды. "
             "gid=null — вся сеть; иначе любые цепочки с участием указанного gid.", {"gid": NULL_GID, "limit": LIMIT}),
    function("dataset_summary", "Общие сведения о выгрузке, ограничениях и возможностях помощника. "
             "Для вопросов без конкретного анализа клиентов; не передаёт отдельные записи.", {}),
]
SCHEMAS = {t["name"]: t["parameters"] for t in TOOLS}
TOOL_LABELS = dict(zip(SCHEMAS, ["Отбор кандидатов", "Карточка участника", "Сравнение приоритета",
                               "Общие получатели", "История операций", "Повторяющиеся маршруты", "Сведения о данных"]))
NODE_FIELDS = ["gid", "role", "priority_score", "role_score", "depth", "is_seed", "in_deg", "out_deg",
               "in_kzt", "out_kzt", "pass_through", "seed_reach", "anomaly_score", "evidence",
               "data_gap", "next_request"]


def records(frame):
    """Preserve 18-digit gids as strings and serialize NaN/dates safely."""
    frame = frame.copy()
    for field in ("gid", "src", "via", "dst"):
        if field in frame:
            frame[field] = frame[field].astype(str)
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=10))


@dataclass
class GraphTools:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    transactions: pd.DataFrame
    repeats: pd.DataFrame
    episodes: pd.DataFrame

    def node(self, gid):
        node = self.nodes[self.nodes.gid == int(gid)]
        if node.empty:
            raise ValueError("Указанный gid отсутствует в текущей выгрузке")
        return node

    def brief(self, frame):
        return records(frame[[c for c in NODE_FIELDS if c in frame]])

    def execute(self, name, args):
        if name not in SCHEMAS:
            return {"error": "Неизвестный инструмент. Допустимы только разрешённые операции чтения."}
        try:
            json.dumps(args, allow_nan=False)
            jsonschema.validate(args, SCHEMAS[name])
        except (ValueError, TypeError, jsonschema.ValidationError):
            return {"error": "Аргументы не соответствуют схеме инструмента. Исправьте типы и диапазоны."}
        try:
            return self._execute(name, args)
        except (ValueError, OverflowError):
            return {"error": "Проверьте gid, уникальность seed и корректность дат. Глубина: 1–4, seed: 2–5."}

    def _execute(self, name, a):
        if name == "dataset_summary":
            tx = self.transactions
            return {"nodes": len(self.nodes), "edges": len(self.edges), "transactions": len(tx),
                    "seeds": int(self.nodes.is_seed.sum()),
                    "period_from": str(tx.date.min())[:10] if not tx.empty else None,
                    "period_to": str(tx.date.max())[:10] if not tx.empty else None,
                    "capabilities": list(TOOL_LABELS.values()),
                    "limitations": "Нет размеченных ролей и персональных данных; глубина 4 обрывается; "
                                   "вход seed неполон; пути структурные; порядок внутри дня неизвестен. "
                                   "Решения и запросы делает аналитик, помощник только читает данные."}
        if name == "rank_nodes":
            frame = self.nodes[(self.nodes.seed_reach >= a["min_seed_count"]) &
                               (self.nodes.in_deg >= a["min_payers"])]
            for field in ("role", "depth"):
                if a[field] is not None:
                    frame = frame[frame[field] == a[field]]
            if a["max_out_ratio"] is not None:
                # This ratio is not meaningful for seed with incomplete inflow or the depth-4 boundary.
                frame = frame[~frame.is_seed & (frame.depth < 4) & (frame.pass_through <= a["max_out_ratio"])]
            ordered = frame.sort_values([a["sort_by"], "gid"], ascending=[False, True])
            return {"matched": len(frame), "shown": min(len(frame), a["limit"]), "filters": a,
                    "nodes": self.brief(ordered.head(a["limit"]))}
        if name == "node_profile":
            node = self.node(a["gid"])
            ranked = self.nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).gid.tolist()
            return {"node": self.brief(node)[0], "rank": ranked.index(int(a["gid"])) + 1,
                    "total_nodes": len(ranked), "priority_parts": records(priority_components(node))[0],
                    "factors": {k: v[0] for k, v in PRIORITY_LABELS.items()}}
        if name == "compare_nodes":
            gids = list(dict.fromkeys(a["gids"]))
            if len(gids) < 2:
                raise ValueError("Нужны разные gid")
            frame = pd.concat([self.node(gid) for gid in gids])
            parts = priority_components(frame)
            view = frame[["gid", "role", "priority_score"]].copy()
            view[list(parts.columns)] = parts
            return {"nodes": records(view), "factors": {k: v[0] for k, v in PRIORITY_LABELS.items()}}
        if name == "find_common_recipients":
            candidates, paths = common_recipients(self.nodes, self.edges, a["seeds"], a["max_hops"])
            results = []
            edge_lookup = {(int(r.src), int(r.dst)): r for r in self.edges.itertuples(index=False)}
            for node in self.brief(candidates.head(a["limit"])):
                evidence = []
                for seed, route in paths[int(node["gid"])].items():
                    links = [{"src": str(src), "dst": str(dst), "sum_kzt": float(edge_lookup[src, dst].sum_kzt),
                              "n_tx": int(edge_lookup[src, dst].n_tx)} for src, dst in zip(route, route[1:])]
                    evidence.append({"seed": str(seed), "path": [str(g) for g in route], "edges": links})
                results.append({"node": node, "paths": evidence})
            return {"matched": len(candidates), "shown": len(results), "candidates": results,
                    "limitation": "Пути структурные, общие участки повторяются; суммы путей нельзя складывать."}
        if name == "transaction_history":
            self.node(a["gid"])
            gid = int(a["gid"])
            tx = self.transactions.copy()
            tx["date"] = pd.to_datetime(tx.date).dt.normalize()
            if a["since"] and a["until"] and pd.Timestamp(a["since"]) > pd.Timestamp(a["until"]):
                raise ValueError("Перепутаны даты")
            mask = ((tx.dst == gid) if a["direction"] == "incoming" else
                    (tx.src == gid) if a["direction"] == "outgoing" else ((tx.src == gid) | (tx.dst == gid)))
            tx = tx[mask]
            if a["since"]:
                tx = tx[tx.date >= pd.Timestamp(a["since"])]
            if a["until"]:
                tx = tx[tx.date <= pd.Timestamp(a["until"])]
            return {"matched": len(tx), "incoming_kzt": float(tx.loc[tx.dst == gid, "sum_kzt"].sum()),
                    "outgoing_kzt": float(tx.loc[tx.src == gid, "sum_kzt"].sum()),
                    "transactions": records(tx.sort_values(["date", "src", "dst"]).tail(a["limit"])),
                    "limitation": "Последние операции по фильтру. Порядок внутри дня неизвестен."}
        if name == "find_repeated_routes":
            frame = self.repeats
            if a["gid"] is not None:
                self.node(a["gid"])
            if frame.empty:
                return {"matched": 0, "routes": [], "limitation": "Повторяющиеся цепочки не найдены или расчёт не обновлён."}
            if a["gid"] is not None:
                frame = frame[frame[["src", "via", "dst"]].eq(int(a["gid"])).any(axis=1)]
            evidence = []
            for row in frame.head(a["limit"]).itertuples(index=False):
                episodes = self.episodes[(self.episodes.src == row.src) & (self.episodes.via == row.via)
                                         & (self.episodes.dst == row.dst)]
                evidence.append({"src": str(row.src), "via": str(row.via), "dst": str(row.dst),
                                 "episode_count": int(row.episode_count), "examples": records(episodes.head(3))})
            return {"matched": len(frame), "routes": evidence,
                    "limitation": "1–2 дня, суммы 80–120%, непересекающиеся окна внутри маршрута. "
                                  "Операции могут повторяться между альтернативными маршрутами; это гипотезы."}
        raise ValueError("Неизвестная операция")
