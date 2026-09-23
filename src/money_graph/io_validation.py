from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMAS = {
    "nodes": {"gid", "depth", "is_seed"},
    "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
    "transactions": {"src", "dst", "date", "sum_kzt"},
}


class DataValidationError(ValueError):
    """Raised when input files violate the analysis contract."""


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        import duckdb

        return duckdb.connect().execute("select * from read_parquet(?)", [str(path)]).fetchdf()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in SCHEMAS:
        path = data_dir / f"{name}.parquet"
        if not path.is_file():
            raise DataValidationError(f"Не найден обязательный файл: {path}")
        frames[name] = _read_parquet(path)

    frames["transactions"]["date"] = pd.to_datetime(
        frames["transactions"]["date"], errors="raise"
    )
    return frames["nodes"], frames["edges"], frames["transactions"]


def validate_data(
    nodes: pd.DataFrame, edges: pd.DataFrame, tx: pd.DataFrame
) -> dict[str, Any]:
    frames = {"nodes": nodes, "edges": edges, "transactions": tx}
    errors: list[str] = []
    warnings: list[str] = []

    for name, required in SCHEMAS.items():
        missing = sorted(required - set(frames[name].columns))
        if missing:
            errors.append(f"{name}: отсутствуют колонки {missing}")
        if frames[name].empty:
            errors.append(f"{name}: файл пуст")
        if frames[name].isna().any().any():
            errors.append(f"{name}: найдены пустые значения")

    if errors:
        raise DataValidationError("; ".join(errors))

    if nodes["gid"].duplicated().any():
        errors.append("nodes: gid должен быть уникальным")
    if edges.duplicated(["src", "dst"]).any():
        errors.append("edges: пара src,dst должна быть уникальной")
    if (edges["sum_kzt"] <= 0).any() or (tx["sum_kzt"] <= 0).any():
        errors.append("Суммы переводов должны быть положительными")
    if (edges["n_tx"] <= 0).any():
        errors.append("n_tx должен быть положительным")
    if not np.isfinite(edges["sum_kzt"]).all() or not np.isfinite(tx["sum_kzt"]).all():
        errors.append("В суммах найдены бесконечные значения")

    gids = set(nodes["gid"].astype(int))
    endpoints = set(edges["src"].astype(int)) | set(edges["dst"].astype(int))
    endpoints |= set(tx["src"].astype(int)) | set(tx["dst"].astype(int))
    unknown = sorted(endpoints - gids)
    if unknown:
        errors.append(f"Найдены связи с отсутствующими gid: {unknown[:5]}")

    if not ((nodes["depth"] == 0) == nodes["is_seed"].astype(bool)).all():
        warnings.append("is_seed и depth=0 совпадают не для всех узлов")

    tx_agg = (
        tx.groupby(["src", "dst"], as_index=False)
        .agg(tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    )
    check = edges.merge(tx_agg, on=["src", "dst"], how="outer", indicator=True)
    if not (check["_merge"] == "both").all():
        errors.append("Наборы пар src,dst в edges и transactions не совпадают")
    else:
        sums_match = np.isclose(check["sum_kzt"], check["tx_sum"], rtol=0, atol=0.01)
        if not sums_match.all():
            errors.append("Суммы transactions не совпадают с edges")
        if not (check["n_tx"] == check["tx_count"]).all():
            errors.append("Количество transactions не совпадает с n_tx")

    if errors:
        raise DataValidationError("; ".join(errors))

    repeated_rows = int(tx.duplicated().sum())
    if repeated_rows:
        warnings.append(
            f"{repeated_rows} полностью совпадающих строк сохранены: transaction_id отсутствует"
        )

    return {
        "status": "ok",
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "warnings": warnings,
        "period": {"from": str(tx["date"].min().date()), "to": str(tx["date"].max().date())},
        "sum_kzt": float(edges["sum_kzt"].sum()),
        "n_seed": int(nodes["is_seed"].sum()),
    }


def file_hashes(data_dir: Path) -> dict[str, str]:
    result = {}
    for name in SCHEMAS:
        path = data_dir / f"{name}.parquet"
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

