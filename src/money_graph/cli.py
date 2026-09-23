from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ транзакционного графа")
    parser.add_argument("--data", type=Path, required=True, help="Папка с тремя parquet")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Папка результатов")
    args = parser.parse_args()

    result = analyze(args.data)
    write_outputs(result, args.out)
    print(f"Готово: {len(result.nodes)} узлов, {len(result.clusters)} кластеров")
    print(f"Результаты: {args.out.resolve()}")
    print(f"Время расчёта: {result.manifest['duration_seconds']} с")


if __name__ == "__main__":
    main()

