"""One-command local launch, using only the standard library to bootstrap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate results and launch the AML viewer")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-ui", action="store_true", help="Calculate CSV files and exit")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10+ is required")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    data, out = args.data.resolve(), args.out.resolve()
    missing = [name for name in ("nodes.parquet", "edges.parquet", "transactions.parquet")
               if not (data / name).is_file()]
    if missing:
        print(f"Missing data in {data}: {', '.join(missing)}\n"
              "Extract the supplied dataset into that folder, or use --data PATH.", file=sys.stderr)
        return 1
    if not args.no_ui:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", args.port))
            except OSError:
                print(f"Port {args.port} is busy. If the app is already running, open "
                      f"http://localhost:{args.port}; otherwise use --port 8502.", file=sys.stderr)
                return 1
    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("[1/3] Creating local Python environment", flush=True)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MONEY_GRAPH_DATA"], env["MONEY_GRAPH_OUT"] = str(data), str(out)
    # Check actual package constraints instead of relying on a stale install marker.
    check = (
        "from importlib.metadata import version; from packaging.requirements import Requirement; "
        "from pathlib import Path; "
        "requirements=[Requirement(x) for x in Path('requirements.txt').read_text().splitlines() "
        "if x.strip() and not x.startswith('#')]; "
        "assert all(version(r.name) in r.specifier for r in requirements)"
    )
    ready = subprocess.run([str(python), "-c", check], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ready.returncode:
        print("[1/3] Installing dependencies (internet required on first launch)", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements.txt"],
                       cwd=ROOT, env=env, check=True)
    else:
        print("[1/3] Dependencies ready", flush=True)
    print("[2/3] Validating data and calculating CSV files", flush=True)
    subprocess.run([str(python), "-m", "money_graph.cli", "--data", str(data),
                    "--out", str(out)], cwd=ROOT, env=env, check=True)
    if args.no_ui:
        print(f"Done: {out}")
        return 0
    print(f"[3/3] Open http://localhost:{args.port} | Stop: Ctrl+C", flush=True)
    return subprocess.call([str(python), "-m", "streamlit", "run", "app.py",
                            "--server.address", "127.0.0.1", "--server.port", str(args.port)],
                           cwd=ROOT, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
