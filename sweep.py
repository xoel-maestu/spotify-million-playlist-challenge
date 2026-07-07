#!/usr/bin/env python3
"""
Barrido de hiperparámetros para SLIM y FISM sobre el dataset trimmed.

Lanza pipeline.py por cada combinación del grid, parsea las métricas
de evaluación y guarda un CSV por método en results/sweep/.

Uso:
    python sweep.py slim
    python sweep.py fism
    python sweep.py all

Asume que las matrices .npz y la popularidad ya están construidas
(usa --skip_reader --skip_popularity en cada llamada).
"""

import csv
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Grids de hiperparámetros — edita aquí
# ---------------------------------------------------------------------------

SLIM_GRID = {
    "lambda_l1":     [0.001, 0.01, 0.1, 1.0],
    "lambda_l2":     [0.001, 0.01, 0.1, 1.0],
    "learning_rate": [0.01],
    "epochs":        [100, 300],
}

FISM_GRID = {
    "n_factors":     [20, 50, 100],
    "lambda_l1":     [0.001, 0.01, 0.1],
    "lambda_l2":     [0.001, 0.01, 0.1],
    "learning_rate": [0.01],
    "epochs":        [200, 500],
}

# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parent
PYTHON   = sys.executable
PIPELINE = ROOT / "pipeline.py"
OUT_DIR  = ROOT / "results" / "sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_RE = re.compile(
    r"R-Precision\s*:\s*([\d.]+).*?NDCG\s*:\s*([\d.]+).*?Clicks\s*:\s*([\d.]+)",
    re.DOTALL,
)


def run_one(method: str, params: dict):
    """Lanza pipeline.py una vez. Devuelve dict con métricas o None si falla."""
    args = [PYTHON, str(PIPELINE), "--method", method, "--trimmed",
            "--skip_reader", "--skip_popularity"]
    for k, v in params.items():
        args += [f"--{k}", str(v)]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silenciar logs INFO/WARNING de TF

    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        print(f"    FAILED (rc={proc.returncode}): {' | '.join(tail)}")
        return None

    m = METRIC_RE.search(proc.stdout)
    if not m:
        print(f"    No se encontraron métricas en la salida")
        return None

    return {
        "R_Precision": float(m.group(1)),
        "NDCG":        float(m.group(2)),
        "Clicks":      float(m.group(3)),
        "time_s":      round(elapsed, 1),
    }


def sweep(method: str, grid: dict):
    keys   = list(grid.keys())
    combos = list(product(*grid.values()))
    rows   = []

    print(f"\n=== {method.upper()}: {len(combos)} combinaciones ===")
    t_start = time.time()

    for i, values in enumerate(combos, 1):
        params    = dict(zip(keys, values))
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i:>3}/{len(combos)}] {param_str}")

        result = run_one(method, params)
        if result is None:
            continue

        rows.append({**params, **result})
        print(f"        R-P={result['R_Precision']:.4f}   "
              f"NDCG={result['NDCG']:.4f}   "
              f"Clicks={result['Clicks']:.2f}   "
              f"({result['time_s']:.0f}s)")

    if not rows:
        print(f"\n{method}: sin resultados — todos los runs fallaron.")
        return

    # Guardado del CSV
    out_path = OUT_DIR / f"{method}_sweep.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Resumen: top 5 por NDCG
    rows.sort(key=lambda r: -r["NDCG"])
    print(f"\nCSV: {out_path}")
    print(f"Tiempo total: {time.time() - t_start:.0f}s\n")
    print("Top 5 por NDCG:")
    hyper_keys = [k for k in rows[0].keys() if k not in
                  {"R_Precision", "NDCG", "Clicks", "time_s"}]
    header = "  ".join(f"{k:>10}" for k in hyper_keys) + \
             f"   {'R-P':>7}  {'NDCG':>7}  {'Clicks':>7}"
    print("  " + header)
    for r in rows[:5]:
        cfg = "  ".join(f"{r[k]!s:>10}" for k in hyper_keys)
        print(f"  {cfg}   {r['R_Precision']:7.4f}  {r['NDCG']:7.4f}  {r['Clicks']:7.2f}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"slim", "fism", "all"}:
        print(f"Uso: python {Path(sys.argv[0]).name} {{slim|fism|all}}")
        sys.exit(1)

    target = sys.argv[1]
    if target in {"slim", "all"}:
        sweep("slim", SLIM_GRID)
    if target in {"fism", "all"}:
        sweep("fism", FISM_GRID)


if __name__ == "__main__":
    main()
