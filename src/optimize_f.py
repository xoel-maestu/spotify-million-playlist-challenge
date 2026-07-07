#!/usr/bin/env python3
"""
Optimización del hiperparámetro f (factores latentes) para SVD.

Estrategias disponibles:
  - grid : evalúa f en [25, 50, 100, 200, 500, 1000] (por defecto)
  - full : evalúa los valores en --f_values

La métrica a optimizar es configurable (R-Precision, NDCG, Clicks).
Para Clicks, se minimiza en lugar de maximizar.

NOTA: a diferencia de KNN, para SVD no es posible reutilizar la
descomposición con distintos f sin recomputar (cada f requiere una SVD
distinta). El coste por evaluación es por tanto mayor (~1-5 min por f).
Se recomienda usar pocos valores bien espaciados.
"""

import gzip, json, time, csv, os
from argparse import ArgumentParser
from typing import Dict, List, Set

import numpy as np
from scipy.sparse import load_npz

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    TRAIN_NPZ_PATH, TEST_NPZ_PATH, SONGS_PATH, POPULARITY_PATH,
    SVD_COMBINED_PATH, SVD_TRAIN_PATH,
    TEST_ZIP_PATH, TEAM_NAME, CONTACT_EMAIL,
)
from compute import (
    compute_svd_combined, compute_svd_train,
    save_svd,
)
from evaluate import evaluate, load_ground_truth, load_submission
from recommender import (
    format_submission,
    generate_all_recommendations,
    get_playlist_seeds,
)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def _get_args():
    parser = ArgumentParser(description="Optimización de f para PureSVD.")

    parser.add_argument("--method",
        choices=["svd_combined", "svd_train"],
        default="svd_combined",
        help="Variante SVD a optimizar")
    parser.add_argument("--strategy",
        choices=["grid", "full"],
        default="grid",
        help="grid: valores predefinidos  |  full: valores en --f_values")
    parser.add_argument("--metric",
        choices=["R-Precision", "NDCG", "Clicks"],
        default="NDCG",
        help="Métrica a optimizar (Clicks se minimiza)")
    parser.add_argument("--f_values", type=int, nargs="+",
        default=[25, 50, 100, 200, 500],
        help="Valores de f para estrategia 'full'")
    parser.add_argument("--num_recommendations", type=int, default=500)
    parser.add_argument("--n_jobs",              type=int, default=-1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score(results: Dict, metric: str) -> float:
    val = results[metric]
    return -val if metric == "Clicks" else val


def evaluate_f(
    f: int,
    method: str,
    train_matrix,
    test_matrix,
    pids: List[int],
    playlist_seeds,
    popular_tracks,
    ground_truth,
    songs,
    inverse_songs,
    num_recommendations: int,
    n_jobs: int,
) -> Dict[str, float]:
    """Calcula la SVD con f factores y devuelve las métricas."""

    if method == "svd_combined":
        svd_data = compute_svd_combined(train_matrix, test_matrix, pids, f=f)
    else:
        svd_data = compute_svd_train(train_matrix, f=f)

    recs_dict = generate_all_recommendations(
        playlist_seeds=playlist_seeds,
        method=method,
        popular_tracks=popular_tracks,
        songs=songs,
        inverse_songs=inverse_songs,
        num_recommendations=num_recommendations,
        n_jobs=n_jobs,
        svd_data=svd_data,
    )

    tmp_path = "./results/submission_optf.csv.gz"
    format_submission(recs_dict, tmp_path, TEAM_NAME, CONTACT_EMAIL)
    recs = load_submission(tmp_path)
    return evaluate(recs, ground_truth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _get_args()

    print(f"Método: {args.method}  |  Estrategia: {args.strategy}  "
          f"|  Optimizando: {args.metric}")
    print()

    # --- Cargar recursos ---
    print("Cargando recursos...")

    with gzip.open(POPULARITY_PATH, "rt", encoding="utf-8") as f:
        popularity_dict = json.load(f)
    popular_tracks = list(popularity_dict.keys())

    playlist_seeds = get_playlist_seeds(TEST_ZIP_PATH)
    pids           = list(playlist_seeds.keys())
    ground_truth   = load_ground_truth(TEST_ZIP_PATH)

    with gzip.open(SONGS_PATH, "rt", encoding="utf-8") as f:
        songs = json.load(f)
    inverse_songs = {v: k for k, v in songs.items()}

    train_matrix = load_npz(TRAIN_NPZ_PATH)
    test_matrix  = load_npz(TEST_NPZ_PATH) if args.method == "svd_combined" else None

    print(f"  Playlists de test : {len(pids)}")
    print(f"  Canciones         : {len(songs)}")
    print()

    # --- Determinar valores de f ---
    # f_grid    = [25, 50, 100, 200, 500, 1000]
    f_grid    = [25, 50, 100, 200, 500]
    f_values  = f_grid if args.strategy == "grid" else sorted(args.f_values)

    print(f"Evaluando f={f_values}")
    print("-" * 70)

    history  = []
    best_f, best_s = None, -np.inf
    t_total  = time.time()

    for f in f_values:
        t = time.time()
        results = evaluate_f(
            f=f,
            method=args.method,
            train_matrix=train_matrix,
            test_matrix=test_matrix,
            pids=pids,
            playlist_seeds=playlist_seeds,
            popular_tracks=popular_tracks,
            ground_truth=ground_truth,
            songs=songs,
            inverse_songs=inverse_songs,
            num_recommendations=args.num_recommendations,
            n_jobs=args.n_jobs,
        )
        s       = score(results, args.metric)
        elapsed = time.time() - t
        history.append((f, results))

        marker = ""
        if s > best_s:
            best_s, best_f = s, f
            marker = "  ◄ mejor"

        print(f"  f={f:>5}  {args.metric}={results[args.metric]:.6f}  "
              f"R-P={results['R-Precision']:.6f}  "
              f"NDCG={results['NDCG']:.6f}  "
              f"Clicks={results['Clicks']:.4f}  "
              f"({elapsed:.1f}s){marker}")

    # --- Resumen ---
    print()
    print("=" * 70)
    print(f"  RESULTADO OPTIMIZACIÓN")
    print("=" * 70)
    print(f"  Método   : {args.method}")
    print(f"  Métrica  : {args.metric}")
    print(f"  Mejor f  : {best_f}")

    best_results = next(r for fv, r in history if fv == best_f)
    print(f"  R-Precision : {best_results['R-Precision']:.6f}")
    print(f"  NDCG        : {best_results['NDCG']:.6f}")
    print(f"  Clicks      : {best_results['Clicks']:.4f}")
    print(f"  Evaluaciones: {len(history)}")
    print(f"  Tiempo total: {time.time() - t_total:.1f}s")
    print("=" * 70)

    # --- Guardar CSV ---
    os.makedirs("./results", exist_ok=True)
    results_path = f"./results/optimize_f_{args.method}_{args.metric}.csv"

    with open(results_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f", "R-Precision", "NDCG", "Clicks", "best"])
        for fv, r in sorted(history, key=lambda x: x[0]):
            writer.writerow([
                fv,
                f"{r['R-Precision']:.6f}",
                f"{r['NDCG']:.6f}",
                f"{r['Clicks']:.4f}",
                "yes" if fv == best_f else "",
            ])

    print(f"  Resultados guardados en '{results_path}'")
    print()
    print(f"  Para usar el f óptimo en el pipeline:")
    print(f"  python pipeline.py --method {args.method} --f {best_f} "
          f"--skip_reader --skip_popularity --skip_svd")
    print(f"  (recuerda recomputar el SVD con --f {best_f} antes de saltar --skip_svd)")