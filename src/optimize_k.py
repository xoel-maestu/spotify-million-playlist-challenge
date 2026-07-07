#!/usr/bin/env python3
"""
Optimización del hiperparámetro k para user-KNN e item-KNN.

Estrategias disponibles:
  - ternary : búsqueda ternaria O(log n) — asume curva unimodal en k
  - grid    : grid search logarítmico — no asume nada sobre la curva
  - full    : evalúa todos los valores en --k_values

La métrica a optimizar es configurable (R-Precision, NDCG, Clicks).
Para Clicks, se minimiza en lugar de maximizar.

Todos los pasos costosos (matrices, vecindarios) se reutilizan.
Solo se re-ejecuta el paso 3 (recomendación) y el paso 4 (evaluación)
para cada valor de k, que es rápido (~60s por evaluación).
"""

import gzip, json, time
from argparse import ArgumentParser
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import load_npz

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    TRAIN_NPZ_PATH, SONGS_PATH, POPULARITY_PATH,
    USER_NEIGHBORHOODS_PATH, ITEM_NEIGHBORHOODS_PATH,
    TEST_ZIP_PATH, TEAM_NAME, CONTACT_EMAIL,
)
from compute import load_neighborhoods
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
    parser = ArgumentParser(description="Optimización de k para KNN.")

    parser.add_argument(
        "--method",
        choices=["user_knn", "item_knn"],
        default="user_knn",
        help="Método KNN a optimizar",
    )
    parser.add_argument(
        "--strategy",
        choices=["ternary", "grid", "full"],
        default="ternary",
        help=(
            "Estrategia de búsqueda:\n"
            "  ternary : búsqueda ternaria O(log n), asume curva unimodal\n"
            "  grid    : grid logarítmico [1,2,5,10,25,50,100,200,500]\n"
            "  full    : evalúa los valores en --k_values"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["R-Precision", "NDCG", "Clicks"],
        default="NDCG",
        help="Métrica a optimizar (Clicks se minimiza, el resto se maximiza)",
    )
    parser.add_argument(
        "--k_min",  type=int, default=1,   help="k mínimo para búsqueda ternaria")
    parser.add_argument(
        "--k_max",  type=int, default=500, help="k máximo para búsqueda ternaria")
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[1, 2, 5, 10, 25, 50, 100, 200, 500],
        help="Valores de k para estrategia 'full'",
    )

    parser.add_argument("--num_recommendations", type=int, default=500)
    parser.add_argument("--n_jobs",              type=int, default=-1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluación de un k concreto
# ---------------------------------------------------------------------------

def evaluate_k(
    k: int,
    method: str,
    playlist_seeds: Dict[int, Set[str]],
    popular_tracks: List[str],
    ground_truth: Dict[int, List[str]],
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    train_matrix,
    user_neighborhoods: Dict,
    item_neighborhoods: Dict,
    submission_path: str,
    num_recommendations: int,
    n_jobs: int,
    team_name: str,
    contact_email: str,
) -> Dict[str, float]:
    """Genera recomendaciones con el k dado y devuelve las métricas."""

    recommendations_dict = generate_all_recommendations(
        playlist_seeds=playlist_seeds,
        method=method,
        popular_tracks=popular_tracks,
        songs=songs,
        inverse_songs=inverse_songs,
        train_matrix=train_matrix,
        user_neighborhoods=user_neighborhoods,
        item_neighborhoods=item_neighborhoods,
        k=k,
        num_recommendations=num_recommendations,
        n_jobs=n_jobs,
    )

    format_submission(
        recommendations_dict, submission_path, team_name, contact_email
    )

    recommendations = load_submission(submission_path)
    return evaluate(recommendations, ground_truth)


def score(results: Dict[str, float], metric: str) -> float:
    """
    Devuelve el valor escalar a MAXIMIZAR.
    Para Clicks (menor es mejor) devolvemos el negativo.
    """
    val = results[metric]
    return -val if metric == "Clicks" else val


# ---------------------------------------------------------------------------
# Estrategias de búsqueda
# ---------------------------------------------------------------------------

def grid_search(
    k_values: List[int],
    eval_fn,
    metric: str,
) -> Tuple[int, float, List[Tuple[int, Dict]]]:
    """
    Evalúa todos los k en k_values y devuelve el óptimo.

    Returns:
        best_k, best_score, history [(k, results), ...]
    """
    history = []
    best_k, best_s = None, -np.inf

    for k in sorted(k_values):
        t = time.time()
        results = eval_fn(k)
        s = score(results, metric)
        elapsed = time.time() - t

        history.append((k, results))
        marker = ""
        if s > best_s:
            best_s, best_k = s, k
            marker = "  ◄ mejor"

        print(f"  k={k:>4}  {metric}={results[metric]:.6f}  "
              f"R-P={results['R-Precision']:.6f}  "
              f"NDCG={results['NDCG']:.6f}  "
              f"Clicks={results['Clicks']:.4f}  "
              f"({elapsed:.1f}s){marker}")

    return best_k, best_s, history


def ternary_search(
    k_min: int,
    k_max: int,
    eval_fn,
    metric: str,
    min_interval: int = 3,
) -> Tuple[int, float, List[Tuple[int, Dict]]]:
    """
    Búsqueda ternaria sobre k asumiendo que la curva métrica vs k es unimodal.

    En cada iteración divide el intervalo [lo, hi] en tres partes y descarta
    el tercio con menor score, convergiendo al máximo en O(log_{3/2} n)
    evaluaciones ≈ 2*log2(n)/log2(3) ≈ 1.26*log2(n).

    Para k_max=500 esto son ~14 evaluaciones vs 500 de búsqueda exhaustiva.

    Si la curva no es unimodal, puede converger a un máximo local.
    En ese caso usar --strategy grid para confirmación.

    Returns:
        best_k, best_score, history [(k, results), ...]
    """
    cache: Dict[int, Dict] = {}
    history = []

    def cached_eval(k: int) -> float:
        if k not in cache:
            t = time.time()
            results = eval_fn(k)
            elapsed = time.time() - t
            cache[k] = results
            history.append((k, results))
            print(f"  k={k:>4}  {metric}={results[metric]:.6f}  "
                  f"R-P={results['R-Precision']:.6f}  "
                  f"NDCG={results['NDCG']:.6f}  "
                  f"Clicks={results['Clicks']:.4f}  "
                  f"({elapsed:.1f}s)")
        return score(cache[k], metric)

    lo, hi = k_min, k_max

    while hi - lo > min_interval:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3

        s1 = cached_eval(m1)
        s2 = cached_eval(m2)

        if s1 < s2:
            lo = m1
        else:
            hi = m2

    # Evaluar todos los enteros en el intervalo final pequeño
    for k in range(lo, hi + 1):
        cached_eval(k)

    best_k = max(cache, key=lambda k: score(cache[k], metric))
    best_s = score(cache[best_k], metric)

    # Marcar el mejor en el histórico
    print(f"\n  ► Mejor encontrado: k={best_k}  {metric}={cache[best_k][metric]:.6f}")

    return best_k, best_s, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _get_args()

    print(f"Método: {args.method}  |  Estrategia: {args.strategy}  "
          f"|  Optimizando: {args.metric}")
    print()

    # --- Cargar recursos (una sola vez) ---
    print("Cargando recursos...")

    with gzip.open(POPULARITY_PATH, "rt", encoding="utf-8") as f:
        popularity_dict = json.load(f)
    popular_tracks = list(popularity_dict.keys())
    print(f"  Popularidad: {len(popular_tracks)} canciones")

    playlist_seeds = get_playlist_seeds(TEST_ZIP_PATH)
    print(f"  Playlists de test: {len(playlist_seeds)}")

    ground_truth = load_ground_truth(TEST_ZIP_PATH)
    print(f"  Ground truth: {len(ground_truth)} playlists")

    with gzip.open(SONGS_PATH, "rt", encoding="utf-8") as f:
        songs = json.load(f)
    inverse_songs = {v: k for k, v in songs.items()}
    print(f"  Canciones: {len(songs)}")

    train_matrix       = None
    user_neighborhoods = None
    item_neighborhoods = None

    if args.method == "user_knn":
        train_matrix = load_npz(TRAIN_NPZ_PATH)
        user_neighborhoods = load_neighborhoods(USER_NEIGHBORHOODS_PATH)
        print(f"  Vecindarios user: {len(user_neighborhoods)}")
    elif args.method == "item_knn":
        item_neighborhoods = load_neighborhoods(ITEM_NEIGHBORHOODS_PATH)
        print(f"  Vecindarios item: {len(item_neighborhoods)}")

    print()

    # --- Función de evaluación parcial ---
    def eval_fn(k: int) -> Dict[str, float]:
        return evaluate_k(
            k=k,
            method=args.method,
            playlist_seeds=playlist_seeds,
            popular_tracks=popular_tracks,
            ground_truth=ground_truth,
            songs=songs,
            inverse_songs=inverse_songs,
            train_matrix=train_matrix,
            user_neighborhoods=user_neighborhoods,
            item_neighborhoods=item_neighborhoods,
            submission_path="./results/submission_optk.csv.gz",
            num_recommendations=args.num_recommendations,
            n_jobs=args.n_jobs,
            team_name=TEAM_NAME,
            contact_email=CONTACT_EMAIL,
        )

    # --- Ejecutar estrategia ---
    t_total = time.time()

    if args.strategy == "ternary":
        print(f"Búsqueda ternaria en k=[{args.k_min}, {args.k_max}]")
        print("-" * 70)
        best_k, best_s, history = ternary_search(
            args.k_min, args.k_max, eval_fn, args.metric
        )

    elif args.strategy == "grid":
        k_grid = [1, 2, 5, 10, 25, 50, 100, 200, 500]
        print(f"Grid logarítmico: k={k_grid}")
        print("-" * 70)
        best_k, best_s, history = grid_search(k_grid, eval_fn, args.metric)

    else:  # full
        print(f"Evaluación completa: k={sorted(args.k_values)}")
        print("-" * 70)
        best_k, best_s, history = grid_search(
            sorted(args.k_values), eval_fn, args.metric
        )

    # --- Resumen ---
    print()
    print("=" * 70)
    print(f"  RESULTADO OPTIMIZACIÓN ({args.strategy})")
    print("=" * 70)
    print(f"  Método   : {args.method}")
    print(f"  Métrica  : {args.metric}")
    print(f"  Mejor k  : {best_k}")

    best_results = next(r for k, r in history if k == best_k)
    print(f"  R-Precision : {best_results['R-Precision']:.6f}")
    print(f"  NDCG        : {best_results['NDCG']:.6f}")
    print(f"  Clicks      : {best_results['Clicks']:.4f}")
    print(f"  Evaluaciones: {len(history)}")
    print(f"  Tiempo total: {time.time() - t_total:.1f}s")
    print("=" * 70)
    # --- Guardar resultados en CSV ---
    import csv, os

    results_path = f"./results/optimize_k_{args.method}_{args.metric}.csv"
    os.makedirs("./results", exist_ok=True)

    history_sorted = sorted(history, key=lambda x: x[0])

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "R-Precision", "NDCG", "Clicks", "best"])
        for k, r in history_sorted:
            writer.writerow([
                k,
                f"{r['R-Precision']:.6f}",
                f"{r['NDCG']:.6f}",
                f"{r['Clicks']:.4f}",
                "yes" if k == best_k else "",
            ])

    print(f"  Resultados guardados en '{results_path}'")
    print()
    print(f"  Para usar el k óptimo en el pipeline:")
    print(f"  python pipeline.py --method {args.method} --k {best_k} "
          f"--skip_reader --skip_popularity --skip_neighborhoods")