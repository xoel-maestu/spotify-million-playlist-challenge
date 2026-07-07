#!/usr/bin/env python3
"""
Pipeline completo del Spotify Million Playlist Challenge.

Ejecuta las cuatro etapas de principio a fin para el método elegido
(popularity, user_knn, item_knn, svd_combined, svd_train, slim o fism):

  1. reader      – Construye las matrices dispersas de training y test (.npz)
  2. compute     – Precomputa el modelo: ranking de popularidad (siempre,
                   actúa como fallback) + vecindarios KNN, descomposición SVD
                   o entrenamiento SLIM/FISM según el método
  3. recommender – Genera las recomendaciones para las playlists de test
  4. evaluate    – Evalúa con R-Precision, NDCG y Clicks

Cada etapa puede omitirse con su flag --skip_* si su salida ya existe.
"""

from config import *

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gzip, json, time
from argparse import ArgumentParser

from scipy.sparse import load_npz

from compute import (
    compute_popularity,
    compute_user_neighborhoods,
    compute_item_neighborhoods,
    compute_svd_combined,
    compute_svd_train,
    compute_slim,
    compute_fism,
    save_popularity_ranking,
    save_neighborhoods,
    save_svd,
    save_slim_model,
    save_fism_model,
    load_neighborhoods,
    load_svd,
    load_slim_model,
    load_fism_model,
)
from evaluate import evaluate, load_ground_truth, load_submission
from reader import train_a_npz, test_a_npz
from recommender import (
    format_submission,
    generate_all_recommendations,
    get_playlist_seeds,
)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def _get_args():
    parser = ArgumentParser(description="Pipeline completo del MPD Challenge.")

    parser.add_argument("--method",
        choices=["popularity", "user_knn", "item_knn", "svd_combined", "svd_train",
                 "slim", "fism"],
        default="popularity",
        help="Método de recomendación")
    parser.add_argument("--K", type=int, default=500,
        help="Vecinos máximos a guardar en precomputación KNN")
    parser.add_argument("--k", type=int, default=50,
        help="Vecinos a usar en inferencia KNN (debe ser ≤ K)")
    parser.add_argument("--f", type=int, default=200,
        help="Número de factores latentes para SVD")
    parser.add_argument("--block_size", type=int, default=256,
        help="Tamaño de bloque para el cálculo de vecindarios")
    parser.add_argument("--n_jobs", type=int, default=-1,
        help="Procesos paralelos (-1 = todos los cores)")
    parser.add_argument("--num_recommendations", type=int, default=500,
        help="Número de canciones a recomendar por playlist")
    parser.add_argument("--skip_reader",        action="store_true",
        help="Saltar paso 1: usar matrices .npz ya existentes")
    parser.add_argument("--skip_popularity",    action="store_true",
        help="Saltar paso 2: usar popularidad ya calculada")
    parser.add_argument("--skip_neighborhoods", action="store_true",
        help="Saltar paso 2b: usar vecindarios ya precomputados (KNN)")
    parser.add_argument("--skip_svd",           action="store_true",
        help="Saltar paso 2b: usar SVD ya precomputado")
    parser.add_argument("--skip_slim",          action="store_true",
        help="Saltar paso 2b: usar modelo SLIM/FISM ya entrenado")
    parser.add_argument("--skip_recommender",   action="store_true",
        help="Saltar paso 3: usar submission ya existente")
    parser.add_argument("--n_factors", type=int, default=20,
        help="Número de factores latentes para FISM")
    parser.add_argument("--lambda_l1", type=float, default=0.1,
        help="Regularización L1 para SLIM/FISM")
    parser.add_argument("--lambda_l2", type=float, default=0.1,
        help="Regularización L2 para SLIM/FISM")
    parser.add_argument("--learning_rate", type=float, default=0.01,
        help="Tasa de aprendizaje para SLIM/FISM")
    parser.add_argument("--epochs", type=int, default=100,
        help="Épocas de entrenamiento para SLIM/FISM")
    parser.add_argument("--max_items", type=int, default=0,
        help="Filtrar a top-N ítems para SLIM (0=sin filtro; necesario si n_items>8000)")
    parser.add_argument("--trimmed", action="store_true",
        help="Usar el dataset reducido (trimmed_dataset.zip) en lugar del completo")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: str, title: str):
    print()
    print(f"{'=' * 55}")
    print(f"  PASO {n}: {title}")
    print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _get_args()

    if args.method in ("user_knn", "item_knn"):
        assert args.k <= args.K, (
            f"--k ({args.k}) debe ser ≤ --K ({args.K})."
        )

    # ------------------------------------------------------------------
    # Selección de rutas según --trimmed
    # ------------------------------------------------------------------
    if args.trimmed and args.method in ("slim", "fism"):
        # Matrices con vocabulario propio del trimmed (sin herencia del
        # dataset completo) para que n_items sea manejable para SLIM/FISM
        _TRAIN_ZIP  = TRIMMED_ZIP_PATH
        _TEST_ZIP   = TRIMMED_ZIP_PATH
        _TRAIN_NPZ  = COMPACT_NPZ_PATH
        _TEST_NPZ   = COMPACT_TEST_NPZ_PATH
        _SONGS      = COMPACT_SONGS_PATH
        _INPUT_JSON = TRIMMED_INPUT_JSON
        _EVAL_JSON  = TRIMMED_EVAL_JSON
        print("  [TRIMMED-COMPACT] Usando dataset reducido con vocabulario propio")
    elif args.trimmed:
        _TRAIN_ZIP  = TRIMMED_ZIP_PATH
        _TEST_ZIP   = TRIMMED_ZIP_PATH
        _TRAIN_NPZ  = TRIMMED_NPZ_PATH
        _TEST_NPZ   = TRIMMED_TEST_NPZ_PATH
        _SONGS      = TRIMMED_SONGS_PATH
        _INPUT_JSON = TRIMMED_INPUT_JSON
        _EVAL_JSON  = TRIMMED_EVAL_JSON
        print("  [TRIMMED] Usando dataset reducido")
    else:
        _TRAIN_ZIP  = TRAIN_ZIP_PATH
        _TEST_ZIP   = TEST_ZIP_PATH
        _TRAIN_NPZ  = TRAIN_NPZ_PATH
        _TEST_NPZ   = TEST_NPZ_PATH
        _SONGS      = SONGS_PATH
        _INPUT_JSON = TEST_INPUT_JSON
        _EVAL_JSON  = TEST_EVAL_JSON

    t_total = time.time()
    print(f"  Método: {args.method}")

    # ------------------------------------------------------------------
    # Paso 1: Construir matrices dispersas de training y test
    # ------------------------------------------------------------------
    _step(1, "Leer datasets → matrices dispersas")

    if args.skip_reader:
        print("  [omitido] Usando matrices existentes.")
    else:
        import os
        t = time.time()

        if os.path.exists(_SONGS):
            with gzip.open(_SONGS, "rt", encoding="utf-8") as f:
                songs = json.load(f)
        else:
            songs = dict()

        print("  Construyendo matriz de training...")
        songs = train_a_npz(_TRAIN_ZIP, _TRAIN_NPZ, songs)

        print("  Construyendo matriz de test...")
        songs = test_a_npz(_TEST_ZIP, _TEST_NPZ, songs, input_json=_INPUT_JSON)

        with gzip.open(_SONGS, "wt", encoding="utf-8") as f:
            json.dump(songs, f)
        print(f"  Correspondencia canciones guardada en '{_SONGS}'")
        print(f"  Tiempo: {time.time() - t:.1f}s")

    # ------------------------------------------------------------------
    # Paso 2: Calcular ranking de popularidad
    # ------------------------------------------------------------------
    _step(2, "Calcular ranking de popularidad")

    if args.skip_popularity:
        print("  [omitido] Usando ranking existente.")
    else:
        t = time.time()

        with gzip.open(_SONGS, "rt", encoding="utf-8") as f:
            songs = json.load(f)

        train_matrix = load_npz(_TRAIN_NPZ)
        print(f"  Playlists: {train_matrix.shape[0]:,}  |  Canciones: {train_matrix.shape[1]:,}")

        popularity_list = compute_popularity(train_matrix, songs)
        save_popularity_ranking(popularity_list, POPULARITY_PATH)

        print("  Top 5 canciones más populares:")
        for i, (uri, count) in enumerate(popularity_list[:5], 1):
            print(f"    {i}. {uri}  ({count:,} playlists)")
        print(f"  Tiempo: {time.time() - t:.1f}s")

    # ------------------------------------------------------------------
    # Paso 2b: Precomputar vecindarios (KNN) o SVD
    # ------------------------------------------------------------------
    if args.method in ("user_knn", "item_knn"):
        _step("2b", f"Precomputar vecindarios ({args.method}, K={args.K})")

        if args.skip_neighborhoods:
            print("  [omitido] Usando vecindarios existentes.")
        else:
            t = time.time()
            train_matrix = load_npz(_TRAIN_NPZ)

            if args.method == "user_knn":
                test_matrix = load_npz(_TEST_NPZ)
                playlist_seeds = get_playlist_seeds(_TEST_ZIP, input_json=_INPUT_JSON)
                pids = list(playlist_seeds.keys())

                neighborhoods = compute_user_neighborhoods(
                    train_matrix=train_matrix,
                    test_matrix=test_matrix,
                    pids=pids,
                    K=args.K,
                    block_size=args.block_size,
                    n_jobs=args.n_jobs,
                )
                save_neighborhoods(neighborhoods, USER_NEIGHBORHOODS_PATH)

            elif args.method == "item_knn":
                neighborhoods = compute_item_neighborhoods(
                    train_matrix=train_matrix,
                    K=args.K,
                    block_size=args.block_size,
                    n_jobs=args.n_jobs,
                )
                save_neighborhoods(neighborhoods, ITEM_NEIGHBORHOODS_PATH)

            print(f"  Tiempo: {time.time() - t:.1f}s")

    elif args.method in ("svd_combined", "svd_train"):
        _step("2b", f"Precomputar SVD ({args.method}, f={args.f})")

        if args.skip_svd:
            print("  [omitido] Usando SVD existente.")
        else:
            t = time.time()
            train_matrix = load_npz(_TRAIN_NPZ)

            if args.method == "svd_combined":
                test_matrix    = load_npz(_TEST_NPZ)
                playlist_seeds = get_playlist_seeds(_TEST_ZIP, input_json=_INPUT_JSON)
                pids           = list(playlist_seeds.keys())

                svd_data = compute_svd_combined(
                    train_matrix, test_matrix, pids, f=args.f
                )
                save_svd(svd_data, SVD_COMBINED_PATH)

            elif args.method == "svd_train":
                svd_data = compute_svd_train(train_matrix, f=args.f)
                save_svd(svd_data, SVD_TRAIN_PATH)

            print(f"  Tiempo: {time.time() - t:.1f}s")

    elif args.method in ("slim", "fism"):
        _step("2b", f"Entrenar {args.method.upper()} "
              f"(λ₁={args.lambda_l1}, β={args.lambda_l2})")

        if args.skip_slim:
            print("  [omitido] Usando modelo existente.")
        else:
            import os
            t = time.time()
            train_matrix = load_npz(_TRAIN_NPZ)

            if args.method == "slim":
                os.makedirs(os.path.dirname(SLIM_MODEL_PATH), exist_ok=True)
                slim_data = compute_slim(
                    train_matrix,
                    lambda_l1=args.lambda_l1,
                    lambda_l2=args.lambda_l2,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    max_items=args.max_items,
                )
                save_slim_model(slim_data, SLIM_MODEL_PATH)

            elif args.method == "fism":
                os.makedirs(os.path.dirname(FISM_P_PATH), exist_ok=True)
                fism_data = compute_fism(
                    train_matrix,
                    n_factors=args.n_factors,
                    lambda_l1=args.lambda_l1,
                    lambda_l2=args.lambda_l2,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                )
                save_fism_model(fism_data, FISM_P_PATH, FISM_Q_PATH)

            print(f"  Tiempo: {time.time() - t:.1f}s")

    # ------------------------------------------------------------------
    # Paso 3: Generar recomendaciones
    # ------------------------------------------------------------------
    _step(3, f"Generar recomendaciones ({args.method})")

    if args.skip_recommender:
        print("  [omitido] Usando submission existente.")
    else:
        t = time.time()

        print("  Cargando ranking de popularidad...")
        with gzip.open(POPULARITY_PATH, "rt", encoding="utf-8") as f:
            popularity_dict = json.load(f)
        popular_tracks = list(popularity_dict.keys())

        print("  Obteniendo semillas de las playlists de test...")
        playlist_seeds = get_playlist_seeds(_TEST_ZIP, input_json=_INPUT_JSON)
        print(f"  {len(playlist_seeds):,} playlists de test cargadas.")

        songs = None
        inverse_songs = None
        train_matrix = None
        user_neighborhoods = None
        item_neighborhoods = None
        svd_data = None
        slim_data = None
        fism_data = None

        if args.method in ("user_knn", "item_knn", "svd_combined", "svd_train",
                           "slim", "fism"):
            with gzip.open(_SONGS, "rt", encoding="utf-8") as f:
                songs = json.load(f)
            inverse_songs = {v: k for k, v in songs.items()}

        if args.method == "user_knn":
            train_matrix = load_npz(_TRAIN_NPZ)
            print("  Cargando vecindarios user-based...")
            user_neighborhoods = load_neighborhoods(USER_NEIGHBORHOODS_PATH)
            print(f"  {len(user_neighborhoods):,} vecindarios cargados.")

        elif args.method == "item_knn":
            print("  Cargando vecindarios item-based...")
            item_neighborhoods = load_neighborhoods(ITEM_NEIGHBORHOODS_PATH)
            print(f"  {len(item_neighborhoods):,} vecindarios cargados.")

        elif args.method == "svd_combined":
            print(f"  Cargando SVD conjunto desde '{SVD_COMBINED_PATH}'...")
            svd_data = load_svd(SVD_COMBINED_PATH)

        elif args.method == "svd_train":
            print(f"  Cargando SVD training desde '{SVD_TRAIN_PATH}'...")
            svd_data = load_svd(SVD_TRAIN_PATH)

        elif args.method == "slim":
            print(f"  Cargando modelo SLIM desde '{SLIM_MODEL_PATH}'...")
            slim_data = load_slim_model(SLIM_MODEL_PATH)

        elif args.method == "fism":
            print(f"  Cargando modelo FISM desde '{FISM_P_PATH}'...")
            fism_data = load_fism_model(FISM_P_PATH, FISM_Q_PATH)

        print("  Generando recomendaciones...")
        recommendations_dict = generate_all_recommendations(
            playlist_seeds=playlist_seeds,
            method=args.method,
            popular_tracks=popular_tracks,
            songs=songs,
            inverse_songs=inverse_songs,
            train_matrix=train_matrix,
            user_neighborhoods=user_neighborhoods,
            item_neighborhoods=item_neighborhoods,
            k=args.k,
            num_recommendations=args.num_recommendations,
            n_jobs=args.n_jobs,
            svd_data=svd_data,
            slim_data=slim_data,
            fism_data=fism_data,
        )

        format_submission(
            recommendations_dict,
            SUBMISSION_PATH,
            TEAM_NAME,
            CONTACT_EMAIL,
        )
        print(f"  Tiempo: {time.time() - t:.1f}s")

    # ------------------------------------------------------------------
    # Paso 4: Evaluar
    # ------------------------------------------------------------------
    _step(4, "Evaluar recomendaciones")

    t = time.time()
    recommendations = load_submission(SUBMISSION_PATH)
    ground_truth    = load_ground_truth(_TEST_ZIP, eval_json=_EVAL_JSON)
    results         = evaluate(recommendations, ground_truth)

    print(f"  R-Precision : {results['R-Precision']:.6f}")
    print(f"  NDCG        : {results['NDCG']:.6f}")
    print(f"  Clicks      : {results['Clicks']:.4f}")
    print(f"  Playlists   : {results['num_evaluated']:,} / {results['num_playlists']:,}")
    print(f"  Tiempo: {time.time() - t:.1f}s")

    print()
    print(f"{'=' * 55}")
    print(f"  Pipeline completado en {time.time() - t_total:.1f}s")
    print(f"{'=' * 55}")