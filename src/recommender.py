#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *

import gzip, json, zipfile
from typing import Dict, List, Set, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, load_npz
from tqdm import tqdm
from compute import _topk_1d, load_neighborhoods, load_svd


# ---------------------------------------------------------------------------
# Carga de seeds
# ---------------------------------------------------------------------------

def get_playlist_seeds(test_zip_path: str,
                       input_json: str = TEST_INPUT_JSON) -> Dict[int, Set[str]]:
    """
    Obtener las semillas de las playlists del JSON de input dentro del zip.

    Args:
        test_zip_path: Ruta al ZIP de test.
        input_json:    Nombre del fichero JSON dentro del ZIP
                       (por defecto TEST_INPUT_JSON de config).
    """
    playlist_seeds = dict()

    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        with zipf.open(input_json) as f:
            file_data = json.load(f)

            for playlist in file_data["playlists"]:
                # Tolerar playlists vacías
                seed_tracks = {
                    track["track_uri"] for track in playlist.get("tracks", list())
                }

                playlist_seeds[playlist["pid"]] = seed_tracks

    return playlist_seeds


# ---------------------------------------------------------------------------
# Popularidad
# ---------------------------------------------------------------------------

def generate_popularity_recommendations(
    seed_tracks: Set[str],
    popular_tracks: List[str],
    num_recommendations: int = 500,
) -> List[str]:
    """
    Generar recomendaciones para una playlist por popularidad global.

    Recorre el ranking de popularidad en orden descendente y añade canciones
    hasta alcanzar num_recommendations, saltando las que ya son semilla.

    Args:
        seed_tracks:         track_uris ya presentes en la playlist
        popular_tracks:      Lista de track_uris ordenada por popularidad desc.
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados (sin seeds, sin duplicados)
    """
    recommendations = list()

    for track_uri in popular_tracks:
        if track_uri not in seed_tracks:
            recommendations.append(track_uri)

            # Parar el bucle cuando se consigan todas las canciones que recomendar
            if len(recommendations) >= num_recommendations:
                break
    else:
        raise Exception("No se pudo llegar a las recomendaciones requeridas")

    return recommendations


# ---------------------------------------------------------------------------
# User-based KNN
# ---------------------------------------------------------------------------

def generate_userknn_recommendations(
    pid: int,
    seed_tracks: Set[str],
    neighborhoods: Dict[int, Tuple[np.ndarray, np.ndarray]],
    train_matrix: csr_matrix,
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    k: int = 150,
    num_recommendations: int = 500,
) -> List[str]:
    """
    Generar recomendaciones para una playlist usando user-based KNN.

    Fórmula:
        score(i) = Σ_{v ∈ V_u} s(u,v) · r(v,i)

    donde V_u son los k vecinos más similares de la playlist u en training,
    s(u,v) es la similitud coseno y r(v,i) es 1 si v contiene el item i.

    Optimizaciones:
    - Solo se puntúan items que aparecen en al menos un vecino del perfil
      (los demás tendrían score 0 por definición).
    - Fallback a popularidad si no hay vecinos precomputados (playlist vacía
      o sin ningún vecino con similitud > 0).

    Args:
        pid:               Identificador de la playlist de test
        seed_tracks:       Canciones ya presentes en la playlist (track_uris)
        neighborhoods:     Vecindarios precomputados pid → (indices, sims)
        train_matrix:      Matriz de training (n_playlists × n_songs)
        songs:             Correspondencia track_uri → columna
        inverse_songs:     Correspondencia columna → track_uri
        popular_tracks:    Ranking de popularidad para fallback
        k:                 Número de vecinos a usar (≤ K guardado en compute)
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    # Fallback si la playlist no tiene vecindario precomputado 
    # (vacía o sin vecinos) — fallback a popularidad
    if pid not in neighborhoods:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    neighbor_indices, neighbor_sims = neighborhoods[pid]

    # Truncar a k vecinos
    neighbor_indices = neighbor_indices[:k]
    neighbor_sims    = neighbor_sims[:k]

    # Extraer las filas de los k vecinos de la matriz de training
    # neighbor_matrix: (k, n_songs) — solo las filas de los vecinos
    neighbor_matrix = train_matrix[neighbor_indices]  # csr disperso

    # scores: (n_songs,) = neighbor_sims (k,) @ neighbor_matrix (k, n_songs)
    scores = np.asarray(neighbor_sims.astype(np.float32) @ neighbor_matrix).flatten()

    # Construir máscara de seeds a partir de track_uri (columnas a excluir)
    seed_cols = {songs[t] for t in seed_tracks if t in songs}

    # Excluír seeds: poner a 0 los scores de seeds
    # Caso límite: podría ser que todas las seeds fuesen exclusivas de test
    if seed_cols:
        seed_arr = np.array(list(seed_cols), dtype=np.int64)
        scores[seed_arr] = 0.0

    # Ordenar por tracks por score descendente y traducir a track_uris
    # argsort sobre scores completo; ignoramos los ceros implícitamente
    # usando solo los nonzero para eficiencia
    nonzero_cols = np.flatnonzero(scores)

    # Si todos los vecinos de la playlist contienen únicamente canciones
    # que ya son seeds — fallback a popularidad
    if len(nonzero_cols) == 0:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    # Ordenar solo los nonzero
    sorted_nonzero = nonzero_cols[_topk_1d(scores[nonzero_cols], num_recommendations)]

    recommendations = [
        inverse_songs[col] for col in sorted_nonzero if col in inverse_songs
    ]

    # Completar con popularidad si no hay suficientes candidatos KNN
    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended,
            popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


# ---------------------------------------------------------------------------
# Item-based KNN
# ---------------------------------------------------------------------------

def generate_itemknn_recommendations(
    seed_tracks: Set[str],
    item_neighborhoods: Dict[int, Tuple[np.ndarray, np.ndarray]],
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    k: int = 150,
    num_recommendations: int = 500,
) -> List[str]:
    """
    Generar recomendaciones para una playlist usando item-based KNN.

    Fórmula:
        score(i) = Σ_{j ∈ J_i ∩ seeds} s(i,j)

    donde J_i son los k items más similares al item i y s(i,j) es la
    similitud coseno entre items.

    Nota: sólo iteramos sobre las seeds porque son los únicos j donde r(u,j) = 1,
    es decir, son las únicas canciones para las que queremos comprobar sus vecinos.

    Optimizaciones:
    - Solo se consideran items que tienen al menos un vecino presente en
      las seeds (los demás tienen score 0).
    - Fallback a popularidad si no hay seeds o no hay vecindarios útiles.

    Args:
        seed_tracks:         Canciones ya presentes en la playlist
        item_neighborhoods:  Vecindarios precomputados col → (indices, sims)
        songs:               Correspondencia track_uri → columna
        inverse_songs:       Correspondencia columna → track_uri
        popular_tracks:      Ranking de popularidad para fallback
        k:                   Número de vecinos a usar (≤ K guardado en compute)
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    # Columnas de las canciones seed presentes en el catálogo
    seed_cols = {songs[t] for t in seed_tracks if t in songs}

    # Si la playlist está vacía — fallback a popularidad
    if not seed_cols:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    # Acumular scores: para cada item candidato, sumar similitudes con seeds
    scores: Dict[int, float] = {}

    # r(u,j) = 1 sólo para las seeds
    # Iteramos directamente sobre ellas y miramos qué items tienen como vecinos
    for seed_col in seed_cols:
        # Continuar cuando una canción sólo aparezca en test
        if seed_col not in item_neighborhoods:
            continue

        neighbor_indices, neighbor_sims = item_neighborhoods[seed_col]

        # Truncar a k vecinos
        neighbor_indices = neighbor_indices[:k]
        neighbor_sims    = neighbor_sims[:k]

        for nb_col, sim in zip(neighbor_indices, neighbor_sims):
            nb_col = int(nb_col)
            # Excluir items que ya son seed
            if nb_col in seed_cols:
                continue
            # Sumatorio de la similitud canciones i,j
            scores[nb_col] = scores.get(nb_col, 0.0) + float(sim)
 
    # Si todas las seeds son exclusivas de test o sus únicos vecinos son otras seeds,
    # no hay candidatos — fallback a popularidad
    if not scores:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    # Ordenar candidatos por score descendente
    cols = np.array(list(scores.keys()), dtype=np.int32)
    vals = np.array(list(scores.values()), dtype=np.float32)
    sorted_cols = cols[_topk_1d(vals, num_recommendations)]

    recommendations = [
        inverse_songs[col] for col in sorted_cols if col in inverse_songs
    ]

    # Completar con popularidad si no hay suficientes candidatos KNN
    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended,
            popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


# ---------------------------------------------------------------------------
# PureSVD (variante A: SVD conjunto)
# ---------------------------------------------------------------------------

def generate_svd_combined_recommendations(
    pid: int,
    seed_tracks: Set[str],
    svd_data: Dict,
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    num_recommendations: int = 500,
) -> List[str]:
    """
    Recomendaciones usando SVD conjunto (train + test input).

    El vector latente de la playlist ya está en U[pid_to_row[pid]].
    El score de la canción i es:
        score_i = u_u · diag(Σ) · v_iᵀ  = (u_u * Σ) · v_iᵀ

    Esto equivale al producto escalar de (U[row] * sigma) con cada columna de Vt,
    que es simplemente:  scores = (U[row] * sigma) @ Vt

    Args:
        pid:             Identificador de la playlist de test
        seed_tracks:     Canciones ya en la playlist
        svd_data:        Diccionario devuelto por compute_svd_combined / load_svd
        songs:           track_uri → columna
        inverse_songs:   columna → track_uri
        popular_tracks:  Ranking de popularidad para fallback
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    pid_to_row = svd_data["pid_to_row"]

    if pid not in pid_to_row:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    row   = pid_to_row[pid]
    U     = svd_data["U"]
    sigma = svd_data["sigma"]
    Vt    = svd_data["Vt"]

    # scores: (n_songs,)
    u_scaled = U[row] * sigma          # (f,)
    scores   = u_scaled @ Vt           # (n_songs,)

    # Enmascarar seeds
    seed_cols = {songs[t] for t in seed_tracks if t in songs}
    if seed_cols:
        scores[np.array(list(seed_cols), dtype=np.int64)] = -np.inf

    order = _topk_1d(scores, num_recommendations)
    recommendations = [
        inverse_songs[i] for i in order
        if i in inverse_songs and scores[i] > -np.inf
    ][:num_recommendations]

    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended,
            popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


# ---------------------------------------------------------------------------
# PureSVD (variante B: SVD solo training + proyección)
# ---------------------------------------------------------------------------

def generate_svd_train_recommendations(
    seed_tracks: Set[str],
    svd_data: Dict,
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    num_recommendations: int = 500,
) -> List[str]:
    """
    Recomendaciones usando SVD entrenado solo sobre training.

    Las playlists de test se proyectan al espacio latente mediante:
        ũ_{m+1} = r_{m+1} · Ṽ · Σ⁻¹

    donde r_{m+1} es un vector binario disperso con 1s en las columnas de
    las seeds. Equivalentemente:

        ũ_{m+1} = sum_{j ∈ seeds} Vt[:, j]ᵀ  ·  Σ⁻¹
                = Vt[:, seed_cols].T.sum(axis=0) / sigma

    El score de la canción i es:
        score_i = (ũ_{m+1} * Σ) · Vt[:, i]  = ũ_{m+1} @ diag(Σ) @ Vt

    Args:
        seed_tracks:     Canciones ya en la playlist
        svd_data:        Diccionario devuelto por compute_svd_train / load_svd
        songs:           track_uri → columna
        inverse_songs:   columna → track_uri
        popular_tracks:  Ranking de popularidad para fallback
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    if not seed_tracks:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    sigma = svd_data["sigma"]   # (f,)
    Vt    = svd_data["Vt"]      # (f, n_songs)

    seed_cols = np.array(
        [songs[t] for t in seed_tracks if t in songs], dtype=np.int64
    )

    if len(seed_cols) == 0:
        return generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations
        )

    # Proyección: r_{m+1} · V · Σ⁻¹
    # Vt[:, seed_cols] shape: (f, n_seeds) → suma por columnas → (f,)
    u_proj = Vt[:, seed_cols].sum(axis=1) / sigma   # (f,)

    # Score: ũ · Σ · Vt  = (u_proj * sigma) @ Vt
    scores = (u_proj * sigma) @ Vt                  # (n_songs,)

    seed_set = {songs[t] for t in seed_tracks if t in songs}
    if seed_set:
        scores[np.array(list(seed_set), dtype=np.int64)] = -np.inf

    order = _topk_1d(scores, num_recommendations)
    recommendations = [
        inverse_songs[i] for i in order
        if i in inverse_songs and scores[i] > -np.inf
    ][:num_recommendations]

    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended,
            popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


# ---------------------------------------------------------------------------
# SLIM / FISM
# ---------------------------------------------------------------------------

def generate_slim_recommendations(
    seed_tracks: Set[str],
    slim_data: Dict,
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    num_recommendations: int = 500,
) -> List[str]:
    """
    Recomendaciones usando la matriz de coeficientes S de SLIM.

    Fórmula:
        score(i) = Σ_{j ∈ seeds} S[j, i]
                 = (vector_seeds @ S)[i]

    Args:
        seed_tracks:         Canciones semilla de la playlist
        slim_data:           Diccionario con clave 'S' (csr_matrix n_items × n_items)
        songs:               track_uri → columna
        inverse_songs:       columna → track_uri
        popular_tracks:      Fallback de popularidad
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    seed_cols = np.array([songs[t] for t in seed_tracks if t in songs], dtype=np.int64)

    if len(seed_cols) == 0:
        return generate_popularity_recommendations(seed_tracks, popular_tracks, num_recommendations)

    S           = slim_data["S"]           # csr_matrix (n_sub × n_sub)
    item_subset = slim_data.get("item_subset")  # None si S cubre todos los ítems

    if item_subset is not None:
        # Mapear seed_cols (índices globales) a índices del subconjunto
        orig_to_sub = {int(orig): sub for sub, orig in enumerate(item_subset)}
        seed_sub = np.array([orig_to_sub[c] for c in seed_cols if c in orig_to_sub],
                            dtype=np.int32)
        if len(seed_sub) == 0:
            return generate_popularity_recommendations(seed_tracks, popular_tracks, num_recommendations)

        scores_sub = np.asarray(S[seed_sub].sum(axis=0)).flatten()

        # Proyectar de vuelta al espacio completo de ítems
        n_items_total = max(inverse_songs.keys()) + 1
        scores = np.zeros(n_items_total, dtype=np.float32)
        scores[item_subset] = scores_sub
    else:
        # scores = suma de filas de S correspondientes a las seeds
        scores = np.asarray(S[seed_cols].sum(axis=0)).flatten()

    # Excluir seeds
    scores[seed_cols] = 0.0

    nonzero_cols = np.flatnonzero(scores)
    if len(nonzero_cols) == 0:
        return generate_popularity_recommendations(seed_tracks, popular_tracks, num_recommendations)

    sorted_cols = nonzero_cols[_topk_1d(scores[nonzero_cols], num_recommendations)]
    recommendations = [inverse_songs[c] for c in sorted_cols if c in inverse_songs]

    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended, popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


def generate_fism_recommendations(
    seed_tracks: Set[str],
    fism_data: Dict,
    songs: Dict[str, int],
    inverse_songs: Dict[int, str],
    popular_tracks: List[str],
    num_recommendations: int = 500,
) -> List[str]:
    """
    Recomendaciones usando las matrices de factores P y Q de FISM.

    Fórmula (equivale a usar S = PQ^T sin materializarla):
        seed_vec = Σ_{j ∈ seeds} P[j]       → (n_factors,)
        score(i) = seed_vec · Q[i]           → escalar

    Args:
        seed_tracks:         Canciones semilla de la playlist
        fism_data:           Diccionario con claves 'P' y 'Q' (np.ndarray)
        songs:               track_uri → columna
        inverse_songs:       columna → track_uri
        popular_tracks:      Fallback de popularidad
        num_recommendations: Número de canciones a recomendar

    Returns:
        Lista de track_uris recomendados
    """
    seed_cols = np.array([songs[t] for t in seed_tracks if t in songs], dtype=np.int64)

    if len(seed_cols) == 0:
        return generate_popularity_recommendations(seed_tracks, popular_tracks, num_recommendations)

    P = fism_data["P"]  # (n_items, n_factors)
    Q = fism_data["Q"]  # (n_items, n_factors)

    # Vector semilla: suma de filas de P para las seeds
    seed_vec = P[seed_cols].sum(axis=0)   # (n_factors,)

    # Score para cada item: producto escalar con cada fila de Q
    scores = Q @ seed_vec                  # (n_items,)

    # Excluir seeds
    scores[seed_cols] = -np.inf

    order = _topk_1d(scores, num_recommendations)
    recommendations = [
        inverse_songs[i] for i in order
        if i in inverse_songs and scores[i] > -np.inf
    ][:num_recommendations]

    if len(recommendations) < num_recommendations:
        seed_tracks_extended = seed_tracks | set(recommendations)
        recommendations += generate_popularity_recommendations(
            seed_tracks_extended, popular_tracks,
            num_recommendations - len(recommendations),
        )

    return recommendations


# ---------------------------------------------------------------------------
# Dispatcher centralizado
# ---------------------------------------------------------------------------

def _recommend_one(
    pid: int,
    seed_tracks: Set[str],
    method: str,
    user_neighborhoods=None,
    train_matrix=None,
    item_neighborhoods=None,
    songs=None,
    inverse_songs=None,
    popular_tracks=None,
    k: int = 50,
    num_recommendations: int = 500,
    svd_data=None,
    slim_data=None,
    fism_data=None,
) -> Tuple[int, List[str]]:
    if method == "user_knn":
        recs = generate_userknn_recommendations(
            pid, seed_tracks, user_neighborhoods,
            train_matrix, songs, inverse_songs, popular_tracks,
            k, num_recommendations,
        )
    elif method == "item_knn":
        recs = generate_itemknn_recommendations(
            seed_tracks, item_neighborhoods,
            songs, inverse_songs, popular_tracks, k, num_recommendations,
        )
    elif method == "svd_combined":
        recs = generate_svd_combined_recommendations(
            pid, seed_tracks, svd_data,
            songs, inverse_songs, popular_tracks, num_recommendations,
        )
    elif method == "svd_train":
        recs = generate_svd_train_recommendations(
            seed_tracks, svd_data,
            songs, inverse_songs, popular_tracks, num_recommendations,
        )
    elif method == "slim":
        recs = generate_slim_recommendations(
            seed_tracks, slim_data,
            songs, inverse_songs, popular_tracks, num_recommendations,
        )
    elif method == "fism":
        recs = generate_fism_recommendations(
            seed_tracks, fism_data,
            songs, inverse_songs, popular_tracks, num_recommendations,
        )
    else:
        recs = generate_popularity_recommendations(
            seed_tracks, popular_tracks, num_recommendations,
        )

    return pid, recs


def generate_all_recommendations(
    playlist_seeds: Dict[int, Set[str]],
    method: str,
    popular_tracks: List[str],
    songs: Dict[str, int] = None,
    inverse_songs: Dict[int, str] = None,
    train_matrix: csr_matrix = None,
    user_neighborhoods: Dict = None,
    item_neighborhoods: Dict = None,
    k: int = 50,
    num_recommendations: int = 500,
    n_jobs: int = -1,
    svd_data: Dict = None,
    slim_data: Dict = None,
    fism_data: Dict = None,
) -> Dict[int, List[str]]:
    """
    Genera recomendaciones para todas las playlists de test en paralelo.

    Args:
        playlist_seeds:      pid → set de track_uris semilla
        method:              'popularity', 'user_knn', 'item_knn',
                             'svd_combined' o 'svd_train'
        popular_tracks:      Ranking de popularidad (fallback y método base)
        songs:               track_uri → columna (requerido para knn, svd, slim, fism)
        inverse_songs:       columna → track_uri (requerido para knn, svd, slim, fism)
        train_matrix:        Matriz de training (requerido para user_knn)
        user_neighborhoods:  Vecindarios user-based (requerido para user_knn)
        item_neighborhoods:  Vecindarios item-based (requerido para item_knn)
        k:                   Número de vecinos a usar (knn)
        num_recommendations: Número de canciones a recomendar por playlist
        n_jobs:              Procesos paralelos (-1 = todos los cores)
        svd_data:            Componentes SVD (requerido para svd_*)
        slim_data:           Modelo SLIM {'S': csr_matrix} (requerido para slim)
        fism_data:           Modelo FISM {'P': ndarray, 'Q': ndarray} (requerido para fism)

    Returns:
        Diccionario pid → lista de track_uris recomendados
    """
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_recommend_one)(
                pid, seed_tracks, method,
                user_neighborhoods=user_neighborhoods,
                train_matrix=train_matrix,
                item_neighborhoods=item_neighborhoods,
                songs=songs,
                inverse_songs=inverse_songs,
                popular_tracks=popular_tracks,
                k=k,
                num_recommendations=num_recommendations,
                svd_data=svd_data,
                slim_data=slim_data,
                fism_data=fism_data,
            )
            for pid, seed_tracks in playlist_seeds.items()
        ),
        total=len(playlist_seeds),
        desc="  Recomendaciones",
        unit="playlist",
    ))

    return dict(results)


# ---------------------------------------------------------------------------
# Formato de submission
# ---------------------------------------------------------------------------

def format_submission(
    recommendations_dict: Dict[int, List[str]],
    recommendations_path: str,
    team_name: str,
    contact_email: str,
):
    """
    Exportar las recomendaciones con el formato de la submission.

    Sigue la especificacion oficial de Spotify Million Playlist Challenge
    - Línea 1: team_info, team_name, contact_email
    - Líneas consecutivas: pid, track_uri_1, track_uri_2, ..., track_uri_n
    - Exactamente 500 canciones por playlist (sin duplicados ni seeds)
    - Salida como CSV comprimido con GZIP

    Args:
        recommendations_dict: Diccionario con las recomendaciones para las playlists
        output_path: Ruta de salida del archivo de recomendaciones (.csv.gz)
        team_name: Nombre del equipo o método
        contact_email: Correo de contacto
    """

    total_playlists = len(recommendations_dict)

    # Escribir el CSV comprimido
    with gzip.open(recommendations_path, "wt", encoding="utf-8") as f:
        # Cabecera con info
        f.write(f"team_info,{team_name},{contact_email}\n")

        # Recomendaciones para las playlists de test (ordenadas por pid)
        for pid in sorted(recommendations_dict.keys()):
            tracks = recommendations_dict[pid]

            # Formato: pid,track_uri_1,track_uri_2,...,track_uri_500
            line = f"{pid}," + ",".join(tracks)
            f.write(line + "\n")

    print(f"Archivo de salida guardado en '{recommendations_path}'")
    print(f"Total de playlists con recomendaciones: {total_playlists}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generación de recomendaciones MPD Challenge.")

    parser.add_argument(
        "--method",
        choices=["popularity", "user_knn", "item_knn", "svd_combined", "svd_train",
                 "slim", "fism"],
        default="popularity",
        help="Método de recomendación a utilizar",
    )
    parser.add_argument("--k", type=int, default=50,
        help="Número de vecinos a usar en KNN (debe ser ≤ K guardado en compute)")
    parser.add_argument("--num_recommendations", type=int, default=500,
        help="Número de recomendaciones por playlist")
    parser.add_argument("--n_jobs", type=int, default=-1,
        help="Procesos paralelos para generación de recomendaciones (-1 = todos)")

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    print("Cargando ranking de popularidad...")
    with gzip.open(POPULARITY_PATH, "rt", encoding="utf-8") as f:
        popularity_dict = json.load(f)
    popular_tracks = list(popularity_dict.keys())
    print(f"  {len(popular_tracks)} canciones cargadas")

    print("Obteniendo semillas de las playlists de test...")
    playlist_seeds = get_playlist_seeds(TEST_ZIP_PATH)
    print(f"  {len(playlist_seeds)} playlists cargadas")

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
        print("Cargando correspondencia de canciones...")
        with gzip.open(SONGS_PATH, "rt", encoding="utf-8") as f:
            songs = json.load(f)
        inverse_songs = {v: k for k, v in songs.items()}
        print(f"  {len(songs)} canciones")

    if args.method == "user_knn":
        train_matrix = load_npz(TRAIN_NPZ_PATH)
        user_neighborhoods = load_neighborhoods(USER_NEIGHBORHOODS_PATH)
        print(f"  {len(user_neighborhoods)} vecindarios cargados")

    elif args.method == "item_knn":
        item_neighborhoods = load_neighborhoods(ITEM_NEIGHBORHOODS_PATH)
        print(f"  {len(item_neighborhoods)} vecindarios cargados")

    elif args.method == "svd_combined":
        print(f"Cargando SVD conjunto desde '{SVD_COMBINED_PATH}'...")
        svd_data = load_svd(SVD_COMBINED_PATH)

    elif args.method == "svd_train":
        print(f"Cargando SVD training desde '{SVD_TRAIN_PATH}'...")
        svd_data = load_svd(SVD_TRAIN_PATH)

    elif args.method == "slim":
        from compute import load_slim_model
        print(f"Cargando modelo SLIM desde '{SLIM_MODEL_PATH}'...")
        slim_data = load_slim_model(SLIM_MODEL_PATH)

    elif args.method == "fism":
        from compute import load_fism_model
        print(f"Cargando modelo FISM desde '{FISM_P_PATH}'...")
        fism_data = load_fism_model(FISM_P_PATH, FISM_Q_PATH)

    print(f"\nGenerando recomendaciones con método '{args.method}'...")
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

    print("\nGuardando submission...")
    format_submission(
        recommendations_dict,
        SUBMISSION_PATH,
        TEAM_NAME,
        CONTACT_EMAIL,
    )