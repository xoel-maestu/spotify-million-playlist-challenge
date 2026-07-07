#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *

import gzip, json
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import load_npz, csr_matrix, save_npz, vstack
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Popularidad
# ---------------------------------------------------------------------------

def compute_popularity(matrix, songs: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Obtener la popularidad de las canciones como recuento del número de
    playlists en las que aparecen.

    Args:
        matrix: Matriz dispersa con playlists en filas y canciones en columnas
        songs:  Correspondencia track_uri → columna

    Returns:
        Lista de tuplas (track_uri, recuento) ordenada por popularidad descendente
    """

    # Suma las columas para agregar el número de playlsits que contienen la canción
    popularity_counts = np.array(matrix.sum(axis=0)).flatten()

    # Crear correspondencia de columna a uri de la canción
    inverse_songs = {col_idx: uri for uri, col_idx in songs.items()}

    # Crea lista de tuplas (track_uri, popularidad) TODO: zip()?
    popularity_list = []
    for col_idx in range(len(popularity_counts)):
        if col_idx in inverse_songs:
            song_uri = inverse_songs[col_idx]
            count = int(popularity_counts[col_idx])
            popularity_list.append((song_uri, count))

    # Ordenar por popularidad descendente
    popularity_list.sort(key=lambda x: x[1], reverse=True)

    return popularity_list


def save_popularity_ranking(
    popularity_list: List[Tuple[str, int]], popularity_path: str
):
    """
    Guardar el ranking de popularidad en un JSON comprimido.

    Args:
        popularity_list: Lista de tuplas (track_uri, recuento)
        popularity_path: Ruta al archivo de salida
    """
    # Convert to dictionary for easier lookup TODO: dict(popularity_list)?
    popularity_dict = {uri: count for uri, count in popularity_list}

    with gzip.open(popularity_path, "wt", encoding="utf-8") as f:
        json.dump(popularity_dict, f)
    print(f"Ranking de popularidad guardado en '{popularity_path}'")


# ---------------------------------------------------------------------------
# Vecindarios KNN
# ---------------------------------------------------------------------------

# --- Utilidades comunes ---

def _row_norms(matrix: csr_matrix) -> np.ndarray:
    """
    Calcula la norma L2 de cada fila de una matriz binaria dispersa.
    Para una matriz binaria, norm(fila) = sqrt(num_elementos_en_fila).

    La normalización elimina la ventaja de vectores más largos (playlists
    con muchas canciones o canciones que aparecen en muchas playlists).
    """
    # matrix.sum(axis=1) devuelve un np.matrix (n,1); lo aplanamos
    return np.sqrt(np.asarray(matrix.sum(axis=1)).flatten())


def _col_norms(matrix: csr_matrix) -> np.ndarray:
    """Calcula la norma L2 de cada columna de una matriz binaria dispersa."""
    return np.sqrt(np.asarray(matrix.sum(axis=0)).flatten())


def _topk_1d(sims: np.ndarray, K: int) -> np.ndarray:
    """
    Devuelve los índices que ordenan sims de mayor a menor, truncando a K.

    Usa argpartition O(n) + argsort O(K log K) en lugar de argsort O(n log n)
    cuando hay más de K elementos.

    Args:
        sims: Array 1D de similitudes
        K:    Número máximo de vecinos a devolver

    Returns:
        Array de índices ordenados de mayor a menor similitud (longitud ≤ K)
    """
    if len(sims) <= K:
        return np.argsort(-sims)

    # argpartition garantiza que los K mayores están en las primeras K posiciones
    part = np.argpartition(-sims, K)[:K]
    return part[np.argsort(-sims[part])]


# --- Procesamiento de bloques ---

def _process_block(
    block_rows: np.ndarray,
    matrix_a: csr_matrix,
    matrix_b_T: csr_matrix,
    norms_a: np.ndarray,
    norms_b: np.ndarray,
    K: int,
    exclude_self: bool = False,
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Calcula los K vecinos más similares (similitud coseno) para un bloque
    de vectores de matrix_a contra todos los vectores de matrix_b.

    Función común para user-KNN e item-KNN:
      - User-KNN:  matrix_a = test,  matrix_b = train  → vecinos de playlists
      - Item-KNN:  matrix_a = train^T, matrix_b = train → vecinos de canciones

    Opera completamente en disperso: nunca densifica la matriz completa.

    Args:
        block_rows:  Índices de fila de matrix_a a procesar en este bloque
        matrix_a:    Matriz de consulta  (n_a × n_features)
        matrix_b_T:  Traspuesta de la matriz objetivo (n_features × n_b),
                     precomputada fuera del bucle para evitar recalcularla
        norms_a:     Normas L2 de las filas de matrix_a
        norms_b:     Normas L2 de las filas de matrix_b (columnas de matrix_b_T)
        K:           Número máximo de vecinos a guardar
        exclude_self: Si True, pone a 0 la similitud de cada elemento consigo
                      mismo (necesario en item-KNN, donde matrix_a = matrix_b)

    Returns:
        ids:     Lista de índices procesados
        indices: Lista de arrays con los índices de los K vecinos
        values:  Lista de arrays con las similitudes correspondientes
    """
    ids_out, indices_out, values_out = [], [], []

    # Multiplicación dispersa: (n_block, n_features) @ (n_features, n_b)
    # Solo tiene entradas donde los vectores comparten features → muy disperso
    block_matrix = matrix_a[block_rows]
    dot_sparse   = (block_matrix @ matrix_b_T).tocsr()

    for local_i, row_id in enumerate(block_rows):
        # Acceso directo a la estructura CSR para evitar duplicar matriz
        # CSR por cada fila: indptr marca el rango de datos de cada fila
        # en los arrays planos indices y data
        row_start = dot_sparse.indptr[local_i]
        row_end   = dot_sparse.indptr[local_i + 1]

        if row_start == row_end:
            # Sin entradas no nulas: ningún vecino en común
            continue

        nb_idx = dot_sparse.indices[row_start:row_end]  # índices de vecinos candidatos
        nb_dot = dot_sparse.data[row_start:row_end]     # productos escalares

        # Similitud coseno: dot(a, b) / (norm(a) * norm(b))
        denom = norms_a[row_id] * norms_b[nb_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            sims = np.where(denom > 0, nb_dot / denom, 0.0)

        if exclude_self:
            # En item-KNN matrix_a = matrix_b, por lo que row_id aparece
            # como su propio vecino con similitud 1.0 — lo excluimos
            self_mask = nb_idx == row_id
            sims[self_mask] = 0.0

        # Top-K y filtrado de similitudes cero
        order   = _topk_1d(sims, K)
        top_idx = nb_idx[order]
        top_val = sims[order]

        nonzero_mask = top_val > 0
        if nonzero_mask.any():
            ids_out.append(int(row_id))
            indices_out.append(top_idx[nonzero_mask].astype(np.int32))
            values_out.append(top_val[nonzero_mask].astype(np.float32))

    return ids_out, indices_out, values_out


# --- Vecindarios ---

def compute_user_neighborhoods(
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    pids: List[int],
    K: int = 500,
    block_size: int = 500,
    n_jobs: int = -1,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Precomputa los K vecinos más similares en training para cada playlist
    de test (user-based KNN con similitud coseno).

    Compara filas de test_matrix contra filas de train_matrix.
    Paralelización con joblib prefer='processes': cada worker recibe
    su bloque de test y opera de forma independiente sin GIL.

    Args:
        train_matrix: Matriz dispersa de training (n_train × n_songs)
        test_matrix:  Matriz dispersa de test    (n_test × n_songs)
        pids:         Lista de PIDs de test a procesar (índices de fila)
        K:            Número máximo de vecinos a guardar
        block_size:   Filas de test por bloque por worker
        n_jobs:       Cores a usar (-1 = todos)

    Returns:
        Diccionario pid → (indices_vecinos, similitudes)
    """
    train_norms = _row_norms(train_matrix)
    test_norms  = _row_norms(test_matrix)

    # train_matrix.T se precomputa UNA sola vez y se pasa a todos los workers
    train_T = train_matrix.T.tocsr()

    # Dividir los PIDs en bloques para procesamiento paralelo
    pids_array = np.array(pids, dtype=np.int64)
    blocks = [
        pids_array[i : i + block_size]
        for i in range(0, len(pids_array), block_size)
    ]

    print(f"  User-KNN: {len(pids)} playlists, K={K}, {len(blocks)} bloques")

    results = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
            delayed(_process_block)(
                block, test_matrix, train_T, test_norms, train_norms, K
            )
            for block in blocks
        ),
        total=len(blocks),
        desc="  Vecindarios user-KNN",
        unit="bloque",
    ))

    neighborhoods: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for ids_b, indices_b, values_b in results:
        for pid, idx, val in zip(ids_b, indices_b, values_b):
            neighborhoods[pid] = (idx, val)

    print(f"  Vecindarios calculados: {len(neighborhoods)} / {len(pids)} playlists")
    return neighborhoods


def compute_item_neighborhoods(
    train_matrix: csr_matrix,
    K: int = 500,
    block_size: int = 64,
    n_jobs: int = -1,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Precomputa los K items más similares para cada item del catálogo
    (item-based KNN con similitud coseno).

    Compara columnas de train_matrix entre sí (transponiendo para operar
    por filas). Paralelización con joblib prefer='threads': las operaciones
    numpy/scipy liberan el GIL internamente, permitiendo paralelismo real
    sin el overhead de copiar matrices entre procesos.

    ADVERTENCIA: con 2M de items es muy costoso. Se recomienda usar un K
    grande y truncar en recommender.py para optimizar k sin recomputar.

    Args:
        train_matrix: Matriz dispersa de training (n_playlists × n_items)
        K:            Número máximo de vecinos a guardar por item
        block_size:   Items por bloque por worker
        n_jobs:       Cores a usar (-1 = todos)

    Returns:
        Diccionario item_col → (indices_vecinos, similitudes)
    """
    item_norms = _col_norms(train_matrix)

    # Trabajamos con la traspuesta: cada canción pasa a ser una fila
    # train_T shape: (n_items, n_playlists)
    # .tocsr() para CSC to CSC
    train_T = train_matrix.T.tocsr()

    # Solo procesar items que aparecen en al menos una playlist (norma > 0)
    active_items = np.where(item_norms > 0)[0]

    blocks = [
        active_items[i : i + block_size]
        for i in range(0, len(active_items), block_size)
    ]

    print(f"  Item-KNN: {len(active_items)} items activos de {train_matrix.shape[1]}, "
          f"K={K}, {len(blocks)} bloques")

    # En item-KNN matrix_a = train_T y matrix_b_T = train_matrix
    # exclude_self=True para no recomendar una canción como vecina de sí misma
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_process_block)(
                block, train_T, train_matrix, item_norms, item_norms, K,
                exclude_self=True,
            )
            for block in blocks
        ),
        total=len(blocks),
        desc="  Vecindarios item-KNN",
        unit="bloque",
    ))

    neighborhoods: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for ids_b, indices_b, values_b in results:
        for item, idx, val in zip(ids_b, indices_b, values_b):
            neighborhoods[item] = (idx, val)

    print(f"  Vecindarios de items calculados: {len(neighborhoods)}")
    return neighborhoods


# ---------------------------------------------------------------------------
# Guardado y Carga de vecindarios
# ---------------------------------------------------------------------------

def save_neighborhoods(
    neighborhoods: Dict[int, Tuple[np.ndarray, np.ndarray]],
    path: str,
):
    """
    Guarda el diccionario de vecindarios en formato NPZ comprimido.

    Formato interno:
        ids_{id}  → array de índices de vecinos
        vals_{id} → array de similitudes

    Args:
        neighborhoods: Diccionario id → (indices, similitudes)
        path:          Ruta del archivo .npz de salida
    """
    arrays = {}
    for key, (idx, val) in neighborhoods.items():
        arrays[f"ids_{key}"]  = idx.astype(np.int32)
        arrays[f"vals_{key}"] = val.astype(np.float32)
    np.savez_compressed(path, **arrays)
    print(f"Vecindarios guardados en '{path}'")


def load_neighborhoods(
    path: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Carga los vecindarios guardados con save_neighborhoods.

    Args:
        path: Ruta del archivo .npz

    Returns:
        Diccionario id → (indices, similitudes)
    """
    data = np.load(path)
    keys = {k.split("_", 1)[1] for k in data.files}
    return {
        int(key): (data[f"ids_{key}"], data[f"vals_{key}"])
        for key in keys
    }


# ---------------------------------------------------------------------------
# PureSVD
# ---------------------------------------------------------------------------

def compute_svd_combined(
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    pids: List[int],
    f: int = 200,
) -> Dict:
    """
    Variante A — SVD conjunto (train + test input).

    Construye una matriz combinada apilando train y las filas de test
    correspondientes a los PIDs de test, calcula la descomposición SVD
    truncada a f factores y devuelve los componentes necesarios para
    puntuar pares (playlist, canción).

    La puntuación estimada es:
        r̂_{u,i} = ũ_u · σ · ṽ_iᵀ  ≡  scores_u = U_f[u] @ diag(Σ_f) @ V_fᵀ

    Args:
        train_matrix: Matriz dispersa de training (n_train × n_songs)
        test_matrix:  Matriz dispersa de test    (n_test × n_songs)
        pids:         PIDs de las playlists de test (índices de fila en test)
        f:            Número de factores latentes a conservar

    Returns:
        Diccionario con claves:
            U        (n_total_rows × f)  — vectores latentes de playlists
            sigma    (f,)                — valores singulares
            Vt       (f × n_songs)       — vectores latentes de canciones
            pid_to_row  dict pid → fila en U para playlists de test
    """
    # Construir matriz combinada: train arriba, filas de test abajo
    test_rows = test_matrix[pids]
    combined  = vstack([train_matrix, test_rows], format="csr")

    n_train = train_matrix.shape[0]
    pid_to_row = {pid: n_train + i for i, pid in enumerate(pids)}

    print(f"  SVD conjunto: shape={combined.shape}, f={f}")
    U, sigma, Vt = svds(combined.astype(np.float32), k=f)

    # svds devuelve valores singulares en orden ASCENDENTE → invertir
    order = np.argsort(-sigma)
    U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

    return {
        "U":          U.astype(np.float32),
        "sigma":      sigma.astype(np.float32),
        "Vt":         Vt.astype(np.float32),
        "pid_to_row": pid_to_row,
    }


def compute_svd_train(
    train_matrix: csr_matrix,
    f: int = 200,
) -> Dict:
    """
    Variante B — SVD solo sobre training.

    Calcula la descomposición SVD truncada de la matriz de training.
    Las playlists de test se proyectarán al espacio latente en recommender
    usando sus seeds:
        ũ_{m+1} = r_{m+1} · Ṽ · Σ⁻¹

    Args:
        train_matrix: Matriz dispersa de training (n_train × n_songs)
        f:            Número de factores latentes a conservar

    Returns:
        Diccionario con claves:
            sigma    (f,)         — valores singulares
            Vt       (f × n_songs) — vectores latentes de canciones
    """
    print(f"  SVD training: shape={train_matrix.shape}, f={f}")
    _, sigma, Vt = svds(train_matrix.astype(np.float32), k=f)

    order = np.argsort(-sigma)
    sigma, Vt = sigma[order], Vt[order, :]

    return {
        "sigma": sigma.astype(np.float32),
        "Vt":    Vt.astype(np.float32),
    }


def save_svd(svd_data: Dict, path: str):
    """Guarda los componentes SVD en un archivo NPZ."""
    np.savez_compressed(path, **svd_data)
    print(f"SVD guardado en '{path}'")


def load_svd(path: str) -> Dict:
    """Carga los componentes SVD guardados con save_svd."""
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        # pid_to_row se guardó como array 0-d con un dict dentro
        if val.ndim == 0:
            result[key] = val.item()
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# SLIM y FISM
# ---------------------------------------------------------------------------

def compute_slim(
    train_matrix: csr_matrix,
    lambda_l1: float = 0.1,
    lambda_l2: float = 0.1,
    learning_rate: float = 0.01,
    epochs: int = 100,
    convergence_threshold: float = 1e-5,
    max_items: int = 0,
) -> Dict:
    """
    Entrena SLIM sobre la matriz de training.

    Args:
        max_items: Si > 0, restringe el entrenamiento a los top-max_items ítems
                   más populares. Necesario cuando n_items es grande (S sería n²).

    Returns:
        Diccionario con claves:
            'S'           → matriz de coeficientes (csr_matrix, shape max_items×max_items)
            'item_subset' → array de índices originales (None si no se filtró)
    """
    from slim import SLIM

    item_subset = None
    X = train_matrix

    if max_items > 0 and train_matrix.shape[1] > max_items:
        # Seleccionar los max_items ítems más populares
        popularity = np.asarray(train_matrix.sum(axis=0)).flatten()
        item_subset = np.argsort(-popularity)[:max_items].astype(np.int32)
        X = train_matrix[:, item_subset]
        print(f"  SLIM: filtrando a top-{max_items} ítems "
              f"(de {train_matrix.shape[1]} totales)")

    print(f"  SLIM: shape={X.shape}, λ₁={lambda_l1}, β={lambda_l2}, épocas={epochs}")
    slim = SLIM(lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                learning_rate=learning_rate, epochs=epochs)
    slim.fit(X, convergence_threshold=convergence_threshold)
    return {"S": slim.get_coefficients(), "item_subset": item_subset}


def compute_fism(
    train_matrix: csr_matrix,
    n_factors: int = 20,
    lambda_l1: float = 0.1,
    lambda_l2: float = 0.1,
    learning_rate: float = 0.01,
    epochs: int = 100,
    convergence_threshold: float = 1e-5,
) -> Dict:
    """
    Entrena FISM sobre la matriz de training.

    Returns:
        Diccionario con claves 'P' y 'Q' → matrices de factores (np.ndarray)
    """
    from fism import FISM

    print(f"  FISM: shape={train_matrix.shape}, f={n_factors}, λ₁={lambda_l1}, β={lambda_l2}, épocas={epochs}")
    fism = FISM(n_factors=n_factors, lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                learning_rate=learning_rate, epochs=epochs)
    fism.fit(train_matrix, convergence_threshold=convergence_threshold)
    return {"P": fism.P, "Q": fism.Q}


def save_slim_model(slim_data: Dict, path: str):
    """Guarda la matriz S de SLIM y el subconjunto de ítems en formato NPZ."""
    save_npz(path, slim_data["S"])
    # Guardar item_subset junto a la matriz S
    subset_path = path.replace(".npz", "_subset.npy")
    if slim_data.get("item_subset") is not None:
        np.save(subset_path, slim_data["item_subset"])
    else:
        # Borrar fichero anterior si ahora no hay subset
        import os
        if os.path.exists(subset_path):
            os.remove(subset_path)
    print(f"Modelo SLIM guardado en '{path}'")


def load_slim_model(path: str) -> Dict:
    """Carga la matriz S (y opcionalmente item_subset) guardada con save_slim_model."""
    import os
    subset_path = path.replace(".npz", "_subset.npy")
    item_subset = np.load(subset_path) if os.path.exists(subset_path) else None
    return {"S": load_npz(path), "item_subset": item_subset}


def save_fism_model(fism_data: Dict, path_p: str, path_q: str):
    """Guarda las matrices P y Q de FISM en formato NPZ."""
    save_npz(path_p, csr_matrix(fism_data["P"]))
    save_npz(path_q, csr_matrix(fism_data["Q"]))
    print(f"Modelo FISM guardado en '{path_p}' y '{path_q}'")


def load_fism_model(path_p: str, path_q: str) -> Dict:
    """Carga las matrices P y Q guardadas con save_fism_model."""
    return {
        "P": load_npz(path_p).toarray(),
        "Q": load_npz(path_q).toarray(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Cálculo de popularidad, vecindarios KNN y SVD.")

    parser.add_argument("--method",
        choices=["popularity", "user_knn", "item_knn", "svd_combined", "svd_train",
                 "slim", "fism"],
        default="popularity", help="Método a ejecutar")

    parser.add_argument("--K", type=int, default=500,
        help="Número máximo de vecinos a guardar (truncar con --k en recommender)")

    parser.add_argument("--f", type=int, default=200,
        help="Número de factores latentes para SVD")

    parser.add_argument("--block_size", type=int, default=256,
        help="Tamaño de bloque para el cálculo por bloques")

    parser.add_argument("--n_jobs", type=int, default=-1,
        help="Número de procesos paralelos (-1 = todos los cores)")
    parser.add_argument("--n_factors", type=int, default=20,
        help="Número de factores latentes para FISM")
    parser.add_argument("--lambda_l1", type=float, default=0.1,
        help="Regularización L1 para SLIM/FISM")
    parser.add_argument("--lambda_l2", type=float, default=0.1,
        help="Regularización L2 para SLIM/FISM")
    parser.add_argument("--learning_rate", type=float, default=0.01,
        help="Tasa de aprendizaje para SLIM/FISM")
    parser.add_argument("--epochs", type=int, default=100,
        help="Épocas máximas para SLIM/FISM")
    parser.add_argument("--max_items", type=int, default=0,
        help="Filtrar a top-N ítems más populares para SLIM (0 = sin filtro)")

    args = parser.parse_args()

    print(f"Cargando matriz de training desde '{TRAIN_NPZ_PATH}'...")
    train_matrix = load_npz(TRAIN_NPZ_PATH)
    print(f"  Shape: {train_matrix.shape}")

    if args.method == "popularity":
        print("\nCalculando ranking de popularidad...")
        with gzip.open(SONGS_PATH, "rt", encoding="utf-8") as f:
            songs = json.load(f)
        popularity_list = compute_popularity(train_matrix, songs)
        save_popularity_ranking(popularity_list, POPULARITY_PATH)
        print("\nTop 10 canciones más populares:")
        for i, (uri, count) in enumerate(popularity_list[:10], 1):
            print(f"  {i}. {uri}: {count} playlists")

    elif args.method == "user_knn":
        from recommender import get_playlist_seeds

        print(f"\nCargando matriz de test desde '{TEST_NPZ_PATH}'...")
        test_matrix = load_npz(TEST_NPZ_PATH)
        print(f"  Shape: {test_matrix.shape}")

        print(f"Obteniendo PIDs de test desde '{TEST_ZIP_PATH}'...")
        playlist_seeds = get_playlist_seeds(TEST_ZIP_PATH)
        pids = list(playlist_seeds.keys())
        print(f"  {len(pids)} playlists de test")

        print("\nComputando vecindarios user-based...")
        neighborhoods = compute_user_neighborhoods(
            train_matrix, test_matrix, pids,
            K=args.K, block_size=args.block_size, n_jobs=args.n_jobs,
        )
        save_neighborhoods(neighborhoods, USER_NEIGHBORHOODS_PATH)

    elif args.method == "item_knn":
        print("\nComputando vecindarios item-based...")
        neighborhoods = compute_item_neighborhoods(
            train_matrix,
            K=args.K, block_size=args.block_size, n_jobs=args.n_jobs,
        )
        save_neighborhoods(neighborhoods, ITEM_NEIGHBORHOODS_PATH)

    elif args.method == "svd_combined":
        from recommender import get_playlist_seeds

        print(f"\nCargando matriz de test desde '{TEST_NPZ_PATH}'...")
        test_matrix = load_npz(TEST_NPZ_PATH)

        print(f"Obteniendo PIDs de test...")
        playlist_seeds = get_playlist_seeds(TEST_ZIP_PATH)
        pids = list(playlist_seeds.keys())

        print(f"\nComputando SVD conjunto (f={args.f})...")
        svd_data = compute_svd_combined(train_matrix, test_matrix, pids, f=args.f)
        save_svd(svd_data, SVD_COMBINED_PATH)

    elif args.method == "svd_train":
        print(f"\nComputando SVD training (f={args.f})...")
        svd_data = compute_svd_train(train_matrix, f=args.f)
        save_svd(svd_data, SVD_TRAIN_PATH)

    elif args.method == "slim":
        import os
        os.makedirs(os.path.dirname(SLIM_MODEL_PATH), exist_ok=True)
        print(f"\nEntrenando SLIM (λ₁={args.lambda_l1}, β={args.lambda_l2})...")
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
        import os
        os.makedirs(os.path.dirname(FISM_P_PATH), exist_ok=True)
        print(f"\nEntrenando FISM (f={args.n_factors}, λ₁={args.lambda_l1}, β={args.lambda_l2})...")
        fism_data = compute_fism(
            train_matrix,
            n_factors=args.n_factors,
            lambda_l1=args.lambda_l1,
            lambda_l2=args.lambda_l2,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
        )
        save_fism_model(fism_data, FISM_P_PATH, FISM_Q_PATH)