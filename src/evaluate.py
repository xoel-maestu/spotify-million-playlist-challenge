#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *

import gzip, json, zipfile
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_submission(submission_path: str) -> Dict[int, List[str]]:
    """
    Carga las recomendaciones desde un archivo .csv.gz con el formato oficial.

    Formato esperado:
        team_info,<team_name>,<contact_email>
        <pid>,<track_uri_1>,<track_uri_2>,...,<track_uri_500>
        ...
    """

    recommendations: Dict[int, List[str]] = {}

    with gzip.open(submission_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("team_info"):
                continue
            parts = line.split(",")
            pid = int(parts[0])
            tracks = parts[1:]
            recommendations[pid] = tracks

    return recommendations


def load_ground_truth(test_zip_path: str,
                      eval_json: str = TEST_EVAL_JSON) -> Dict[int, List[str]]:
    """
    Carga el ground truth desde el ZIP de test.

    Args:
        test_zip_path: Ruta al ZIP de test.
        eval_json:     Nombre del fichero JSON dentro del ZIP
                       (por defecto TEST_EVAL_JSON de config).
    """

    ground_truth = {}

    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        with zipf.open(eval_json) as f:
            data = json.load(f)
            for pl in data["playlists"]:
                ground_truth[pl["pid"]] = [t["track_uri"] for t in pl["tracks"]]

    return ground_truth


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def r_precision(recommended: List[str], relevant: List[str]) -> float:
    """
    R-Precision.

    Definición teórica:
        Dada G el conjunto de canciones relevantes (ground truth) y R la lista
        ordenada de canciones recomendadas, R-Precision se define como:

            R-Precision = |G ∩ R_{1:|G|}| / |G|

        donde R_{1:|G|} son los primeros |G| elementos de R.

        Es decir, se trunca la lista de recomendaciones al mismo número de
        elementos que hay en el ground truth, y se mide qué fracción de esas
        canciones relevantes han sido recuperadas. No tiene en cuenta el orden
        dentro de esos primeros |G| puestos.

    Args:
        recommended: Lista ordenada de track_uris recomendados.
        relevant: Lista de track_uris del ground truth (canciones ocultas).

    Returns:
        Valor en [0, 1]. Devuelve 0 si no hay canciones relevantes.
    """

    r = len(relevant)
    if r == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for t in recommended[:r] if t in relevant_set)

    return hits / r


def ndcg(recommended: List[str], relevant: List[str]) -> float:
    """
    NDCG (Normalized Discounted Cumulative Gain).

    Definición teórica:
        Con relevancia binaria (rel_i = 1 si R_i ∈ G, 0 si no), el DCG
        acumula las ganancias de cada acierto penalizadas por su posición:

            DCG = Σ_{i=1}^{|R|}  rel_i / log2(i + 1)

        El DCG ideal (IDCG) asume que los |G| primeros puestos son todos
        aciertos, obteniendo así la puntuación máxima alcanzable:

            IDCG = Σ_{i=1}^{|G|}  1 / log2(i + 1)

        El NDCG normaliza el DCG entre 0 y 1:

            NDCG = DCG / IDCG

        A diferencia de R-Precision, sí penaliza los aciertos que aparecen
        en posiciones tardías de la lista.

    Args:
        recommended: Lista ordenada de track_uris recomendados.
        relevant: Lista de track_uris del ground truth (canciones ocultas).

    Returns:
        Valor en [0, 1]. Devuelve 0 si no hay canciones relevantes.
    """

    if not relevant:
        return 0.0

    relevant_set = set(relevant)

    # DCG
    dcg = 0.0
    for i, track in enumerate(recommended, start=1):
        if track in relevant_set:
            dcg += 1.0 / np.log2(i + 1)

    # IDCG: colocamos todos los relevantes al principio
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, len(relevant) + 1))

    if idcg > 0:
        return dcg / idcg

    return 0


def recommended_songs_clicks(recommended: List[str], relevant: List[str]) -> float:
    """
    Recommended Songs Clicks.

    Definición teórica:
        Simula la feature "Recommended Songs" de Spotify, que presenta las
        canciones en grupos de 10. La métrica mide cuántas veces el usuario
        tiene que solicitar un nuevo grupo (cuántos "clics") hasta encontrar
        la primera canción relevante:

            clicks = floor( (pos_{1er acierto} - 1) / 10 )

        donde pos_{1er acierto} es la posición (1-indexed) de la primera
        canción de R que pertenece a G.

        Ejemplos:
          - Primer acierto en posición 1–10  → 0 clicks (está en la 1ª página)
          - Primer acierto en posición 11–20 → 1 click
          - Primer acierto en posición 21–30 → 2 clicks
          - Sin ningún acierto               → 51 (penalización máxima,
                                                    1 más que el máximo posible)

    Args:
        recommended: Lista ordenada de track_uris recomendados.
        relevant: Lista de track_uris del ground truth (canciones ocultas).

    Returns:
        Número entero de clicks (float por compatibilidad con np.mean).
        Devuelve 51.0, un valor fuera del rango de clicks posibles,
        si no hay ningún acierto en la lista.
    """

    relevant_set = set(relevant)

    for i, track in enumerate(recommended, start=1):
        if track in relevant_set:
            return (i - 1) // 10

    return 51.0  # penalización máxima si no hay ningún acierto


# ---------------------------------------------------------------------------
# Evaluación global
# ---------------------------------------------------------------------------

def evaluate(
    recommendations: Dict[int, List[str]],
    ground_truth: Dict[int, List[str]],
) -> Dict[str, float]:
    """
    Calcula las métricas agregadas sobre todas las playlists.

    Devuelve un diccionario con las medias de R-Precision, NDCG y Clicks.
    """

    rp_scores, ndcg_scores, clicks_scores = [], [], []

    pids_evaluated = 0
    pids_missing = 0

    for pid, relevant in ground_truth.items():
        if pid not in recommendations:
            pids_missing += 1
            # Penalizar con 0 en precisión/NDCG y máximo en clicks
            rp_scores.append(0.0)
            ndcg_scores.append(0.0)
            clicks_scores.append(51.0)
            continue

        rec = recommendations[pid]
        rp_scores.append(r_precision(rec, relevant))
        ndcg_scores.append(ndcg(rec, relevant))
        clicks_scores.append(recommended_songs_clicks(rec, relevant))
        pids_evaluated += 1

    if pids_missing:
        print(f"Aviso: {pids_missing} playlists del ground truth no tienen recomendaciones.")

    return {
        "R-Precision": float(np.mean(rp_scores)),
        "NDCG": float(np.mean(ndcg_scores)),
        "Clicks": float(np.mean(clicks_scores)),
        "num_playlists": pids_evaluated + pids_missing,
        "num_evaluated": pids_evaluated,
    }

if __name__ == "__main__":

    print("Cargando recomendaciones...")
    recommendations = load_submission(SUBMISSION_PATH)
    print(f"  {len(recommendations)} playlists con recomendaciones cargadas.")
    print()

    print("Cargando ground truth...")
    ground_truth = load_ground_truth(TEST_ZIP_PATH)
    print(f"  {len(ground_truth)} playlists en el ground truth.")
    print()

    print("Evaluando...")
    results = evaluate(recommendations, ground_truth)
    print()

    print("=" * 40)
    print("  RESULTADOS")
    print("=" * 40)
    print(f"  R-Precision : {results['R-Precision']:.6f}")
    print(f"  NDCG        : {results['NDCG']:.6f}")
    print(f"  Clicks      : {results['Clicks']:.4f}")
    print(f"  Playlists   : {results['num_evaluated']} / {results['num_playlists']}")
    print("=" * 40)
