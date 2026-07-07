#!/usr/bin/env python3

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *

import gzip, json, os, zipfile

from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm


def playlists_to_csr(content, songs) -> tuple[list, list]:
    """Convierte una JSON que contenga info de playlists a una matriz dispersa"""

    rows, cols = list(), list()

    for playlist in content["playlists"]:
        pid = playlist["pid"]

        for song in playlist["tracks"]:
            song_id = song["track_uri"]

            # Si ya hemos visto la canción usamos la columna asignada
            # Si no, la añadimos al diccionario con el índice de la
            # columna correspondiente.
            if (song_col := songs.get(song_id)) is None:
                song_col = len(songs)
                songs[song_id] = song_col

            # Guardamos un 1 en la fila-columna que corresponde.
            rows.append(pid)
            cols.append(song_col)

    return rows, cols


def train_a_npz(zip_path: str, npz_path: str, songs: dict) -> dict:
    """Convierte los archivos contenidos en un zip a una matriz dispersa en npz"""

    rows, cols = list(), list()

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for file in tqdm(zipf.namelist(), desc="Archivos de train"):
            if file.endswith(".json"):
                with zipf.open(file) as f:
                    file_rows, file_cols = playlists_to_csr(json.load(f), songs)

                    rows.extend(file_rows)
                    cols.extend(file_cols)

    assert len(rows) == len(
        cols
    ), f"El número de filas ({len(rows)}) y columnas ({len(cols)}) debe coincidir."

    values = [1] * len(rows)

    # Construir la matriz compriida por filas
    matrix = csr_matrix((values, (rows, cols)), shape=(max(rows) + 1, len(songs)))

    # Guardar a NPZ
    save_npz(npz_path, matrix, compressed=False)
    print(f"Matriz CSR creada y guardada en '{npz_path}'.")

    return songs


def test_a_npz(test_path: str, npz_path: str, songs: dict,
               input_json: str = TEST_INPUT_JSON) -> dict:
    """Convierte el archivo de input de test a una matriz dispersa en npz"""

    rows, cols = list(), list()

    with zipfile.ZipFile(test_path, "r") as zipf:
        with zipf.open(input_json) as f:
            file_rows, file_cols = playlists_to_csr(json.load(f), songs)

            rows.extend(file_rows)
            cols.extend(file_cols)

    assert len(rows) == len(
        cols
    ), f"El número de filas ({len(rows)}) y columnas ({len(cols)}) debe coincidir."

    values = [1] * len(rows)

    # Construir la matriz compriida por filas
    matrix = csr_matrix((values, (rows, cols)), shape=(max(rows) + 1, len(songs)))

    # Guardar a NPZ
    save_npz(npz_path, matrix, compressed=False)
    print(f"Matriz CSR creada y guardada en '{npz_path}'.")

    return songs


if __name__ == "__main__":

    # Cargar la correspondencia de canciones a columnas si existe
    if os.path.exists(SONGS_PATH):
        with gzip.open(SONGS_PATH, "rt", encoding="utf-8") as f:
            songs = json.load(f)
    else:
        songs = dict()

    # Obtenemos la matriz de training
    songs = train_a_npz(TRAIN_ZIP_PATH, TRAIN_NPZ_PATH, songs)

    # Obtenemos también la matriz de playlists de test
    songs = test_a_npz(TEST_ZIP_PATH, TEST_NPZ_PATH, songs)

    # Guardar la relación de columnas a canciones
    with gzip.open(SONGS_PATH, "wt", encoding="utf-8") as f:
        json.dump(songs, f)
    print(
        f"Guardada la correspondencia de canciones a columnas de la matriz en '{SONGS_PATH}'"
    )
