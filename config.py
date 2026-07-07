#!/usr/bin/env python3
"""
Configuración estructural del proyecto.

Contiene rutas de datos, resultados e identidad del equipo.
Los hiperparámetros (K, k, f, block_size, n_jobs, num_recommendations)
se pasan como argumentos de línea de comandos en cada script.
"""

# ---------------------------------------------------------------------------
# Rutas de datos — dataset completo
# ---------------------------------------------------------------------------

TRAIN_ZIP_PATH  = "./dataset/raw/spotify_train_dataset.zip"
TEST_ZIP_PATH   = "./dataset/raw/spotify_test_playlists.zip"
TRAIN_NPZ_PATH  = "./dataset/train_matrix.npz"
TEST_NPZ_PATH   = "./dataset/test_matrix.npz"
SONGS_PATH      = "./dataset/track_mapping.json.gz"

# Nombres de los JSON dentro del ZIP de test (dataset completo)
TEST_INPUT_JSON = "test_input_playlists.json"
TEST_EVAL_JSON  = "test_eval_playlists.json"

# ---------------------------------------------------------------------------
# Rutas de datos — dataset reducido (trimmed)
# ---------------------------------------------------------------------------

TRIMMED_ZIP_PATH      = "./dataset/raw/trimmed_dataset.zip"
TRIMMED_NPZ_PATH      = "./dataset/train_matrix_trimmed.npz"
TRIMMED_TEST_NPZ_PATH = "./dataset/test_matrix_trimmed.npz"
TRIMMED_SONGS_PATH    = "./dataset/track_mapping_trimmed.json.gz"

# Nombres de los JSON dentro del ZIP trimmed
TRIMMED_INPUT_JSON = "test_input_trimmed.json"
TRIMMED_EVAL_JSON  = "test_eval_trimmed.json"

# ---------------------------------------------------------------------------
# Rutas de datos — dataset reducido con vocabulario compacto
# Matrices construidas solo con trimmed_dataset.zip, sin vocabulario heredado
# del dataset completo. Necesario para que SLIM/FISM sean viables (n_items real).
# ---------------------------------------------------------------------------

COMPACT_NPZ_PATH      = "./dataset/train_matrix_compact.npz"
COMPACT_TEST_NPZ_PATH = "./dataset/test_matrix_compact.npz"
COMPACT_SONGS_PATH    = "./dataset/track_mapping_compact.json.gz"

# ---------------------------------------------------------------------------
# Rutas de resultados
# ---------------------------------------------------------------------------

POPULARITY_PATH         = "./results/popularity_ranking.json.gz"
USER_NEIGHBORHOODS_PATH = "./results/knn/user_neighborhoods.npz"
ITEM_NEIGHBORHOODS_PATH = "./results/knn/item_neighborhoods.npz"
SVD_COMBINED_PATH       = "./results/svd/svd_combined.npz"   # SVD sobre train+test
SVD_TRAIN_PATH          = "./results/svd/svd_train.npz"      # SVD solo sobre train
SLIM_MODEL_PATH         = "./results/slim/slim_model.npz"
FISM_P_PATH             = "./results/fism/fism_p.npz"
FISM_Q_PATH             = "./results/fism/fism_q.npz"
SUBMISSION_PATH         = "./results/submission.csv.gz"

# ---------------------------------------------------------------------------
# Identidad del equipo (cabecera team_info de la submission del challenge)
# ---------------------------------------------------------------------------

TEAM_NAME     = "EXP"
CONTACT_EMAIL = "xoel.maestu@gmail.com"
