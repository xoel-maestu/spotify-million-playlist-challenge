# Spotify Million Playlist Challenge

Automatic playlist continuation with classic collaborative filtering, built from scratch: KNN, PureSVD, SLIM and FISM over sparse matrices, with a resumable four-stage pipeline.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The [Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) (ACM RecSys Challenge 2018) poses a playlist continuation task: given the first tracks of a playlist, produce a ranked list of 500 recommended tracks. The dataset is the largest public collection of music playlists in the world: 1,000,000 playlists containing 2,262,292 unique tracks by roughly 300,000 artists.

This project implements six recommenders for that task, all from first principles on top of SciPy sparse matrices, with the two model-based methods (SLIM, FISM) trained via TensorFlow automatic differentiation:

- **Popularity** baseline (also the fallback for every other method)
- **User-based KNN** and **item-based KNN** with cosine similarity
- **PureSVD** in two variants: joint factorization, and train-only factorization with fold-in projection
- **SLIM** (Sparse Linear Methods) and **FISM** (Factored Item Similarity Models)

## Results

Evaluation follows the official challenge metrics: **R-Precision** (fraction of held-out tracks recovered in the first |G| positions), **NDCG** (rank-sensitive gain) and **Clicks** (number of 10-track pages until the first hit; lower is better).

| Model | R-Precision | NDCG | Clicks | Runtime |
|---|---|---|---|---|
| Popularity | 0.0257 | 0.0904 | 17.31 | ~5 min |
| User-KNN (k=500) | **0.1596** | 0.3460 | 4.49 | ~6 min |
| Item-KNN (k=500) | 0.1518 | 0.3287 | 5.75 | ~20 min |
| PureSVD, joint (f=100) | 0.1262 | 0.2825 | 7.49 | ~10 min |
| PureSVD, fold-in (f=100) | 0.1272 | 0.2876 | 5.77 | ~9 min |
| SLIM (λ₁=0.001, β=1.0, 100 epochs) | 0.1379 | **0.4031** | **3.45** | ~5 s |
| FISM (f=100, λ₁=0.1, β=0.001, 200 epochs) | 0.0345 | 0.2833 | 4.14 | ~7 s |

An honest note on protocols: the first five models were evaluated on 10,000 test playlists against the full dataset. SLIM and FISM train an item-item matrix that is quadratic in the vocabulary size, so they were evaluated on a small trimmed subset (29 playlists, of which 20 are cold-start with zero seed tracks and fall back to popularity). Their rows are not directly comparable with the rest, but within that subset SLIM still achieves the best NDCG and Clicks of everything we ran. Runtimes measured on a laptop with a GTX 1650 Ti (4 GB VRAM).

## Architecture

A four-stage pipeline, orchestrated by `pipeline.py`. Every stage writes its output to disk and can be skipped with a `--skip_*` flag when its artifact already exists, so expensive steps (neighborhoods, SVD, training) are computed once and reused across experiments.

```
reader ──► compute ──► recommender ──► evaluate
  .npz       model        ranked         R-Precision,
matrices   artifacts    submissions      NDCG, Clicks
```

1. **`src/reader.py`** parses the playlist JSONs inside the dataset zips and builds two CSR sparse matrices (playlists × tracks, binary), sharing a single `track_uri → column` mapping so train and test indices are consistent.
2. **`src/compute.py`** precomputes the model: the popularity ranking (always, as fallback), KNN neighborhoods, the truncated SVD, or the trained SLIM/FISM factors, depending on the method.
3. **`src/recommender.py`** scores and ranks 500 tracks per test playlist, excluding seeds, with popularity fallback for cold-start playlists, and writes a challenge-format submission (`team_info` header included).
4. **`src/evaluate.py`** computes R-Precision, NDCG and Clicks against the held-out ground truth.

All paths and structural constants live in `config.py`; hyperparameters are command-line arguments.

## Models

**User-based KNN.** Neighborhoods are precomputed with cosine similarity, comparing each test playlist against all training playlists in blocks. The score of a track is the similarity-weighted count over the k nearest neighbors that contain it:

$$\hat{r}_{u,i} = \sum_{v \in V_u} s_{u,v} \cdot r_{v,i}$$

**Item-based KNN.** Each track is compared against all others (2.26M × 2.26M, computed blockwise). Since playlist entries are binary, the score reduces to a sum of similarities restricted to the seeds:

$$\hat{r}_{u,i} = \sum_{j \in J_i \cap \text{seeds}_u} s_{i,j}$$

**PureSVD.** Truncated decomposition $R \approx \tilde{U} \tilde{\Sigma} \tilde{V}^T$ via `scipy.sparse.linalg.svds`, in two variants: *joint* (train and test stacked and factorized together, so test playlists get latent vectors directly) and *fold-in* (train-only factorization; test playlists are projected as $\tilde{u} = r \tilde{V} \tilde{\Sigma}^{-1}$ at recommendation time).

**SLIM.** Learns a sparse non-negative item-item coefficient matrix $S$ by minimizing

$$\min_S \|X - XS\|_F^2 + \lambda\|S\|_1 + \beta\|S\|_F^2 \quad \text{s.t.} \quad S \geq 0,\ \mathrm{diag}(S) = 0$$

where the diagonal constraint prevents an item from predicting itself. Optimized with Adam and projected gradient (clip to non-negative, re-zero the diagonal) in TensorFlow.

**FISM.** Factorizes the item-item matrix as $S = PQ^T$ with $P, Q \in \mathbb{R}^{n \times f}$, which drops the quadratic parameter count and scores as

$$\hat{r}_{u,i} = \Big(\sum_{j \in \text{seeds}_u \setminus \{i\}} P_j\Big) \cdot Q_i^T$$

## Engineering notes

The dataset's scale made memory the central engineering problem; the interesting fixes:

- **Blockwise KNN.** Materializing a full similarity matrix is impossible at 2.26M tracks. Similarities are computed in blocks (`--block_size`), parallelized across cores with joblib, and only the top K=500 neighbors per row are kept, stored per id as index/value arrays in a compressed NPZ. Precomputing with a large K means `k` can then be tuned at inference time without recomputing anything.
- **SLIM's diagonal without a mask.** The first implementation enforced $\mathrm{diag}(S)=0$ with a dense n×n mask: at 64,052 items that is 64,052² floats ≈ 16 GB, an instant OOM. `tf.linalg.set_diag` re-zeroes the diagonal in-place with no auxiliary matrix.
- **Loss over observed entries only.** Computing the full residual $X - XS$ densifies X. Both SLIM and FISM evaluate the loss only on observed entries; a naive dense formulation of FISM's intermediates at full vocabulary attempted a 16.5 TiB allocation before the rewrite.
- **Compact vocabulary.** Even with those fixes, S is quadratic in the vocabulary. For SLIM/FISM the matrices are rebuilt with a vocabulary drawn only from the reduced dataset (1,704 items instead of the 64,052 inherited from the full corpus), which brings S down to ~11 MB and training to seconds on a 4 GB GPU. A `--max_items` flag additionally caps the vocabulary to the N most popular tracks as a guard for larger corpora.
- **Initialization under a non-negativity projection.** SLIM projects onto $S \geq 0$ after every step, so a symmetric (Glorot) initialization would lose half its weights on the first projection. Weights start at uniform[0, 0.01] instead.

## Getting started

Requires Python ≥ 3.12. TensorFlow is only needed for SLIM/FISM; GPU support is optional but recommended.

```bash
pip install -r requirements.txt
```

### Dataset

The Million Playlist Dataset is available from [AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) after registering (free, research use). The pipeline expects it packaged as zips of playlist JSONs under `dataset/raw/`: a training zip, and a test zip containing seed playlists plus held-out ground truth for local evaluation:

```
dataset/raw/
├── spotify_train_dataset.zip      # training playlists (JSON slices)
├── spotify_test_playlists.zip     # test_input_playlists.json + test_eval_playlists.json
└── trimmed_dataset.zip            # optional reduced subset for fast experiments (--trimmed)
```

File names and the internal JSON names are constants in `config.py`; adjust them there to match your local packaging.

### Running

```bash
# Popularity baseline (builds the sparse matrices on first run)
python pipeline.py --method popularity

# User-KNN / Item-KNN
python pipeline.py --method user_knn --K 500 --k 500 --n_jobs -1 --skip_reader
python pipeline.py --method item_knn --K 500 --k 500 --n_jobs -1 --skip_reader --skip_popularity

# PureSVD (joint, or train-only + fold-in)
python pipeline.py --method svd_combined --f 100 --skip_reader --skip_popularity
python pipeline.py --method svd_train    --f 100 --skip_reader --skip_popularity

# SLIM and FISM (on the reduced dataset with compact vocabulary)
python pipeline.py --method slim --trimmed --lambda_l1 0.001 --lambda_l2 1.0 --epochs 100
python pipeline.py --method fism --trimmed --n_factors 100 --lambda_l1 0.1 --lambda_l2 0.001 --epochs 200 --skip_reader --skip_popularity
```

Every script accepts `--help`. Once a stage's artifact exists, skip it (`--skip_reader`, `--skip_popularity`, `--skip_neighborhoods`, `--skip_svd`, `--skip_slim`, `--skip_recommender`) to iterate on later stages in seconds.

### Hyperparameter search

```bash
# k for KNN: ternary search (assumes unimodal curve), log grid, or explicit values
python src/optimize_k.py --method user_knn --strategy ternary --metric NDCG

# f for PureSVD
python src/optimize_f.py --method svd_combined --metric NDCG

# Full grid sweep for SLIM and FISM (grids defined at the top of sweep.py)
python sweep.py all
```

Search results land in `results/` as CSVs; neighborhoods and matrices are reused across evaluations, so each k probe costs about a minute.

## Project structure

```
.
├── config.py             # Paths and structural constants
├── pipeline.py           # End-to-end runner for any method
├── sweep.py              # SLIM/FISM hyperparameter grid sweep
└── src/
    ├── reader.py         # Zips of playlist JSONs → CSR matrices (.npz)
    ├── compute.py        # Popularity, KNN neighborhoods, SVD, SLIM/FISM training
    ├── recommender.py    # Scoring and ranked submission generation
    ├── evaluate.py       # R-Precision, NDCG, Clicks
    ├── slim.py           # SLIM model (TensorFlow)
    ├── fism.py           # FISM model (TensorFlow)
    ├── optimize_k.py     # k search for KNN (ternary / grid / explicit)
    └── optimize_f.py     # f search for SVD
```

`dataset/` and `results/` are generated locally and git-ignored.

## Credits

This was a team effort by team **EXP**, built together with [Pablo Fernández (@appuchias)](https://github.com/appuchias) and [Enrique Pardo García (@EnriquePardoGarcia)](https://github.com/EnriquePardoGarcia). My part was the recommendation models and the pipeline: the popularity baseline and evaluator, both KNN recommenders, the two PureSVD variants, the TensorFlow SLIM/FISM training with its memory optimizations, and the end-to-end orchestration. Pablo built the dataset reader and most of the project infrastructure; Enrique contributed the first SLIM prototype and the sweep testing.

## References

- [ACM RecSys Challenge 2018](https://www.recsyschallenge.com/2018/) and the [Million Playlist Dataset on AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- H. Zamani et al. (2019). [*An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation*](https://arxiv.org/abs/1810.01520). ACM TIST.
- P. Cremonesi, Y. Koren, R. Turrin (2010). *Performance of Recommender Algorithms on Top-N Recommendation Tasks* (PureSVD). RecSys.
- X. Ning, G. Karypis (2011). *SLIM: Sparse Linear Methods for Top-N Recommender Systems*. ICDM.
- S. Kabbur, X. Ning, G. Karypis (2013). *FISM: Factored Item Similarity Models for Top-N Recommender Systems*. KDD.

## License

[MIT](LICENSE)
