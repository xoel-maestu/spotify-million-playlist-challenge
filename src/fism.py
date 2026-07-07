#!/usr/bin/env python3
"""
Factored Item Similarity Model (FISM) [Kabbur et al., 2013]

Extiende SLIM descomponiendo S en dos matrices de menor rango:

    S = PQᵀ    P, Q ∈ ℝ^{n_items × f}

El problema de optimización resultante es:

    min_{P,Q}  ‖X − XPQᵀ‖²_F  +  λ(‖P‖₁ + ‖Q‖₁)  +  β(‖P‖²_F + ‖Q‖²_F)

Reducir el rango limita la capacidad del modelo (regularización implícita)
y hace la inferencia más eficiente: score(u,i) = (Σ_{j∈seeds} P_j) · Q_iᵀ
sin materializar S.
"""

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm

from slim import _converged


class FISM:

    def __init__(self, n_factors=20, lambda_l1=0.1, lambda_l2=0.1,
                 learning_rate=0.01, epochs=100, verbose=True):
        self.n_factors     = n_factors
        self.lambda_l1     = lambda_l1
        self.lambda_l2     = lambda_l2
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        self.P = None   # (n_items, n_factors)
        self.Q = None   # (n_items, n_factors)

    def fit(self, X, convergence_threshold=1e-5):
        """
        Entrena FISM usando diferenciación automática de TensorFlow.

        Args:
            X:                     Matriz de interacción usuario-ítem (csr_matrix)
            convergence_threshold: Para cuando la mejora relativa < umbral

        Returns:
            self
        """
        n_items = X.shape[1]

        coo = X.tocoo()
        X_sparse = tf.sparse.reorder(tf.SparseTensor(
            indices=np.column_stack((coo.row, coo.col)).astype(np.int64),
            values=tf.cast(coo.data, tf.float32),
            dense_shape=coo.shape,
        ))
        init = tf.keras.initializers.GlorotUniform()
        P = tf.Variable(init((n_items, self.n_factors)), trainable=True, dtype=tf.float32)
        Q = tf.Variable(init((n_items, self.n_factors)), trainable=True, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        prev_loss = float("inf")
        iterator  = tqdm(range(self.epochs), disable=not self.verbose, desc="FISM")

        # Precompute user and item index arrays for observed entries
        u_obs = X_sparse.indices[:, 0]
        i_obs = X_sparse.indices[:, 1]
        targets = X_sparse.values  # interaction values (typically 1.0)

        for step in iterator:
            with tf.GradientTape() as tape:
                # U_factors[u] = sum_{j in seeds_u} P[j]  for all users
                U_factors = tf.sparse.sparse_dense_matmul(X_sparse, P)  # (n_users, n_factors)

                # Para predecir r_{u,i}, excluir el propio ítem i del vector de usuario
                # (equivalente a diag(S)=0 en SLIM; evita que un ítem se prediga a sí mismo)
                U_obs = tf.gather(U_factors, u_obs) - tf.gather(P, i_obs)  # (nnz, n_factors)
                Q_obs = tf.gather(Q, i_obs)                                 # (nnz, n_factors)

                # Predicted score for each observed interaction
                pred = tf.reduce_sum(U_obs * Q_obs, axis=1)  # (nnz,)

                # Loss on observed entries only (avoids materializing dense matrices)
                loss_obs = tf.reduce_sum(tf.square(targets - pred))
                reg = (self.lambda_l1 * (tf.reduce_sum(tf.abs(P)) + tf.reduce_sum(tf.abs(Q)))
                       + self.lambda_l2 * (tf.reduce_sum(tf.square(P)) + tf.reduce_sum(tf.square(Q))))
                loss = loss_obs + reg

            optimizer.apply_gradients(zip(tape.gradient(loss, [P, Q]), [P, Q]))

            loss_val = float(loss)

            if self.verbose and (step + 1) % 100 == 0:
                iterator.set_postfix({"loss": f"{loss_val:.6f}"})

            if _converged(prev_loss, loss_val, convergence_threshold):
                if self.verbose:
                    print(f"\n  Convergencia en paso {step + 1}")
                break

            prev_loss = loss_val

        self.P = P.numpy()
        self.Q = Q.numpy()
        return self

    def save(self, path_p: str, path_q: str):
        if self.P is None:
            raise ValueError("Modelo no entrenado")
        save_npz(path_p, csr_matrix(self.P))
        save_npz(path_q, csr_matrix(self.Q))

    @classmethod
    def load(cls, path_p: str, path_q: str):
        inst   = cls.__new__(cls)
        inst.P = load_npz(path_p).toarray()
        inst.Q = load_npz(path_q).toarray()
        return inst
