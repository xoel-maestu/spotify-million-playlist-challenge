#!/usr/bin/env python3
"""
Sparse Linear Methods (SLIM) [Ning & Karypis, 2012]

Aprende una matriz de coeficientes dispersa S resolviendo:

    min_S  ‖X − XS‖²_F  +  λ‖S‖₁  +  β‖S‖²_F
    s.t.   S ≥ 0,  diag(S) = 0

La restricción diagonal evita que un ítem se prediga a sí mismo
(lo que trivialmente minimizaría el error sin aprender nada útil).
La no-negatividad garantiza que las contribuciones sean aditivas.

ADVERTENCIA: S es densa (n_items × n_items). Solo es viable cuando n_items
es pequeño (< ~8 000 para la GTX 1650 Ti / 2 601 MB VRAM). Para catálogos
grandes filtrar los ítems más populares con el parámetro max_items en
compute_slim, o usar FISM en su lugar.
"""

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm


def _converged(prev_loss: float, curr_loss: float, threshold: float) -> bool:
    """True si la mejora relativa entre épocas es positiva pero inferior al umbral."""
    return prev_loss > 0 and 0.0 <= (prev_loss - curr_loss) / prev_loss < threshold


class SLIM:

    def __init__(self, lambda_l1=0.1, lambda_l2=0.1, learning_rate=0.01,
                 epochs=100, verbose=True):
        self.lambda_l1     = lambda_l1
        self.lambda_l2     = lambda_l2
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        self.S             = None   # Matriz de coeficientes aprendida (csr_matrix)

    def fit(self, X, convergence_threshold=1e-5):
        """
        Entrena SLIM usando diferenciación automática de TensorFlow.

        En cada iteración:
          1. Aplica diag(S)=0 con tf.linalg.set_diag — sin materializar máscara n×n
          2. Calcula XS (n_users × n_items) y la norma de Frobenius completa ‖X−XS‖²_F
          3. Evalúa la función de pérdida con GradientTape
          4. Aplica el gradiente (Adam)
          5. Proyecta S al cono no-negativo manteniendo diag=0

        Args:
            X:                     Matriz de interacción usuario-ítem (csr_matrix)
            convergence_threshold: Para cuando la mejora relativa < umbral

        Returns:
            self
        """
        n_items = X.shape[1]

        # X como SparseTensor ordenado (requisito de TF para operaciones dispersas)
        coo = X.tocoo()
        X_sparse = tf.sparse.reorder(tf.SparseTensor(
            indices=np.column_stack((coo.row, coo.col)).astype(np.int64),
            values=tf.cast(coo.data, tf.float32),
            dense_shape=coo.shape,
        ))

        # X densa para calcular ‖X − XS‖²_F — viable porque con las matrices de
        # vocabulario compacto n_items es pequeño (n_users × n_items × 4B ≈ 30 MB)
        X_dense = tf.sparse.to_dense(X_sparse)

        # Inicialización con valores pequeños no negativos — evita que la
        # primera proyección max(·,0) descarte ~50% de la inicialización
        S = tf.Variable(
            tf.random.uniform((n_items, n_items), minval=0.0, maxval=0.01),
            trainable=True, dtype=tf.float32,
        )

        optimizer   = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        prev_loss   = float("inf")
        iterator    = tqdm(range(self.epochs), disable=not self.verbose, desc="SLIM")
        zeros_diag  = tf.zeros(n_items, dtype=tf.float32)  # reutilizado cada paso

        for step in iterator:
            with tf.GradientTape() as tape:
                # diag(S) = 0 sin materializar la máscara densa (n_items × n_items)
                S_proj = tf.linalg.set_diag(S, zeros_diag)

                # Norma de Frobenius completa: ‖X − XS‖²_F
                # Incluye tanto entradas observadas (X=1) como no observadas (X=0)
                XS = tf.sparse.sparse_dense_matmul(X_sparse, S_proj)

                # Pérdida: ‖X − XS‖²_F + λ‖S‖₁ + β‖S‖²_F
                loss = (tf.reduce_sum(tf.square(X_dense - XS))
                        + self.lambda_l1 * tf.reduce_sum(tf.abs(S_proj))
                        + self.lambda_l2 * tf.reduce_sum(tf.square(S_proj)))

            optimizer.apply_gradients(zip(tape.gradient(loss, [S]), [S]))

            # Proyección: S ≥ 0 y diag(S) = 0
            S.assign(tf.maximum(tf.linalg.set_diag(S, zeros_diag), 0.0))

            loss_val = float(loss)

            if self.verbose and (step + 1) % 10 == 0:
                iterator.set_postfix({"loss": f"{loss_val:.6f}"})

            if _converged(prev_loss, loss_val, convergence_threshold):
                if self.verbose:
                    print(f"\n  Convergencia en paso {step + 1}")
                break

            prev_loss = loss_val

        # Umbralizar valores cercanos a cero para obtener S verdaderamente dispersa
        S_np = S.numpy()
        S_np[S_np < 1e-4] = 0.0
        self.S = csr_matrix(S_np)
        return self

    def get_coefficients(self) -> csr_matrix:
        if self.S is None:
            raise ValueError("Modelo no entrenado")
        return self.S

    def save(self, path: str):
        if self.S is None:
            raise ValueError("Modelo no entrenado")
        save_npz(path, self.S)

    @classmethod
    def load(cls, path: str):
        inst   = cls.__new__(cls)
        inst.S = load_npz(path)
        return inst
