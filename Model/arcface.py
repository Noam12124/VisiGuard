"""
arcface.py — ArcFace (Additive Angular Margin) loss layer.

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
Face Recognition", CVPR 2019. https://arxiv.org/abs/1801.07698

Key design decisions:
  • Numerically stable: clips cos(θ) to [-1+ε, 1-ε] before arccos.
  • Easy cos(θ+m): uses the angle-addition identity so we never need
    the label one-hot inside the kernel — only during the forward pass.
  • Scale s (temperature) applied after the margin shift.
  • get_config / from_config implemented for model serialisation.
"""

import math
import tensorflow as tf
from tensorflow.keras.layers import Layer
import config


class ArcFaceLayer(Layer):
    """
    ArcFace classification head.

    During training:
        logits = s * [cos(θ_i + m)  if i == true_class
                      cos(θ_i)       otherwise]

    At inference the layer is NOT used; only the L2-normalised 512-dim
    embedding vector is needed for cosine similarity matching.

    Args:
        num_classes:  Number of identities in the training set.
        margin:       Additive angular margin (radians). Default 0.5 ≈ 28.6°.
        scale:        Feature scale / logit temperature.  Default 64.
    """

    def __init__(
        self,
        num_classes: int,
        margin: float = config.ARCFACE_MARGIN,
        scale: float  = config.ARCFACE_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin      = margin
        self.scale       = scale

        # Precompute trigonometric constants once.
        self.cos_m = tf.cast(math.cos(margin), tf.float32)
        self.sin_m = tf.cast(math.sin(margin), tf.float32)
        # Threshold: if cos(θ) < this, adding the margin would push θ+m past π.
        # In that case we fall back to cos(θ) − sin(m)*m (the linear approx).
        self.threshold = tf.cast(math.cos(math.pi - margin), tf.float32)
        self.mm        = tf.cast(math.sin(math.pi - margin) * margin, tf.float32)

    def build(self, input_shape):
        # W: (embedding_dim, num_classes)  — one unit-vector per identity.
        self.W = self.add_weight(
            name        = "arcface_weights",
            shape       = (int(input_shape[-1]), self.num_classes),
            initializer = "glorot_uniform",
            trainable   = True,
            regularizer = tf.keras.regularizers.l2(config.L2_REGULARIZER),
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None, training=False):
        """
        Args:
            embeddings: (batch, embedding_dim)  — already L2-normalised.
            labels:     (batch,) integer class indices.  Required for training.
            training:   bool.

        Returns:
            logits: (batch, num_classes)
        """
        # Determine current compute precision (e.g., float16 under mixed precision)
        dtype = embeddings.dtype

        # L2-normalise both embeddings and weights → cos(θ) = emb · W_norm
        emb_norm = tf.nn.l2_normalize(embeddings, axis=1)
        W_norm   = tf.nn.l2_normalize(tf.cast(self.W, dtype), axis=0)

        # cos(θ)  shape: (batch, num_classes)
        cos_theta = tf.matmul(emb_norm, W_norm)
        
        # Cast scalar thresholds to match compute precision
        one_cast = tf.cast(1.0, dtype)
        eps_cast = tf.cast(1e-7, dtype)
        cos_theta = tf.clip_by_value(cos_theta, -one_cast + eps_cast, one_cast - eps_cast)

        if not training or labels is None:
            return tf.cast(self.scale, dtype) * cos_theta

        # sin(θ) via Pythagorean identity  (numerically stable, no arccos needed)
        sin_theta = tf.sqrt(tf.maximum(one_cast - tf.square(cos_theta), eps_cast))

        # Dynamically cast precomputed weights to match precision
        cos_m_cast = tf.cast(self.cos_m, dtype)
        sin_m_cast = tf.cast(self.sin_m, dtype)
        threshold_cast = tf.cast(self.threshold, dtype)
        mm_cast = tf.cast(self.mm, dtype)

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        cos_theta_m = cos_theta * cos_m_cast - sin_theta * sin_m_cast

        # Fall-back for θ + m > π (prevents going "past" the antipodal point)
        cos_theta_m = tf.where(
            cos_theta > threshold_cast,
            cos_theta_m,
            cos_theta - mm_cast,
        )

        # One-hot mask for the true class
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes, dtype=dtype)

        # Replace true-class cosine with margin-shifted version
        logits = one_hot * cos_theta_m + (one_cast - one_hot) * cos_theta

        return tf.cast(self.scale, dtype) * logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            num_classes = self.num_classes,
            margin      = self.margin,
            scale       = self.scale,
        )
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)