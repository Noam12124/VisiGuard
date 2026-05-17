import tensorflow as tf
import math


class ArcFace(tf.keras.layers.Layer):
    """
    ArcFace layer implementing additive angular margin penalty.
    Produces logits for softmax classification.
    """

    def __init__(self, num_classes, margin=0.5, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Precompute constants for stability
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        embedding_dim = input_shape[0][-1]

        self.W = self.add_weight(
            name="W",
            shape=(embedding_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        embeddings, labels = inputs

        # Normalize embeddings and weights
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # Cosine similarity
        cosine = tf.matmul(embeddings, W)
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sine
        sine = tf.sqrt(1.0 - tf.square(cosine) + 1e-7)

        # Apply angular margin
        phi = cosine * self.cos_m - sine * self.sin_m

        # Stability boundary
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot labels
        one_hot = tf.one_hot(labels, depth=self.num_classes)

        # Combine margin logits with normal logits
        logits = cosine * (1.0 - one_hot) + phi * one_hot

        # Scale logits
        return logits * self.scale
