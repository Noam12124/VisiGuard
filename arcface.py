import tensorflow as tf
import math


class ArcFace(tf.keras.layers.Layer):
    """
    Stable ArcFace implementation (training-safe + numerically stable)
    """

    def __init__(self, num_classes, margin=0.3, scale=64.0, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # constants
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

    def call(self, inputs, training=True):

        embeddings, labels = inputs

        # normalize
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cosine
        cosine = tf.matmul(embeddings, W)
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # sine (stable)
        sine = tf.sqrt(
            tf.clip_by_value(1.0 - tf.square(cosine), 0.0, 1.0)
        )

        # phi (angular margin)
        phi = cosine * self.cos_m - sine * self.sin_m

        # boundary condition
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot labels
        one_hot = tf.one_hot(labels, depth=self.num_classes)

        # combine
        logits = (1.0 - one_hot) * cosine + one_hot * phi

        return logits * self.scale