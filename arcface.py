import tensorflow as tf
import math


class ArcFace(tf.keras.layers.Layer):

    def __init__(self, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # stable precomputations
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, embeddings, labels):

        # normalize
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # cosine similarity
        cosine = tf.matmul(embeddings, W)
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        sine = tf.sqrt(1.0 - tf.square(cosine) + 1e-7)

        # phi (angular margin)
        phi = cosine * self.cos_m - sine * self.sin_m

        # stability boundary
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot labels
        one_hot = tf.one_hot(labels, depth=self.num_classes)

        logits = cosine * (1.0 - one_hot) + phi * one_hot

        return logits * self.scale