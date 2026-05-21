"""
GestureSpeak — Model Architectures
architectures.py

Only change from v1:
  get_config() now returns the full parent config so the custom
  AttentionLayer survives tf.lite.TFLiteConverter without errors.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────
# CUSTOM ATTENTION LAYER
# ─────────────────────────────────────────────
class AttentionLayer(layers.Layer):
    """
    Soft (Bahdanau-style) attention over timesteps.
    Input : (batch, timesteps, features)
    Output: (batch, features) — weighted sum across timesteps
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score   = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context

    def get_config(self):
        # Return full config so TFLite converter and load_model
        # can reconstruct this layer without errors
        base = super().get_config()
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ─────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────
def build_bilstm_attention(seq_len: int, feature_size: int, num_classes: int) -> Model:
    inp = layers.Input(shape=(seq_len, feature_size), name="sequence_input")

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    x = AttentionLayer(name="attention")(x)

    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="drop_3")(x)

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3, name="drop_4")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="GestureSpeak_BiLSTM_Attention")


if __name__ == "__main__":
    model = build_bilstm_attention(seq_len=30, feature_size=126, num_classes=21)
    model.summary()