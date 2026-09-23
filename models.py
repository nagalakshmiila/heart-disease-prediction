"""
models.py
---------
Two model families are defined here:

1. build_baseline_ann(input_dim)
   A plain fully-connected ANN, similar to what most existing / base-paper
   heart-disease work reports (this is what you train as your "previous
   work" comparison baseline, on the SAME optimized feature set, so the
   comparison is apples-to-apples).

2. build_hybrid_cnn_bilstm_attention(input_dim)
   The PROPOSED model for this project - the "Hybrid Deep Learning" part
   of the title:
     - Reshapes the optimized feature vector into a small sequence.
     - A 1D-CNN branch learns local interactions between neighboring
       (correlated) clinical features.
     - A Bidirectional LSTM branch learns longer-range contextual
       dependencies across the whole feature sequence.
     - The two branches are concatenated and passed through a
       self-attention layer so the network learns to weigh the most
       diagnostically important features per patient.
     - A small dense head outputs the probability of heart disease.

3. build_stacked_ensemble(...)
   The final "hybrid" decision layer: a Gradient Boosting meta-learner
   trained on [hybrid-DL probability, RandomForest probability,
   SVM probability] to squeeze out extra accuracy - this is the ensemble
   trick that pushes accuracy above a plain single-model baseline.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_baseline_ann(input_dim: int) -> Model:
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out, name="baseline_ann")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def _attention_block(x, name="attention"):
    # simple additive self-attention over the time/feature axis
    score = layers.Dense(1, activation="tanh")(x)
    weights = layers.Softmax(axis=1, name=f"{name}_weights")(score)
    context = layers.Multiply()([x, weights])

    # Sum across the feature/time axis without using Lambda
    context = layers.GlobalAveragePooling1D(name=f"{name}_context")(context)

    return context

def build_hybrid_cnn_bilstm_attention(input_dim: int) -> Model:
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Reshape((input_dim, 1))(inp)

    # CNN branch: local feature-interaction patterns
    cnn = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(cnn)
    cnn = layers.MaxPooling1D(pool_size=1)(cnn)

    # BiLSTM branch: contextual dependencies across the feature sequence
    bilstm = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    merged = layers.Concatenate(axis=-1)([cnn, bilstm])
    attended = _attention_block(merged)

    d = layers.Dense(64, activation="relu")(attended)
    d = layers.Dropout(0.4)(d)
    d = layers.Dense(32, activation="relu")(d)
    d = layers.Dropout(0.2)(d)
    out = layers.Dense(1, activation="sigmoid", name="disease_probability")(d)

    model = Model(inp, out, name="hybrid_cnn_bilstm_attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def build_stacked_ensemble():
    """Meta-learner that fuses the hybrid DL model with two classical
    optimized ML models (Random Forest, SVM) for the final prediction."""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                       learning_rate=0.05, random_state=42)
