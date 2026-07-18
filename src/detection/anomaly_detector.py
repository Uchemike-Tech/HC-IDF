import numpy as np
from sklearn.ensemble import IsolationForest
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


class AnomalyDetector:
    def __init__(self, config: dict):
        self.config = config["detection"]["anomaly"]
        self.isolation_forest = None
        self.autoencoder = None
        self.encoder = None
        self.input_dim = None

    def build_isolation_forest(self):
        cfg = self.config["isolation_forest"]
        self.isolation_forest = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=42,
        )

    def build_autoencoder(self, input_dim: int):
        if not _HAS_TF:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        self.input_dim = input_dim
        cfg = self.config["autoencoder"]
        encoding_dim = cfg["encoding_dim"]

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim * 2, activation="relu")(input_layer)
        encoded = layers.Dense(encoding_dim, activation="relu")(encoded)
        decoded = layers.Dense(encoding_dim * 2, activation="relu")(encoded)
        decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        self.autoencoder.compile(optimizer="adam", loss="mse")

    def fit_autoencoder(self, X_train, X_val=None):
        cfg = self.config["autoencoder"]
        validation_data = (X_val, X_val) if X_val is not None else None
        self.autoencoder.fit(
            X_train, X_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=validation_data,
            verbose=0,
        )

    def predict_isolation_forest(self, X):
        predictions = self.isolation_forest.predict(X)
        return np.where(predictions == -1, 1, 0)

    def predict_autoencoder(self, X):
        if not _HAS_TF:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean((X - reconstructions) ** 2, axis=1)
        threshold = np.percentile(mse, 95)
        return (mse > threshold).astype(int), mse
