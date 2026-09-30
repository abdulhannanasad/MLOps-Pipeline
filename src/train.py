"""Stage 3: build and train the ANN (Flatten -> Dense/ReLU -> Dropout -> Dense(10)/Softmax)."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow import keras


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    tf.keras.utils.set_random_seed(p["seed"])
    d = np.load("data/processed/fashion_mnist_processed.npz")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        d["x_train"], d["y_train"],
        validation_data=(d["x_val"], d["y_val"]),
        epochs=p["epochs"], batch_size=p["batch_size"], verbose=2,
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index_label="epoch")


if __name__ == "__main__":
    main()
