"""Stage 4: evaluate on the test set, write metrics.json and a confusion matrix."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model("models/model.h5")
    d = np.load("data/processed/fashion_mnist_processed.npz")
    x_test, y_test = d["x_test"], d["y_test"]

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=120)
    print(f"test_loss={loss:.4f} test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
