"""Stage 2: normalize to [0, 1] and split a validation set."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["preprocess"]

    raw = np.load("data/raw/fashion_mnist_raw.npz")
    x_train = raw["x_train"].astype("float32") / 255.0   # normalization step
    x_test = raw["x_test"].astype("float32") / 255.0
    y_train, y_test = raw["y_train"], raw["y_test"]

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=p["val_size"], random_state=p["seed"], stratify=y_train,
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        "data/processed/fashion_mnist_processed.npz",
        x_train=x_tr, y_train=y_tr, x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    print(f"train={x_tr.shape}, val={x_val.shape}, test={x_test.shape}")


if __name__ == "__main__":
    main()
