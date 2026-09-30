# fashion-ann-pipeline

Fully-connected ANN on Fashon-MNIST, versioned with Git + DVC (Google Drive remote).

```
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib
dvc pull      # fetch data/models from the Drive remote
dvc repro     # rebuild the whole pipeline
dvc metrics show
```
Hyperparameters live in `params.yaml`.
