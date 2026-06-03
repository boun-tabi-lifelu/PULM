
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import accuracy_score

from plm_benchmark.config import CACHE_DIR, EMBED_CACHE_SUFFIX, LR_EMBED_HEAD
from plm_benchmark.models.esm import mean_pool_embeddings
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_sequences


def _cache_paths(cache_dir: Path, split: str, model_name: str) -> Path:
    suffix = EMBED_CACHE_SUFFIX.get(model_name, model_name)
    return cache_dir / f"{split}_{suffix}.pkl"


def _load_or_embed(
    checkpoint: str,
    df: pd.DataFrame,
    split: str,
    cache_dir: Path,
    gpu: int,
    reembed: bool,
    max_length: int,
    embed_batch: int,
    task_name: str,
    model_name: str,
) -> pd.DataFrame:
    path = _cache_paths(cache_dir, split, model_name)
    if path.exists() and not reembed:
        return pd.read_pickle(path)
    emb = mean_pool_embeddings(
        checkpoint,
        df,
        gpu,
        max_length=max_length,
        batch_size=embed_batch,
        desc=f"{task_name} {split}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    emb.to_pickle(path)
    return emb


def _xy(emb: pd.DataFrame):
    return np.array(emb.iloc[:, :-2]), np.array(emb.iloc[:, -1])


def _train_head(train_emb, valid_emb, test_emb, spec: TaskSpec, epochs: int, seed: int, batch: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_train, y_train = _xy(train_emb)
    x_valid, y_valid = _xy(valid_emb)
    x_test, y_test = _xy(test_emb)

    norm = layers.Normalization(axis=-1)
    norm.adapt(x_train)

    loss = (
        keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if spec.task_type == "classification"
        else "mean_squared_error"
    )
    model = keras.Sequential(
        [norm, layers.Dense(32, activation="relu"), layers.Dropout(0.2), layers.Dense(spec.num_labels)]
    )
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(LR_EMBED_HEAD, epsilon=1e-7))

    history = []

    class CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict(x_valid, verbose=0)
            if spec.task_type == "classification":
                s = accuracy_score(y_valid, pred.argmax(axis=-1))
            else:
                s = scipy.stats.spearmanr(y_valid, pred.squeeze()).correlation
            history.append(float(s))
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs:
                print(f"  epoch {epoch + 1}/{epochs}  val_{spec.metric}={s:.4f}", flush=True)

    model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[CB()], verbose=0)
    pred = model.predict(x_test, verbose=0)
    if spec.task_type == "classification":
        test = float(accuracy_score(y_test, pred.argmax(axis=-1)))
    else:
        test = float(scipy.stats.spearmanr(y_test, pred.squeeze()).correlation)
    return test, max(history) if history else None


def run_embed_head(
    spec: TaskSpec,
    checkpoint: str,
    model_name: str,
    train_df,
    valid_df,
    test_df,
    *,
    gpu: int,
    epochs: int,
    batch: int,
    seed: int,
    reembed: bool,
    max_length: int,
    embed_batch: int,
) -> dict:
    train_df, valid_df, test_df = map(preprocess_sequences, (train_df, valid_df, test_df))
    cache_dir = CACHE_DIR / spec.name.lower()

    print(f"\n=== {spec.name} | embed_head | {checkpoint} ===", flush=True)
    train_emb = _load_or_embed(
        checkpoint, train_df, "train", cache_dir, gpu, reembed, max_length, embed_batch, spec.name, model_name
    )
    valid_emb = _load_or_embed(
        checkpoint, valid_df, "valid", cache_dir, gpu, reembed, max_length, embed_batch, spec.name, model_name
    )
    test_emb = _load_or_embed(
        checkpoint, test_df, "test", cache_dir, gpu, reembed, max_length, embed_batch, spec.name, model_name
    )

    test, val = _train_head(train_emb, valid_emb, test_emb, spec, epochs, seed, batch)
    print(f"Test {spec.metric}: {test:.4f} | best val: {val}", flush=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": spec.name,
        "model": model_name,
        "checkpoint": checkpoint,
        "method": "embed_head",
        "metric": spec.metric,
        "test_score": round(test, 6),
        "val_score": round(val, 6) if val is not None else "",
        "epochs": epochs,
        "lr": LR_EMBED_HEAD,
        "batch": batch,
        "seed": seed,
        "run_dir": str(cache_dir),
    }
    append_experiment(row)
    return row
