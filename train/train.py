#!/usr/bin/env python3
"""Caminho C — treina um classificador GBDT (xgboost) OFFLINE e exporta as
árvores para inferência em Go puro.

Lê os vetores JÁ vetorizados de references.json.gz (campos `vector`[14] + `label`)
— NÃO replica o vectorize.go. Treina binary:logistic, e exporta:

  resources/model.json         -> {base_score, tau, n_features, trees:[...]}
  resources/model_parity.json  -> {vectors:[[...]], proba:[...]}  (âncora de paridade p/ o teste Go)

Uso (a partir do dir da API):
  python -m pip install xgboost scikit-learn   # uma vez
  python train/train.py                        # defaults abaixo

O runtime continua Go puro: o model.json vira uma tabela embutida; nada de Python na imagem.
"""
import gzip
import json
import os
import time

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.environ.get("REFERENCES_PATH", os.path.join(API_DIR, "resources", "references.json.gz"))
OUT_MODEL = os.environ.get("MODEL_OUT", os.path.join(API_DIR, "resources", "model.json"))
OUT_PARITY = os.environ.get("PARITY_OUT", os.path.join(API_DIR, "resources", "model_parity.json"))

DIMS = 14
N_ROUNDS = int(os.environ.get("N_ROUNDS", "100"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "8"))
ETA = float(os.environ.get("ETA", "0.3"))
TAU = float(os.environ.get("TAU", "0.5"))  # approved = P(fraud) < TAU; sweep no cmd/diag depois
SEED = 42
# If set, train on 5-NN IMITATION labels from cmd/genlabels (target = the gabarito's
# 5-NN decision) instead of the noisy individual reference labels. See JORNADA §9.
LABELS_BIN = os.environ.get("LABELS_BIN", "")


def load_refs(path):
    """Stream references.json.gz into X (n,14) float32 and y (n,) uint8."""
    t0 = time.time()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)  # array of {"vector":[...], "label":"fraud"|"legit"}
    n = len(data)
    X = np.empty((n, DIMS), dtype=np.float32)
    y = np.empty(n, dtype=np.uint8)
    for i, rec in enumerate(data):
        X[i] = rec["vector"]
        y[i] = 1 if rec["label"] == "fraud" else 0
    print(f"loaded {n} refs in {time.time()-t0:.1f}s  (fraude={y.mean()*100:.2f}%)")
    return X, y


def load_imitation(path):
    """Load 5-NN imitation labels (cmd/genlabels binary: 15 float32 per example =
    14-dim vector + 5-NN fraud_score). Target y = the 5-NN "denied" decision."""
    t0 = time.time()
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, DIMS + 1)
    X = raw[:, :DIMS].copy()
    score = raw[:, DIMS]
    y = (score >= 0.6).astype(np.uint8)  # denied (approved = score < 0.6)
    print(f"loaded {len(y)} imitation examples in {time.time()-t0:.1f}s  (denied={y.mean()*100:.2f}%)")
    return X, y


def main():
    if LABELS_BIN:
        print(f"IMITATION mode: target = 5-NN decision ({LABELS_BIN})")
        X, y = load_imitation(LABELS_BIN)
    else:
        print("INDIVIDUAL mode: target = each ref's own label")
        X, y = load_refs(REF_PATH)

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xval, label=yval)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": MAX_DEPTH,
        "eta": ETA,
        "tree_method": "hist",
        "base_score": 0.5,
        "seed": SEED,
    }
    print(f"training xgboost: rounds={N_ROUNDS} depth={MAX_DEPTH} eta={ETA} ...")
    t0 = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=N_ROUNDS,
                    evals=[(dval, "val")], verbose_eval=50)
    print(f"trained in {time.time()-t0:.1f}s")

    # --- base_score (margin offset) from the model config ---
    cfg = json.loads(bst.save_config())
    # xgboost 3.x reports base_score as a vector-format string, e.g. '[5E-1]'.
    raw_bs = cfg["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(raw_bs).strip("[]"))
    print(f"base_score = {base_score}  (raw {raw_bs!r})")

    # --- export trees (one JSON object per tree) ---
    trees = [json.loads(s) for s in bst.get_dump(dump_format="json")]
    model = {
        "base_score": base_score,   # probability; Go converts to margin = logit(base_score)
        "tau": TAU,                 # approved when P(fraud) < tau
        "n_features": DIMS,
        "n_trees": len(trees),
        "trees": trees,
    }
    with open(OUT_MODEL, "w") as f:
        json.dump(model, f)
    print(f"wrote {OUT_MODEL}  ({os.path.getsize(OUT_MODEL)/1e6:.2f} MB, {len(trees)} trees)")

    # --- parity sample: Go Score(v) must match these probas within 1e-5 ---
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=256, replace=False)
    sample = X[idx]
    proba = bst.predict(xgb.DMatrix(sample))
    parity = {"vectors": sample.astype(float).tolist(), "proba": proba.astype(float).tolist()}
    with open(OUT_PARITY, "w") as f:
        json.dump(parity, f)
    print(f"wrote {OUT_PARITY}  ({len(idx)} samples)")

    # --- quick sanity on the val split (individual-label accuracy, NOT the 5-NN target) ---
    pval = bst.predict(dval)
    acc = ((pval >= TAU).astype(np.uint8) == yval).mean()
    print(f"val individual-label acc @tau={TAU}: {acc*100:.3f}%  "
          f"(real target = 5-NN agreement, medido no cmd/diag)")


if __name__ == "__main__":
    main()
