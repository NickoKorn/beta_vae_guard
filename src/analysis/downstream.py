import json
from pathlib import Path

import numpy as np
import pandas as pd


def auroc_ovo(x_train, y_train, x_valid, y_valid):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import get_scorer

    known = np.isin(y_valid, np.unique(y_train))
    classifier = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    return float(get_scorer("roc_auc_ovo")(classifier, x_valid[known], y_valid[known]))


def analyze_posteriors(folder, annotation, target="CANCER_TYPE_ACRONYM", draws=5):
    folder = Path(folder)
    index = [json.loads(line) for line in (folder / "posterior_index.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for checkpoint in index:
        data = np.load(folder / checkpoint["path"])
        y, keep = {}, {}
        for split in ("train", "valid"):
            labels = annotation[target].reindex(data[f"{split}_ids"])
            keep[split] = labels.notna().to_numpy()
            y[split] = labels.to_numpy()[keep[split]].astype(str)
        if len(np.unique(y["train"])) < 2:
            continue
        for representation, draw in [("mu", None)] + [("sampled_z", d) for d in range(draws)]:
            rng = np.random.default_rng(17 + (draw or 0))
            x = {}
            for split in ("train", "valid"):
                mu, logvar = data[f"{split}_mu"], data[f"{split}_logvar"]
                x[split] = (mu if draw is None else mu + np.exp(.5 * logvar) * rng.standard_normal(mu.shape))[keep[split]]
            rows.append(dict(epoch_number=checkpoint.get("epoch_number"), beta=checkpoint.get("beta"),
                             path=checkpoint["path"], representation=representation, draw=draw,
                             n_train=len(y["train"]), n_valid=len(y["valid"]),
                             auroc_ovo=auroc_ovo(x["train"], y["train"], x["valid"], y["valid"])))
    table = pd.DataFrame(rows)
    table.to_csv(folder / "auroc.csv", index=False)
    return table


def summarize_auroc(table):
    t = pd.DataFrame(table)
    return t.groupby(["epoch_number", "representation"]).auroc_ovo.agg(
        auroc_mean="mean", auroc_min="min", auroc_max="max", draws="size").reset_index()
