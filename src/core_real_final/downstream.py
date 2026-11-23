"""Class information in the latent space, evaluated like AUTOENCODIX (analysis only, no training).

AUTOENCODIX (evaluate/_general_evaluator.py): LogisticRegression(max_iter=1000) fitted on the train
split, scored with roc_auc_ovo on the other splits; "Latent" is result.latentspaces, which for Varix is
the sampled z of the forward pass. Here the same classifier and metric on the stored posterior snapshots:

    analyze_posteriors(folder, annotation)   per snapshot: AUROC on mu and on sampled z (draws
        reported separately: one fixed draw once gave a spurious 0.64 after full collapse) -> auroc.csv
    auroc_ovo(x_train, y_train, x_valid, y_valid)

Failure to show AUROC > 0.5 is not proof that no class information exists. Test data are not used
to choose checkpoints, controller targets or beta candidates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auroc_ovo(x_train, y_train, x_valid, y_valid):
    """AUTOENCODIX: LogisticRegression(max_iter=1000) on train, roc_auc_ovo on valid (classes seen in train)."""
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
    """AUROC per snapshot and representation; sampled z: mean and range over draws."""
    t = pd.DataFrame(table)
    return t.groupby(["epoch_number", "representation"]).auroc_ovo.agg(
        auroc_mean="mean", auroc_min="min", auroc_max="max", draws="size").reset_index()
