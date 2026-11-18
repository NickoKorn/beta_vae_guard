"""Class information in the latent space (analysis only, no training).

    analyze_posteriors(folder, annotation)  per stored checkpoint and class: validation
        one-vs-rest AUC of a train-fitted linear probe on mu and on sampled z, with pointwise
        stratified bootstrap intervals. z is drawn ``draws`` times (independent seeds); one
        fixed draw repeated over checkpoints gave a constant spurious AUC (0.64 on
        weak_signal_lowvar after full collapse), so every draw is reported separately.
    auc_curve(folder)  macro AUC on mu at every checkpoint of a saved pipeline (pipeline.pkl)
    auc_interval(y, score)

Failure to demonstrate AUC > 0.5 is not proof that no class information exists. Test data
are not used to choose checkpoints, controller targets or beta candidates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def auc_interval(y, score, *, seed=17, repeats=200):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    rng = np.random.default_rng(seed)
    samples = np.concatenate([pos[rng.integers(len(pos), size=(repeats, len(pos)))],
                              neg[rng.integers(len(neg), size=(repeats, len(neg)))]], axis=1)
    ranks = rankdata(samples, axis=1)
    bootstrap = (ranks[:, :len(pos)].sum(1) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(roc_auc_score(y, score)), *map(float, np.quantile(bootstrap, [.025, .975]))


def analyze_posteriors(folder, annotation, target='CANCER_TYPE_ACRONYM', draws=5):
    folder = Path(folder)
    index = [json.loads(line) for line in (folder / 'posterior_index.jsonl').read_text().splitlines()]
    rows = []
    for checkpoint in index:
        data = np.load(folder / checkpoint['path'])
        labels = {split: annotation[target].reindex(data[f'{split}_ids']) for split in ('train', 'valid')}
        keep = {split: labels[split].notna().to_numpy() for split in labels}
        y = {split: labels[split].to_numpy()[keep[split]].astype(str) for split in labels}
        if len(np.unique(y['train'])) < 2:
            continue
        variants = [('mu', None)] + [('sampled_z', d) for d in range(draws)]
        for representation, draw in variants:
            x = {}
            rng = np.random.default_rng(17 + (draw or 0))
            for split in ('train', 'valid'):
                mu, logvar = data[f'{split}_mu'], data[f'{split}_logvar']
                features = mu if draw is None else mu + np.exp(.5 * logvar) * rng.standard_normal(mu.shape)
                x[split] = features[keep[split]]
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=17))
            probe.fit(x['train'], y['train'])
            scores = probe.predict_proba(x['valid'])
            classes = probe[-1].classes_
            for j, label in enumerate(classes):
                is_class = y['valid'] == label
                result = auc_interval(is_class, scores[:, j])
                row = dict(**checkpoint, representation=representation, draw=draw, cancer_class=str(label),
                           n_train=len(y['train']), n_valid=len(y['valid']), n_valid_positive=int(is_class.sum()),
                           auc=None, ci_low=None, ci_high=None, evidence='insufficient_validation_support')
                if result is not None:
                    row.update(auc=result[0], ci_low=result[1], ci_high=result[2],
                               evidence='above_chance' if result[1] > .5 else
                                        'below_chance' if result[2] < .5 else 'inconclusive_vs_chance')
                rows.append(row)
        pd.DataFrame(rows).to_csv(folder / 'class_auc.csv', index=False)
    return pd.DataFrame(rows)


def summarize_auc(table):
    """Macro AUC per checkpoint and representation; sampled z: mean and range over draws."""
    t = pd.DataFrame(table)
    per = t.groupby(['epoch_number', 'representation', 'draw'], dropna=False).auc.mean().reset_index()
    return per.groupby(['epoch_number', 'representation']).auc.agg(auc_mean='mean', auc_min='min',
                                                                    auc_max='max', draws='size').reset_index()

def auc_curve(folder, target="CANCER_TYPE_ACRONYM", seed=0):
    """Downstream AUC at every epoch with stored train/valid posterior means.

    Autoencodix keeps mu for train and valid only at checkpoints (every
    ``checkpoint_interval`` epochs). Per checkpoint: LogisticRegression on mu(train),
    macro one-vs-rest ROC AUC on mu(valid). Missing mu (all zero after total collapse,
    not stored by Autoencodix) counts as zeros. Writes auc_curve.csv in the run folder.
    """
    import autoencodix as acx
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from src.core.load_data import tcga_files

    folder = Path(folder).resolve()
    labels = pd.read_parquet(tcga_files()["clinical"])[target]
    pipeline = acx.Varix.load(str(folder / "pipeline.pkl"))
    result = pipeline.result
    epochs = sorted(e for e in result.sample_ids._data if e >= 0)
    del pipeline
    beta = {r["epoch"]: r["effective_beta"] for r in map(json.loads, (folder / "collapse_history.jsonl").read_text().splitlines())}
    rows = []
    for epoch in epochs:
        data = {}
        for split in ("train", "valid"):
            ids = np.asarray(result.sample_ids.get(epoch=epoch, split=split)).astype(str)
            mu = np.asarray(result.mus.get(epoch=epoch, split=split))
            mu = np.zeros((len(ids), pipeline_dim(result))) if mu.size == 0 else mu
            y = labels.reindex(ids)
            keep = y.notna().to_numpy()
            data[split] = (mu[keep], y[keep].to_numpy())
        (xt, yt), (xv, yv) = data["train"], data["valid"]
        known = np.isin(yv, np.unique(yt))
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(xt, yt)
        proba = clf.predict_proba(xv[known])
        if len(clf.classes_) == 2:          # binary (e.g. weak_signal): score of the second class
            auc = roc_auc_score(yv[known] == clf.classes_[1], proba[:, 1])
        else:
            auc = roc_auc_score(yv[known], proba, multi_class="ovr", average="macro", labels=clf.classes_)
        rows.append(dict(epoch=int(epoch), beta=beta.get(int(epoch)), auc_valid=float(auc), n_train=len(yt),
                         n_valid=int(known.sum()), mu_all_zero=bool(np.ptp(xt) == 0)))
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "auc_curve.csv", index=False)
    return frame


def pipeline_dim(result):
    """Latent dimension from any stored latent space."""
    for e in sorted(result.latentspaces._data):
        for split in ("train", "valid", "test"):
            z = result.latentspaces.get(epoch=e, split=split)
            if z is not None and np.asarray(z).size:
                return np.asarray(z).shape[1]
    raise ValueError("no latent space stored")

