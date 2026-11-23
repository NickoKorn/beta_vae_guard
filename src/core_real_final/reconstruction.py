"""Reconstruction scores for exported posterior snapshots."""

import numpy as np


def reconstruction_r2(x, recon):
    """Reconstruction R^2 of a samples-by-features matrix, three conventions.

    r2_sample    AUTOENCODIX (reproducibility repo, src/utils/utils.py):
                 torcheval r2_score(recon.T, x.T, multioutput="uniform_average"), i.e. R^2 of each
                 sample's profile over the features, averaged over samples
    r2_feature   classic R^2 per feature over the samples (sklearn r2_score, uniform average);
                 features without variance in this split are left out
    r2_weighted  1 - sum SSE / sum SST over all features (sklearn "variance_weighted")
    """
    x, recon = np.asarray(x, dtype=np.float64), np.asarray(recon, dtype=np.float64)
    if x.shape != recon.shape or x.ndim != 2:
        raise ValueError("x and recon must be matching samples-by-features matrices")
    sse = (x - recon) ** 2
    sst_sample = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
    sst_feature = ((x - x.mean(0)) ** 2).sum(0)
    ok_s, ok_f = sst_sample > 0, sst_feature > 0
    return dict(r2_sample=float(np.mean(1 - sse.sum(1)[ok_s] / sst_sample[ok_s])),
                r2_feature=float(np.mean(1 - sse.sum(0)[ok_f] / sst_feature[ok_f])),
                r2_weighted=float(1 - sse.sum() / sst_feature.sum()))


