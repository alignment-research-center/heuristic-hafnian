import numpy as np
from thewalrus import hafnian
from tqdm import tqdm

from .wishart import random_complex_double_wishart


def explained_variance(
    n,
    estimator,
    *,
    sampler=random_complex_double_wishart,
    n_tries=1000,
    n_resamples=10000,
    progress_bar=True,
    **estimator_kwargs,
):
    estimates = []
    exacts = []
    for _ in tqdm(range(n_tries), disable=not progress_bar):
        covariance = sampler(n)
        estimates.append(estimator(covariance, **estimator_kwargs))
        exacts.append(hafnian(covariance, method="recursive"))
    estimates = np.array(estimates)
    exacts = np.array(exacts)
    ev = 1 - (np.abs(exacts - estimates) ** 2).mean() / exacts.var()
    btstrap_indices = np.random.randint(0, n_tries, size=(n_resamples, n_tries))
    btstrap_estimates = estimates[btstrap_indices]
    btstrap_exacts = exacts[btstrap_indices]
    btstrap_evs = 1 - (np.abs(btstrap_exacts - btstrap_estimates) ** 2).mean(1) / (
        btstrap_exacts.var(1)
    )
    ev_std = btstrap_evs.std()
    return ev, ev_std


def linear_regression(
    n,
    features,
    *,
    include_constant=False,
    sampler=random_complex_double_wishart,
    n_tries=1000,
    progress_bar=True,
):
    if include_constant:
        features.append(lambda covariance: 1.0)
    X = []
    y = []
    for _ in tqdm(range(n_tries), disable=not progress_bar):
        covariance = sampler(n)
        X.append([feature(covariance) for feature in features])
        y.append([hafnian(covariance, method="recursive")])
    X = np.array(X)
    y = np.array(y)
    inv = np.linalg.pinv if n == 1 else np.linalg.inv
    XtXinv = inv(X.transpose() @ X)
    beta = XtXinv @ (X.transpose() @ y)
    yhat = X @ beta
    beta_std = (
        np.diag(XtXinv) * (y.transpose() @ y - y.transpose() @ yhat) / n_tries
    ) ** 0.5
    if include_constant:
        rsquared = ((yhat - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    else:
        rsquared = (yhat.transpose() @ yhat) / (y.transpose() @ y)
    return beta.flatten().tolist(), beta_std.flatten().tolist(), rsquared.item()
