import ast
import warnings
from typing import List, Optional

import numpy as np
import typer
from thewalrus import hafnian, perm
from tqdm import tqdm

from . import estimates, propagation, sampling

targets = {
    "hafnian": lambda mat: hafnian(mat, method="recursive"),
    "permanent": perm,
}


def invert(mat):
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix, using pseudoinverse")
        return np.linalg.pinv(mat)


def regression_stats(X, y, *, fit_coeffs=True):
    if fit_coeffs:
        XtXinv = invert(X.swapaxes(-2, -1) @ X)
        beta = XtXinv @ (X.swapaxes(-2, -1) @ y)
        eps = y - X @ beta
        beta_var = np.diagonal(XtXinv, axis1=-2, axis2=-1) * (eps**2).mean(-2)
        beta_std = np.maximum(beta_var, 0) ** 0.5
    else:
        beta = np.ones(X.shape[:-2] + (X.shape[-1], 1))
        eps = y - X @ beta
        beta_std = np.zeros(X.shape[:-2] + (X.shape[-1],))
    r_squared = 1 - (eps**2).mean(-2) / y.var(-2)
    return beta, beta_std, r_squared


def linear_regression(
    n,
    features,
    *,
    fit_coeffs=True,
    include_constant=False,
    sampler=sampling.random_complex_double_wishart,
    target="hafnian",
    n_tries=1000,
    n_resamples=10000,
    progress_bar=True,
):
    if include_constant:
        features = features + [lambda mat: 1.0]
    X = []
    y = []
    for _ in tqdm(range(n_tries), disable=not progress_bar):
        mat = sampler(n)
        X.append([feature(mat) for feature in features])
        y.append([targets[target](mat)])
    X = np.array(X)
    y = np.array(y)
    beta, beta_std, r_squared = regression_stats(X, y, fit_coeffs=fit_coeffs)
    btstrap_indices = np.random.randint(0, n_tries, size=(n_resamples, n_tries))
    _, _, btstrap_r_squared = regression_stats(
        X[btstrap_indices], y[btstrap_indices], fit_coeffs=fit_coeffs
    )
    r_squared_std = btstrap_r_squared.std()
    return (
        beta.flatten().tolist(),
        beta_std.flatten().tolist(),
        r_squared.item(),
        r_squared_std.item(),
    )


def hafnian_estimator(kwargs_str):
    try:
        kwargs = ast.literal_eval("{" + kwargs_str + "}")
    except Exception as exn:
        raise ValueError(f"Invalid kwargs string: {kwargs_str}") from exn
    impute = kwargs.pop("impute", False)

    def estimator(mat):
        propagation_fn = (
            propagation.cumulant_propagation_with_imputation
            if impute
            else propagation.cumulant_propagation
        )
        return propagation_fn(mat, **kwargs)

    return estimator


def main(
    min_n: int = 1,
    max_n: int = 20,
    features: List[str] = ["uniq"],
    fit_coeffs: bool = True,
    include_constant: bool = True,
    sampler: str = "01",
    dof: Optional[int] = None,
    target: str = "permanent",
    n_tries: Optional[int] = None,
    n_resamples: int = 10000,
    progress_bar: bool = False,
):
    suffix = "_symmetric" if target == "hafnian" else ""
    if sampler not in ["sign" + suffix, "01" + suffix]:
        assert n_tries is not None
    if "wishart" not in sampler:
        assert dof is None

    feature_strs = features
    sampler_str = sampler
    n_tries_or_none = n_tries
    if target == "permanent":
        features = [estimates.__dict__["est_" + s] for s in feature_strs]
    else:
        features = [hafnian_estimator(s) for s in feature_strs]
    sampler = sampling.__dict__["random_" + sampler_str]

    sampler_kwargs = {}
    if n_tries_or_none is None:
        sampler_kwargs["without_replacement"] = True
        if target == "hafnian":
            sampler_kwargs["constant_diagonal"] = True
    if dof is not None:
        sampler_kwargs["dof"] = dof

    warnings.simplefilter("always")
    for n in range(min_n, max_n + 1):
        if n_tries_or_none is None:
            n_tries = 2 ** (n**2)
            if target == "hafnian":
                n_tries = 2 ** ((n * (n - 1)) // 2)
        else:
            n_tries = n_tries_or_none
        beta, beta_std, r_squared, r_squared_std = linear_regression(
            n,
            features=features,
            fit_coeffs=fit_coeffs,
            include_constant=include_constant,
            sampler=lambda n: sampler(n, **sampler_kwargs),
            target=target,
            n_tries=n_tries,
            n_resamples=n_resamples,
            progress_bar=progress_bar,
        )
        print(f"{n=}: {beta=}, {beta_std=}, {r_squared=}, {r_squared_std=}")


if __name__ == "__main__":
    typer.run(main)
