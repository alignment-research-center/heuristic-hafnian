# Heuristic hafnian estimation

Code related to the paper: [**Towards a Law of Iterated Expectations for Heuristic Estimators**](https://arxiv.org/abs/2410.01290)

This repo contains utilities for calculating heuristic estimates of permanents and hafnians, as well as a script for estimating OLS regression coefficients over different distributions.

Estimators:

- Estimators for the permanent only include `est_row`, `est_col`, `est_sum`, `est_uniq` and `product_over_permutation`.
- Estimators for the hafnian include `cumulant_propagation`, `cumulant_propagation_with_imputation` and `product_over_pairing`.

Sampling functions:

- Functions for sampling matrices suitable for the permanent include `random_normal`, `random_sign`, `random_01` and `random_wishart`.
- Functions for sampling matrices suitable for the hafnian include `random_normal_symmetric`, `random_sign_symmetric`, `random_01_symmetric` and `random_double_wishart`.

Example for the permanent:

```
>>> from heuristic_hafnian import est_row, est_sum, random_wishart, zero_block_diag, cumulant_propagation
>>> from thewalrus import perm
>>> mat = random_wishart(5)
>>> est_row(mat)   # n!/n^n times product of row sums
10251.233861678926
>>> est_sum(mat)   # n!/n^(2n) times sum of entries to the power n
11781.581263992726
>>> cumulant_propagation(zero_block_diag(mat))  # Exact permanent using cumulant propagation
32871.99246021791
>>> perm(mat)  # Faster exact permanent
32871.9924602179
```

Example for the hafnian:

```
>>> from heuristic_hafnian import cumulant_propagation, cumulant_propagation_with_imputation, random_double_wishart
>>> from thewalrus import hafnian
>>> cov = random_double_wishart(10)
>>> cumulant_propagation(cov, order=1)  # Mean propagation estimate
0
>>> cumulant_propagation(cov, order=2)  # Covariance propagation estimate
10856.32167102877
>>> cumulant_propagation(cov, order=3)  # 3rd-order cumulant propagation estimate
11414.589102340478
>>> cumulant_propagation_with_imputation(cov, order=3)  # 3rd-order cumulant propagation with imputation estimate
12260.347921286422
>>> cumulant_propagation(cov)  # Exact hafnian using cumulant propagation
32274.905033773324
>>> hafnian(cov, method="recursive")  # Faster exact hafnian
32274.905033773324
```

OLS regression coefficients of different estimators over different distributions can be calculated using the `evaluation` module. For example, to estimate joint OLS regression coefficients for the row sum, column sum and matrix sum estimators for the permanent over matrices with independent standard Gaussian entries, run:

```
python -m heuristic_hafnian.evaluation --features row --features col --features sum --no-include-constant --sampler normal --target permanent --n-tries 10000
```

See the docstring in that module for further usage instructions.