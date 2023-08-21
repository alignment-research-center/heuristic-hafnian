# Heuristic hafnian approximation

Utilities for estimating hafnians of matrices using cumulant propagation and related algorithms.

Example:

```
>>> from heuristic_hafnian import cumulant_propagation, random_double_wishart
>>> cov = random_double_wishart(10)
>>> cumulant_propagation(cov, order=1)  # Mean propagation estimate
0
>>> cumulant_propagation(cov, order=2)  # Covariance propagation estimate
10856.32167102877
>>> cumulant_propagation(cov, order=3)  # Third cumulant propagation estimate
11414.589102340478
>>> cumulant_propagation(cov, order=None)  # Exact hafnian calculation
32274.905033773324
```

Use the `evaluation` module to calculate the explained variance of different estimators over different distributions.
