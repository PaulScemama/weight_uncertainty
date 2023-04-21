from bayesian_nn import sigmas_from_rhos, logvariational_fn, samplevariational_fn, rhos_from_sigmas
import time
import numpy as np
import scipy.stats as stats
import pytest
import torch


@pytest.mark.parametrize(
    "weights, mus, rhos",
    [
        (
            torch.ones((3, 2)),
            torch.ones((3, 2)) * 2,
            torch.arange(1, 7, dtype=float).reshape(3, 2),
        ),
        (
            torch.ones((2, 1)),
            torch.ones((2, 1)) * 2,
            torch.arange(2, 4, dtype=float).reshape(2, 1),
        ),
    ],
)
def test_logvariational_fn(weights, mus, rhos):
    sigmas = sigmas_from_rhos(rhos)
    covariance_diagonals = sigmas.square()
    results = logvariational_fn(weights, mus, rhos)
    for i, result in enumerate(results):
        ground_truth = stats.multivariate_normal.logpdf(weights[i], mus[i], covariance_diagonals[i])
        assert np.allclose(result, ground_truth)



def test_logvariational_fn_2():
    n = 64**2
    d = 4

    means = torch.FloatTensor(n,d).uniform_(-1, 1)
    covariances = torch.FloatTensor(n,d).uniform_(0, 2)
    rhos = rhos_from_sigmas(covariances.sqrt())
    X = torch.FloatTensor(n,d).uniform_(-1, 1)

    refs = []

    ref_start = time.time()
    for x, mean, covariance in zip(X, means, covariances):
        refs.append(stats.multivariate_normal.logpdf(x, mean, covariance))
    ref_time = time.time() - ref_start

    fast_start = time.time()
    results = logvariational_fn(X, means, rhos)
    fast_time = time.time() - fast_start

    print("Reference time:", ref_time)
    print("Vectorized time:", fast_time)
    print("Speedup:", ref_time / fast_time)

    refs = np.array(refs)

    print(results)
    print(refs)
    assert np.allclose(results, refs, atol=1e-2)