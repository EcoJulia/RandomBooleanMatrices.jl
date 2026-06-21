# RandomBooleanMatrices

[![codecov](https://codecov.io/gh/EcoJulia/RandomBooleanMatrices.jl/graph/badge.svg?token=XdZckrNGI9)](https://app.codecov.io/gh/EcoJulia/RandomBooleanMatrices.jl)

## Work In Progress for a scientific publication

Create random boolean matrices that maintain row and column sums. This is a very
common use case for null models in ecology.

The package offers the newest and most efficient unbiased algorithms for generating
random matrices. Legacy approaches have used different forms of swapping, or some
alternative approaches like the `quasiswap` algorithm in R's vegan package. These
methods are neither efficient, nor are they guaranteed to sample the possible
distribution of boolean vectors with a given row and column sum equally.

The package offers three algorithms, selected with the `method` keyword. There
are two forms for each: a `randomize_matrix!` function that randomizes a sparse
boolean `Matrix` in place, and a generator form:

```julia
using SparseArrays, RandomBooleanMatrices

# in-place
m = sprand(Bool, 1000, 1000, 0.1)
randomize_matrix!(m)                    # curveball by default
randomize_matrix!(m, method = sis)     # or sis / exact

# using a Matrix generator object
m = sprand(Bool, 1000, 1000, 0.1)
rmg = matrixrandomizer(m, method = sis)
m1 = rand(rmg) # creates a new random matrix
m2 = rand(rmg)

# the curveball generator is warm-started from an independent draw, and `trades`
# sets how many trades separate successive samples (raise it to thin correlation)
rmg = matrixrandomizer(m, method = curveball, trades = 10_000)

# You can also avoid copying by
m3 = rand!(rmg)
# but notice that this will not create a new copy of the Matrix, so generating multiple matrices at once with this is impossible
```

## Methods

  * `curveball` (default) — the curveball algorithm of Strona et al. (2014). A
    fast in-place trade that performs a Markov step from the current matrix. It is
    **sequential**: each call continues the chain from where the previous one left
    off, so successive draws are *correlated*. The chain's stationary distribution
    is the uniform one, so over a run it samples fixed-margin matrices uniformly.
    The `matrixrandomizer` generator warm-starts the chain from an independent
    `sis` draw, so even the first sample is decorrelated from the input matrix, and
    the `trades` keyword sets how many trades separate successive draws — raise it
    to thin out the correlation between samples.
  * `sis` — sequential importance sampling (Harrison & Miller 2013). Each call
    draws an **independent** matrix from a fast, near-uniform approximation of the
    fixed-margin distribution. Scales to large matrices and is the natural choice
    when independent draws are wanted.
  * `exact` — exact uniform sampling (Miller & Harrison 2013). Each call draws an
    **independent**, *exactly* uniform matrix, and is feasible for small matrices
    (roughly `rows + cols ≤ 100`, or very sparse). The matrix count is computed
    once by `matrixrandomizer` and reused by every `rand`, so the generator form
    is much cheaper than `randomize_matrix!(m, method = exact)` for repeated
    sampling.

In short: `curveball` is a sequential Markov chain (correlated draws, uniform in
the limit), while `exact` and `sis` produce independent draws (exactly uniform,
resp. close to uniform) on every call. The `sis` proposal is the uniform special
case of the importance sampler of Harrison & Miller; non-uniform (weighted)
sampling is a planned extension.

## References

Strona, G., Nappo, D., Boccacci, F., Fattorini, S. & San-Miguel-Ayanz, J. (2014)
A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals.
Nature Communications, 5, 4114.

Miller, J.W. & Harrison, M.T. (2013) Exact sampling and counting for fixed-margin matrices.
The Annals of Statistics, 41, 1569-1592.

Harrison, M.T. & Miller, J.W. (2013) Importance sampling for weighted binary random matrices with specified margins.
arXiv:1301.3928.
