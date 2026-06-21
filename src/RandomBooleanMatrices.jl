module RandomBooleanMatrices

using Random
using RandomNumbers.Xorshifts
using SparseArrays
using StatsBase

include("common.jl")
include("curveball.jl")
include("exact.jl")
include("weights.jl")
include("sis.jl")

@enum matrixrandomizations curveball exact sis

"""
    randomize_matrix!(m [,rng]; method = curveball, trades = 5 * size(m, 2))

Randomize the sparse boolean Matrix `m` in place while maintaining its row and
column sums. The algorithm is chosen with `method`:

  * `curveball` — the curveball trade of Strona et al. (2014). Applies `trades`
    trades, each a Markov step from the current matrix; repeated calls explore the
    space of fixed-margin matrices.
  * `sis` — sequential importance sampling (Harrison & Miller 2013). Draws an
    independent matrix from a fast, near-uniform approximation. Suited to large
    matrices.
  * `exact` — exact uniform sampling (Miller & Harrison 2013). Draws an
    independent, exactly uniform matrix; feasible for small matrices. The counting
    step is repeated on every call, so prefer [`matrixrandomizer`](@ref), which
    counts once and reuses the result.

`trades` applies only to `curveball` and sets how far the chain moves per call.
"""
function randomize_matrix!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG; method::matrixrandomizations = curveball, trades::Int = 5 * size(m, 2))
    method == curveball && return _curveball!(m, rng, trades)
    method == sis       && return _sis!(m, rng)
    method == exact     && return _exact!(m, rng)
    error("undefined method")
end

# A generator stores a method-specific sampler that both selects the draw
# algorithm (by dispatch, via `_draw!`) and carries whatever that method reuses
# across draws.
const GeneratorState = Union{CurveballSampler, SISSampler, ExactCounts}

struct MatrixGenerator{R<:AbstractRNG, M}
    m::M
    rng::R
    state::GeneratorState
end

Base.show(io::IO, m::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = print(io, "Boolean MatrixGenerator with size $(size(m.m)) and $(nnz(m.m)) occurrences")

# Build the sampler for `method`, performing any one-off preparation of `sm`:
# `exact` caches its (expensive) matrix count, and `curveball` warm-starts the
# stored matrix from an independent SIS draw so the chain begins in the bulk of
# the distribution rather than at the input, then records its per-draw `trades`.
function _sampler!(method::matrixrandomizations, sm::SparseMatrixCSC{Bool, Int}, rng, trades)
    method == sis   && return SISSampler()
    method == exact && return _exact_counts(_margins(sm)...)
    if method == curveball
        _sis!(sm, rng)
        return CurveballSampler(something(trades, 5 * size(sm, 2)))
    end
    error("undefined method")
end

"""
    matrixrandomizer(m [,rng]; method = curveball, trades = 5 * size(m, 2))

Create a matrix generator that returns a random boolean matrix every time it is
called, maintaining row and column sums. Non-boolean input matrices are
interpreted as boolean, where values != 0 are `true`. See [`randomize_matrix!`](@ref)
for the available `method`s.

For `method = exact` the matrix count is computed once here and reused by every
draw. For `method = curveball` the chain is warm-started from an independent `sis`
draw, so even the first sample is decorrelated from `m`; `trades` then sets how
many curveball trades separate successive draws — raise it to reduce correlation
between samples.

# Examples
```
m = rand(0:4, 5, 6)
rmg = matrixrandomizer(m)

random1 = rand(rmg)
random2 = rand(rmg)
``
"""
matrixrandomizer(m, rng) = error("No matrixrandomizer defined for $(typeof(m))")
matrixrandomizer(m) = error("No matrixrandomizer defined for $(typeof(m))")
function matrixrandomizer(m::AbstractMatrix, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball, trades = nothing)
    sm = SparseMatrixCSC{Bool, Int}(dropzeros!(sparse(m)))
    MatrixGenerator{typeof(rng), SparseMatrixCSC{Bool, Int}}(sm, rng, _sampler!(method, sm, rng, trades))
end
function matrixrandomizer(m::SparseMatrixCSC{Bool, Int}, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball, trades = nothing)
    sm = dropzeros(m)
    MatrixGenerator{typeof(rng), SparseMatrixCSC{Bool, Int}}(sm, rng, _sampler!(method, sm, rng, trades))
end

# A single draw from the generator, in place in its stored matrix; the algorithm
# is selected by the type of the stored sampler state.
_generate!(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = _draw!(r.m, r.rng, r.state)

Random.rand(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = copy(_generate!(r))
Random.rand!(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = _generate!(r)

"""
    WeightedSIS

A sequential importance sampler for the weighted fixed-margin target
P*(z) ∝ ∏ w[i,j]^z[i,j]. Built by [`importance_sampler`](@ref); each `rand` draws
an independent matrix and returns it with the log of its importance weight.
"""
struct WeightedSIS{R<:AbstractRNG}
    rowsums::Vector{Int}
    colsums::Vector{Int}
    model::WeightMatrix
    rng::R
end

Base.show(io::IO, s::WeightedSIS) =
    print(io, "WeightedSIS sampler for a $(length(s.rowsums))×$(length(s.colsums)) fixed-margin matrix")

"""
    importance_sampler(m, w [,rng])

Create a sequential importance sampler (Harrison & Miller 2013) for the weighted
distribution over binary matrices with the row and column sums of `m`,

    P*(z) ∝ ∏ w[i,j]^z[i,j],

where `w` is a matrix of nonnegative weights the size of `m`; the uniform special
case is `w` all ones. Each `rand` returns a `(matrix, logweight)` pair: an
independent draw and the log of its importance weight `∏ w^z / Q*(z)`. Monte-Carlo
estimates reweight by `exp(logweight)` — for example `mean(exp(logweight))`
estimates the normalising constant `κ = Σ_z ∏ w^z`, and
`sum(exp(logweight) .* h) / sum(exp(logweight))` estimates `E[h(Z)]` under P*.

A zero weight `w[i,j] == 0` is a structural zero: it forbids a one at that
position. A draw the zeros leave no way to complete is returned with
`logweight == -Inf` (importance weight zero) and should simply be discarded.
The proposal is built from the Sinkhorn-canonical form of `w` unless
`canonicalize = false`; this never changes the importance weights, only the
sampler's efficiency.

# Examples
```
m = sprand(Bool, 20, 15, 0.3)
w = rand(20, 15) .+ 0.5
sampler = importance_sampler(m, w)
draw = rand(sampler)
draw.matrix      # an independent fixed-margin sample
draw.logweight   # its log importance weight
```
"""
function importance_sampler(m::AbstractMatrix, w::AbstractMatrix, rng = Xoroshiro128Plus();
                            canonicalize::Bool = true)
    size(w) == size(m) || throw(DimensionMismatch("weights `w` must match the size of `m`"))
    sm = SparseMatrixCSC{Bool, Int}(dropzeros!(sparse(m)))
    rowsums, colsums = _margins(sm)
    WeightedSIS(rowsums, colsums, WeightMatrix(w, rowsums, colsums; canonicalize), rng)
end

function Random.rand(s::WeightedSIS)
    columns, logq, logp = _sis(s.rowsums, s.colsums, s.model, s.rng)
    (matrix = _columnsmatrix(columns, length(s.rowsums)), logweight = logp - logq)
end

export randomize_matrix!, matrixrandomizer, matrixrandomizations, importance_sampler
export curveball, exact, sis

end
