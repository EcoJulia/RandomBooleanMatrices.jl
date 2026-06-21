module RandomBooleanMatrices

using Random
using RandomNumbers.Xorshifts
using SparseArrays
using StatsBase

include("common.jl")
include("curveball.jl")
include("exact.jl")
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

export randomize_matrix!, matrixrandomizer, matrixrandomizations
export curveball, exact, sis

end
