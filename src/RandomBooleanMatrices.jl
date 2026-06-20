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
    randomize_matrix!(m [,rng]; method = curveball)

Randomize the sparse boolean Matrix `m` in place while maintaining its row and
column sums. The algorithm is chosen with `method`:

  * `curveball` — the curveball trade of Strona et al. (2014). A Markov step from
    the current matrix; repeated calls explore the space of fixed-margin matrices.
  * `sis` — sequential importance sampling (Harrison & Miller 2013). Draws an
    independent matrix from a fast, near-uniform approximation. Suited to large
    matrices.
  * `exact` — exact uniform sampling (Miller & Harrison 2013). Draws an
    independent, exactly uniform matrix; feasible for small matrices. The counting
    step is repeated on every call, so prefer [`matrixrandomizer`](@ref), which
    counts once and reuses the result.
"""
function randomize_matrix!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG; method::matrixrandomizations = curveball)
    method == curveball && return _curveball!(m, rng)
    method == sis       && return _sis!(m, rng)
    method == exact     && return _exact!(m, rng)
    error("undefined method")
end

struct MatrixGenerator{R<:AbstractRNG, M}
    m::M
    method::matrixrandomizations
    rng::R
    cache::Union{Nothing, ExactCounts}
end

show(io::IO, m::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = println(io, "Boolean MatrixGenerator with size $(size(m.m)) and $(nnz(m.m)) occurrences")

# Precompute whatever a method reuses across draws. Only the exact method needs
# it: the (potentially expensive) matrix count is cached for repeated sampling.
_cache(m::SparseMatrixCSC{Bool, Int}, method::matrixrandomizations) =
    method == exact ? _exact_counts(_margins(m)...) : nothing

"""
    matrixrandomizer(m [,rng]; method = curveball)

Create a matrix generator that returns a random boolean matrix every time it is
called, maintaining row and column sums. Non-boolean input matrices are
interpreted as boolean, where values != 0 are `true`. See [`randomize_matrix!`](@ref)
for the available `method`s; for `method = exact` the matrix count is computed
once here and reused by every draw.

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
function matrixrandomizer(m::AbstractMatrix, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball)
    sm = SparseMatrixCSC{Bool, Int}(dropzeros!(sparse(m)))
    MatrixGenerator{typeof(rng), SparseMatrixCSC{Bool, Int}}(sm, method, rng, _cache(sm, method))
end
function matrixrandomizer(m::SparseMatrixCSC{Bool, Int}, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball)
    sm = dropzeros(m)
    MatrixGenerator{typeof(rng), SparseMatrixCSC{Bool, Int}}(sm, method, rng, _cache(sm, method))
end

# A single draw from the generator, in place in its stored matrix. The exact
# method samples from the cached count; the others delegate to randomize_matrix!.
function _generate!(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R
    r.method == exact || return randomize_matrix!(r.m, r.rng, method = r.method)
    columns = [Int[] for _ in 1:size(r.m, 2)]
    _exact_sample!(columns, r.cache, r.rng)
    _writecols!(r.m, columns)
end

Random.rand(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = copy(_generate!(r))
Random.rand!(r::MatrixGenerator{R, SparseMatrixCSC{Bool, Int}}) where R = _generate!(r)

export randomize_matrix!, matrixrandomizer, matrixrandomizations
export curveball, exact, sis

end
