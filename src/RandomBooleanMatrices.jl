module RandomBooleanMatrices

using Random
using RandomNumbers.Xorshifts
using SparseArrays
using StatsBase

include("curveball.jl")

@enum matrixrandomizations curveball

"""
    randomize_matrix!(m; method = curveball)

Randomize the sparse boolean Matrix `m` while maintaining row and column sums
"""
function randomize_matrix!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG; method::matrixrandomizations = curveball)
    if method == curveball
        return _curveball!(m, rng)
    end
    error("undefined method")
end

struct MatrixGenerator{R<:AbstractRNG}
    m::SparseMatrixCSC{Bool, Int}
    method::matrixrandomizations
    rng::R
end

show(io::IO, m::MatrixGenerator) = println(io, "Boolean MatrixGenerator with size $(size(m.m)) and $(nnz(m.m)) occurrences")

"""
    matrixgenerator(m [,rng]; method = curveball)

Create a matrix generator function that will return a random boolean matrix
every time it is called, maintaining row and column sums. Non-boolean input
matrix are interpreted as boolean, where values != 0 are `true`.

# Examples
```
m = rand(0:4, 5, 6)
rmg = matrixgenerator(m)

random1 = rmg()
random2 = rmg()
``
"""
matrixgenerator(m::AbstractMatrix, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball) =
    MatrixGenerator{typeof(rng)}(dropzeros!(sparse(m)), method, rng)
matrixgenerator(m::SparseMatrixCSC{Bool, Int}, rng = Xoroshiro128Plus(); method::matrixrandomizations = curveball) =
    MatrixGenerator{typeof(rng)}(dropzeros(m), method, rng)

Random.rand(r::MatrixGenerator; method::matrixrandomizations = curveball) = copy(randomize_matrix!(r.m, r.rng, method = r.method))
Random.rand!(r::MatrixGenerator; method::matrixrandomizations = curveball) = randomize_matrix!(r.m, r.rng, method = r.method)

export randomize_matrix!, matrixgenerator
export curveball

end
