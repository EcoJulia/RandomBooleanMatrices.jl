module RandomBooleanMatrices

include("curveball.jl")

@enum matrixrandomizations curveball

"""
    randomize_matrix!(m; method = curveball)

Randomize the sparse boolean Matrix `m` while maintaining row and column sums
"""
function randomize_matrix!(m; method::matrixrandomizations = curveball)
    if method == curveball
        return _curveball!(m)
    end
    error("undefined method")
end

struct MatrixGenerator
    m::SparseMatrixCSC{Bool, Int}
    method::methods
end

"""
    random_matrices(m; method = curveball)

Create a matrix generator function that will return a random boolean matrix
every time it is called, maintaining row and column sums. Non-boolean input
matrix are interpreted as boolean, where values != 0 are `true`.

# Examples
```
m = rand(0:4, 5, 6)
rmg = random_matrices(m)

random1 = rmg()
random2 = rmg()
``
"""
random_matrices(m::AbstractMatrix; method::matrixrandomizations = curveball) =
    MatrixGenerator(m, method = method) #for this case there's an implicit `copy` already
random_matrices(m::SparseMatrixCSC{Bool, Int}; method::matrixrandomizations = curveball) =
    MatrixGenerator(copy(m), method = method)

(r::MatrixGenerator)(; method::matrixrandomizations = curveball) = randomize_matrix!(r.m, method = r.method)

export randomize_matrix, random_matrices, matrixrandomizations
end
