# Helpers shared by the fixed-margin samplers (exact.jl and sis.jl).

"""
    _margins(m)

Return the `(rowsums, colsums)` of a sparse boolean matrix, reading them directly
off the `SparseMatrixCSC` structure: column sums are the gaps between column
pointers, row sums are the occurrences of each row index.
"""
function _margins(m::SparseMatrixCSC{Bool, Int})
   colsums = diff(m.colptr)
   rowsums = zeros(Int, size(m, 1))
   @inbounds for r in m.rowval
      rowsums[r] += 1
   end
   rowsums, colsums
end

"""
    _conjugate(sizes, len)

The conjugate (Ferrers transpose) of a multiset of column sums, truncated to
`len` levels: `conjugate[k]` is the number of columns whose sum is at least `k`.
It encodes the column sums in the form the Gale–Ryser feasibility test needs.
"""
function _conjugate(sizes, len::Int)
   conjugate = zeros(Int, len)
   @inbounds for s in sizes, k in 1:min(s, len)
      conjugate[k] += 1
   end
   conjugate
end

"""
    _writecols!(m, columns)

Overwrite the row indices of each column of `m` in place from `columns`, a vector
of row-index lists. The column sums (and hence `m.colptr` and `m.nzval`) are
unchanged, so only `m.rowval` is rewritten, with the rows of each column sorted.
"""
function _writecols!(m::SparseMatrixCSC{Bool, Int}, columns)
   @inbounds for j in 1:size(m, 2)
      rows = sort!(columns[j])
      copyto!(view(m.rowval, m.colptr[j]:m.colptr[j+1]-1), rows)
   end
   m
end

"""
    _columnsmatrix(columns, nrows)

Assemble a fresh `nrows × length(columns)` sparse boolean matrix from `columns`, a
per-column vector of row-index lists. Used by samplers that build a matrix from
scratch rather than overwriting an existing one.
"""
function _columnsmatrix(columns, nrows::Int)
   ncols = length(columns)
   colptr = Vector{Int}(undef, ncols + 1)
   colptr[1] = 1
   @inbounds for j in 1:ncols
      colptr[j+1] = colptr[j] + length(columns[j])
   end
   rowval = Vector{Int}(undef, colptr[end] - 1)
   @inbounds for j in 1:ncols
      copyto!(view(rowval, colptr[j]:colptr[j+1]-1), sort!(columns[j]))
   end
   SparseMatrixCSC(nrows, ncols, colptr, rowval, fill(true, length(rowval)))
end
