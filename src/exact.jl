# Exact uniform sampling (and counting) of fixed-margin binary matrices, after
#
#   Miller & Harrison (2013), "Exact sampling and counting for fixed-margin
#   matrices".
#
# Matrices are built row by row, rows taken in order of decreasing sum. Filling a
# row means distributing its ones among the columns; columns with equal residual
# sums are interchangeable, so we only track how many ones land in each *block*
# of equal-residual columns. A column's residual profile is summarised by its
# conjugate, and the number of completions depends only on that conjugate — which
# makes the recursion memoizable and turns an exponential search into a dynamic
# program that is exact and tractable for small matrices. The same table that
# counts the matrices also drives uniform sampling: at each choice we branch in
# proportion to the number of completions it leaves.

# ---------------------------------------------------------------------------
# Bundled state
# ---------------------------------------------------------------------------

"""
    ExactProblem

The dynamic-programming machinery for one set of margins: the row sums `p` sorted
descending and zero-padded, the number of `nrows`, a table of `binomial`
coefficients (choices of columns within a block), and the memo `table` mapping a
column conjugate to its number of completions.
"""
struct ExactProblem
   p::Vector{Int}
   nrows::Int
   binomial::Matrix{BigInt}
   table::Dict{Vector{Int}, BigInt}
   L::Int
end

"""
    ExactCounts

A counted problem, ready for uniform sampling: the dynamic-programming `problem`,
the `rowperm` mapping sorted rows back to their original positions, the original
`colsums`, the initial column `conjugate`, and the `total` number of matrices.
"""
struct ExactCounts
   problem::ExactProblem
   rowperm::Vector{Int}
   colsums::Vector{Int}
   conjugate::Vector{Int}
   total::BigInt
end

"""
    Cursor(row, block, placed, psum, csum)

A position in the recursion: filling `row`, currently at column `block`, with
`placed` ones already in this row. `psum` and `csum` are the running row- and
column-sum tallies the Gale–Ryser bound is computed from.
"""
struct Cursor
   row::Int
   block::Int
   placed::Int
   psum::Int
   csum::Int
end

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

"""
    _pascal(n)

Pascal's triangle of binomial coefficients up to `n` as exact integers, with
`binomial[a+1, b+1]` holding `C(a, b)`.
"""
function _pascal(n::Int)
   binomial = zeros(BigInt, n + 1, n + 1)
   for a in 0:n
      binomial[a+1, 1] = 1
      for b in 1:a
         binomial[a+1, b+1] = binomial[a, b+1] + binomial[a, b]
      end
   end
   binomial
end

# How many ones may go into the current block, and how many columns it spans.
# The lower bound keeps later blocks able to absorb the rest of the row; the
# upper bound is Gale–Ryser together with the block size and the row's remainder.
function _block_range(prob::ExactProblem, c::Vector{Int}, cur::Cursor)
   width = c[cur.block] - c[cur.block+1]
   remaining = prob.p[cur.row] - cur.placed
   smallest = max(0, remaining - c[cur.block+1])
   largest = min(width, remaining, cur.csum - cur.placed - cur.psum)
   width, smallest:largest
end

# Advance to the start of the next row once the current one is full.
function _next_row(prob::ExactProblem, c::Vector{Int}, cur::Cursor)
   Cursor(cur.row + 1, 1, 0, prob.p[cur.row+2], c[1])
end

# Advance to the next block after placing `s` ones in the current one (the caller
# has already reduced `c[block]` by `s`).
function _next_block(prob::ExactProblem, c::Vector{Int}, cur::Cursor, s::Int)
   Cursor(cur.row, cur.block + 1, cur.placed + s,
          cur.psum + prob.p[cur.row+cur.block+1], cur.csum + c[cur.block+1])
end

"""
    _count(prob, c, cur)

Number of ways to complete the matrix from position `cur` given the live column
conjugate `c`. Dispatches on whether the current row is full.
"""
function _count(prob::ExactProblem, c::Vector{Int}, cur::Cursor)
   cur.placed == prob.p[cur.row] ? _count_rows(prob, c, cur) : _count_blocks(prob, c, cur)
end

# Row finished: look the column state up in the memo, or recurse into the next
# row and store the result (skipping trivially-determined states).
function _count_rows(prob::ExactProblem, c::Vector{Int}, cur::Cursor)
   haskey(prob.table, c) && return prob.table[c]
   result = _count(prob, c, _next_row(prob, c, cur))
   c[1] != prob.p[cur.row+1] && (prob.table[copy(c)] = result)
   result
end

# Sum over the feasible numbers of ones in the current block, weighting each by
# the ways to choose its columns times the completions it leaves.
function _count_blocks(prob::ExactProblem, c::Vector{Int}, cur::Cursor)
   width, range = _block_range(prob, c, cur)
   result = zero(BigInt)
   for s in range
      c[cur.block] -= s
      result += prob.binomial[width+1, s+1] * _count(prob, c, _next_block(prob, c, cur, s))
      c[cur.block] += s
   end
   result
end

"""
    _exact_counts(rowsums, colsums)

Count the fixed-margin binary matrices and return an [`ExactCounts`](@ref) that
caches the result for repeated uniform sampling.
"""
function _exact_counts(rowsums::Vector{Int}, colsums::Vector{Int})
   L = maximum(colsums, init = 0) + 2
   conjugate = _conjugate(colsums, L)
   rowperm = sortperm(rowsums, rev = true)
   p = vcat(rowsums[rowperm], zeros(Int, L + 2))
   table = Dict{Vector{Int}, BigInt}(zeros(Int, L) => big(1))
   prob = ExactProblem(p, length(rowsums), _pascal(length(colsums)), table, L)
   c = copy(conjugate)
   total = _count(prob, c, Cursor(1, 1, 0, p[2], c[1]))
   table[copy(conjugate)] = total
   ExactCounts(prob, rowperm, colsums, conjugate, total)
end

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

"""
    _sample_plan!(plan, ex, rng)

Draw a uniform fixed-margin matrix as a `plan`, where `plan[row, block]` is the
number of ones the sorted `row` places in column-block `block`. Each branch of
the recursion is taken in proportion to the number of completions it leaves, so
the resulting matrix is exactly uniform.
"""
function _sample_plan!(plan::Matrix{Int}, ex::ExactCounts, rng)
   prob = ex.problem
   c = copy(ex.conjugate)
   fill!(plan, 0)
   _sample!(plan, prob, c, Cursor(1, 1, 0, prob.p[2], c[1]), ex.total, rng)
   plan
end

function _sample!(plan::Matrix{Int}, prob::ExactProblem, c::Vector{Int}, cur::Cursor, total::BigInt, rng)
   if cur.placed == prob.p[cur.row]
      cur.row == prob.nrows + 1 && return
      return _sample!(plan, prob, c, _next_row(prob, c, cur), total, rng)
   end
   width, range = _block_range(prob, c, cur)
   target = rand(rng, zero(total):(total - one(total)))
   cumulative = zero(BigInt)
   for s in range
      c[cur.block] -= s
      completions = _count(prob, c, _next_block(prob, c, cur, s))
      cumulative += prob.binomial[width+1, s+1] * completions
      if cumulative > target
         plan[cur.row, cur.block] = s
         _sample!(plan, prob, c, _next_block(prob, c, cur, s), completions, rng)
         c[cur.block] += s
         return
      end
      c[cur.block] += s
   end
   error("exact sampler reached an inconsistent state")
end

# Turn a sampled plan into the actual matrix: for each row and block, pick which
# of the equal-residual columns receive the row's ones, uniformly at random.
function _plan_to_columns!(columns, plan::Matrix{Int}, ex::ExactCounts, rng)
   residual = copy(ex.colsums)
   for col in columns
      empty!(col)
   end
   @inbounds for row in 1:ex.problem.nrows
      for level in 1:ex.problem.L-1
         block = findall(==(level), residual)
         for col in sample(rng, block, plan[row, level], replace = false)
            push!(columns[col], ex.rowperm[row])
            residual[col] -= 1
         end
      end
   end
   columns
end

"""
    _exact_sample!(columns, ex, rng)

Draw one uniform fixed-margin matrix into `columns` (row indices per column).
"""
function _exact_sample!(columns, ex::ExactCounts, rng)
   plan = Matrix{Int}(undef, ex.problem.nrows, ex.problem.L)
   _sample_plan!(plan, ex, rng)
   _plan_to_columns!(columns, plan, ex, rng)
end

function _exact!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG)
   rowsums, colsums = _margins(m)
   columns = [Int[] for _ in colsums]
   _exact_sample!(columns, _exact_counts(rowsums, colsums), rng)
   _writecols!(m, columns)
end

# Generator draw: the cached `ExactCounts` is the generator state, so repeated
# draws reuse the (expensive) count instead of recomputing it.
function _draw!(m::SparseMatrixCSC{Bool, Int}, rng, ex::ExactCounts)
   columns = [Int[] for _ in 1:size(m, 2)]
   _exact_sample!(columns, ex, rng)
   _writecols!(m, columns)
end
