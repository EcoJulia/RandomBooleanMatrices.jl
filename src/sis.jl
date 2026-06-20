# Sequential importance sampling of fixed-margin binary matrices, after
#
#   Harrison & Miller (2013), "Importance sampling for weighted binary random
#   matrices with specified margins".
#
# The matrix is built one column at a time. Each column is a binary vector with a
# prescribed number of ones, drawn from a fast approximation of the conditional
# distribution that (a) respects the column total, (b) keeps the remaining
# problem completable (Gale–Ryser), and (c) prefers configurations in proportion
# to a combinatorial estimate of how many completions they admit. A dynamic
# program turns those per-row preferences and feasibility bounds into exact
# forward sampling probabilities for the column. This implements the uniform case
# (every fixed-margin matrix targeted with equal weight), the natural counterpart
# to the curveball sampler.

# ---------------------------------------------------------------------------
# Combinatorial approximation of the per-row odds of a one
# ---------------------------------------------------------------------------

"""
    ColumnScorer

A combinatorial approximation that scores how likely each row is to carry a one
in the column currently being sampled. Concrete scorers are callable: given a
row's residual sum they return that row's (unconstrained) probability of a one.
New approximations (e.g. Greenhill–McKay for sparse margins) can be added as
further subtypes without touching the dynamic program.
"""
abstract type ColumnScorer end

"""
    Canfield(nrows, ncols, total, weightA)

The Canfield–Greenhill–McKay (2008) approximation used by Harrison & Miller for
near-regular margins. `ncols` and `total` describe the columns and ones that
remain *after* the current column, and `weightA` collects the curvature term of
the asymptotic enumeration so that scoring a row is a single `exp`.
"""
struct Canfield <: ColumnScorer
   nrows::Int
   ncols::Int
   total::Int
   weightA::Float64
end

function Canfield(nrows::Int, ncols::Int, total::Int, sumsq::Int)
   weightA = if total == 0 || nrows * ncols == total
      0.0
   else
      η = nrows * ncols / (total * (nrows * ncols - total))
      ν = η * (sumsq - total^2 / ncols)
      η * (1 - ν) / 2
   end
   Canfield(nrows, ncols, total, weightA)
end

# Probability that a row with residual sum `r` carries a one in this column.
# Rows that are empty (r == 0) or saturated (r == ncols + 1) fall out at 0 and 1.
function (s::Canfield)(r::Int)
   odds = r * exp(s.weightA * (1 - 2 * (r - s.total / s.nrows)))
   odds / (s.ncols + 1 - r + odds)
end

# ---------------------------------------------------------------------------
# The residual problem: margins still to be filled, evolving column by column
# ---------------------------------------------------------------------------

"""
    SISResidual

The part of the problem still to be sampled. Rows are tracked by `residual` sum
and kept in descending `order`; columns are summarised by their `conjugate`
(for feasibility) together with the running `total` of ones, sum of squares
`sumsq` (for the scorer) and number `ncols` still to place.
"""
mutable struct SISResidual
   residual::Vector{Int}    # residual row sums, indexed by original row
   order::Vector{Int}       # original row indices, sorted by descending residual
   conjugate::Vector{Int}   # conjugate of the columns not yet sampled
   total::Int               # ones still to place
   sumsq::Int               # Σ colsum² over columns not yet sampled
   ncols::Int               # columns not yet sampled
   nrows::Int
end

function SISResidual(rowsums::Vector{Int}, colsums::Vector{Int})
   nrows = length(rowsums)
   SISResidual(copy(rowsums), sortperm(rowsums, rev = true),
               _conjugate(colsums, nrows),
               sum(colsums), sum(abs2, colsums), length(colsums), nrows)
end

# Remove a column of sum `k` from the column-side statistics, leaving the
# residual describing the *other* remaining columns (what a completion must fit).
function _advance!(res::SISResidual, k::Int)
   @inbounds for level in 1:k
      res.conjugate[level] -= 1
   end
   res.total -= k
   res.sumsq -= k^2
   res.ncols -= 1
   res
end

# Decrement the rows that received a one and restore the descending order.
function _place!(res::SISResidual, rows)
   @inbounds for r in rows
      res.residual[r] -= 1
   end
   sortperm!(res.order, res.residual, rev = true)
   res
end

# ---------------------------------------------------------------------------
# Per-column dynamic program
# ---------------------------------------------------------------------------

"""
    SISWorkspace

Reusable scratch space for sampling one column: the per-row one-probabilities
`p`, the feasibility band `lo:hi` on the partial column sum after each row, the
forward transition probabilities `S`, and two rolling buffers for the backward
dynamic-programming weights.
"""
struct SISWorkspace
   p::Vector{Float64}
   lo::Vector{Int}
   hi::Vector{Int}
   S::Matrix{Float64}
   gcur::Vector{Float64}
   gnext::Vector{Float64}
end

SISWorkspace(nrows::Int, maxcol::Int) =
   SISWorkspace(Vector{Float64}(undef, nrows), Vector{Int}(undef, nrows),
                Vector{Int}(undef, nrows), Matrix{Float64}(undef, maxcol + 1, nrows),
                Vector{Float64}(undef, maxcol + 2), Vector{Float64}(undef, maxcol + 2))

# Score every row for the current column.
function _score!(ws::SISWorkspace, res::SISResidual, scorer::ColumnScorer)
   @inbounds for i in 1:res.nrows
      ws.p[i] = scorer(res.residual[res.order[i]])
   end
   ws
end

# Feasibility band for the partial column sum s_i = x_1 + … + x_i (rows in
# descending order). Gale–Ryser bounds it below; the column total and row count
# bound it above. `res.conjugate` is the conjugate of the *other* columns here.
function _band!(ws::SISWorkspace, res::SISResidual, k::Int)
   m = res.nrows
   rowsum = 0     # prefix of residual row sums
   conjsum = 0    # prefix of the conjugate of the other columns
   lo = 0
   @inbounds for i in 1:m
      rowsum += res.residual[res.order[i]]
      conjsum += res.conjugate[i]
      lo = max(lo, rowsum - conjsum, k - (m - i))
      ws.lo[i] = lo
      ws.hi[i] = min(k, i)
   end
   ws
end

# Backward dynamic program (Harrison & Geman 2009): turn the per-row odds and the
# feasibility band into forward transition probabilities. `g[s]` is the total
# weight of feasible completions reaching column total `k` from partial sum `s`;
# `S[s+1, i]` is the resulting probability that row `i` is a one given partial sum
# `s` before it.
function _transitions!(ws::SISWorkspace, k::Int)
   m = length(ws.p)
   gnext, gcur, S = ws.gnext, ws.gcur, ws.S
   fill!(gnext, 0.0)
   gnext[k+1] = 1.0                       # base case: all rows placed, sum == k
   @inbounds for i in m:-1:1
      lo = i == 1 ? 0 : ws.lo[i-1]        # band on the partial sum before row i
      hi = i == 1 ? 0 : ws.hi[i-1]
      p = ws.p[i]
      fill!(gcur, 0.0)
      total = 0.0
      for s in lo:hi
         w1 = p * gnext[s+2]              # row i is a one  -> partial sum s+1
         w0 = (1 - p) * gnext[s+1]        # row i is a zero -> partial sum s
         weight = w0 + w1
         gcur[s+1] = weight
         S[s+1, i] = weight > 0 ? w1 / weight : 0.0
         total += weight
      end
      total > 0 && (@views gcur[lo+1:hi+1] ./= total)   # rescale to avoid underflow
      gnext, gcur = gcur, gnext
   end
   ws
end

# Forward pass: walk the rows, drawing each from its transition probability, and
# accumulate the log-probability of the column under the proposal.
function _draw_column!(rows, ws::SISWorkspace, order, k::Int, rng)
   s = 0
   logq = 0.0
   @inbounds for i in eachindex(ws.p)
      p = ws.S[s+1, i]
      if rand(rng) < p
         push!(rows, order[i])
         s += 1
         logq += log(p)
         s == k && break
      else
         logq += log(1 - p)
      end
   end
   logq
end

"""
    _sample_column!(rows, res, k, ws, rng)

Sample one column of total `k` into `rows` (original row indices), update the
residual problem, and return the log proposal probability of the draw.
"""
function _sample_column!(rows, res::SISResidual, k::Int, ws::SISWorkspace, rng)
   k == 0 && return 0.0
   _advance!(res, k)
   _score!(ws, res, Canfield(res.nrows, res.ncols, res.total, res.sumsq))
   _band!(ws, res, k)
   _transitions!(ws, k)
   logq = _draw_column!(rows, ws, res.order, k, rng)
   _place!(res, rows)
   logq
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    _sis(rowsums, colsums, rng)

Sample a fixed-margin binary matrix by sequential importance sampling, returning
the row indices of each column together with the log proposal probability of the
whole matrix. Columns are visited in order of decreasing sum, which the authors
find improves the approximation.
"""
function _sis(rowsums::Vector{Int}, colsums::Vector{Int}, rng)
   res = SISResidual(rowsums, colsums)
   ws = SISWorkspace(res.nrows, maximum(colsums, init = 0))
   columns = [Int[] for _ in colsums]
   logq = 0.0
   for j in sortperm(colsums, rev = true)
      logq += _sample_column!(columns[j], res, colsums[j], ws, rng)
   end
   columns, logq
end

function _sis!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG)
   rowsums, colsums = _margins(m)
   columns, _ = _sis(rowsums, colsums, rng)
   _writecols!(m, columns)
end
