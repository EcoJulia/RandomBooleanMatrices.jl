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
   pos::Vector{Int}         # inverse of `order`: where each row sits within it
   conjugate::Vector{Int}   # conjugate of the columns not yet sampled
   total::Int               # ones still to place
   sumsq::Int               # Σ colsum² over columns not yet sampled
   ncols::Int               # columns not yet sampled
   nrows::Int
end

function SISResidual(rowsums::Vector{Int}, colsums::Vector{Int})
   nrows = length(rowsums)
   order = sortperm(rowsums, rev = true)
   SISResidual(copy(rowsums), order, invperm(order), _conjugate(colsums, nrows),
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

# Decrement the rows that received a one and restore the descending order. Each
# of those rows dropped by exactly one, so it only has to slide past the block of
# rows still holding its previous value — an O(ones placed) repair rather than a
# full re-sort. Repairing the lowest-valued rows first keeps every swap local.
function _place!(res::SISResidual, rows)
   order, pos, residual = res.order, res.pos, res.residual
   @inbounds for r in rows
      residual[r] -= 1
   end
   @inbounds for idx in lastindex(rows):-1:firstindex(rows)
      r = rows[idx]
      i = pos[r]
      value = residual[r]
      j = i
      while j < res.nrows && residual[order[j+1]] > value
         j += 1
      end
      if j > i
         displaced = order[j]
         order[i], order[j] = displaced, r
         pos[r], pos[displaced] = j, i
      end
   end
   res
end

# ---------------------------------------------------------------------------
# Per-column dynamic program
# ---------------------------------------------------------------------------

"""
    SISWorkspace

Reusable scratch space for sampling one column: the per-row probability `p` of a
one and the (possibly weighted) `one`-weight that biases it, the feasibility band
`lo:hi` on the partial column sum after each row, the forward transition
probabilities `S`, and two rolling buffers for the backward dynamic-programming
weights. For the uniform target `one == p`; a weight model scales `one` by its
per-row factor while leaving the zero-weight `1 - p` alone.
"""
struct SISWorkspace
   p::Vector{Float64}
   one::Vector{Float64}
   lo::Vector{Int}
   hi::Vector{Int}
   S::Matrix{Float64}
   gcur::Vector{Float64}
   gnext::Vector{Float64}
end

SISWorkspace(nrows::Int, maxcol::Int) =
   SISWorkspace(Vector{Float64}(undef, nrows), Vector{Float64}(undef, nrows),
                Vector{Int}(undef, nrows), Vector{Int}(undef, nrows),
                Matrix{Float64}(undef, maxcol + 1, nrows),
                Vector{Float64}(undef, maxcol + 2), Vector{Float64}(undef, maxcol + 2))

# Score every row for the current column: `p` is the combinatorial probability of
# a one, and `one` is that probability scaled by the weight model's factor (so the
# odds of a one become p·v : 1-p). A weight factor of `Inf` flags a row the
# remaining weights force to a one, which we encode as p = 1.
function _score!(ws::SISWorkspace, res::SISResidual, scorer::ColumnScorer,
                 model::WeightModel, pos::Int)
   @inbounds for i in 1:res.nrows
      row = res.order[i]
      ρ = res.residual[row]
      p = scorer(ρ)
      v = _weightfactor(model, row, ρ, pos)
      if isinf(v)
         ws.p[i] = 1.0
         ws.one[i] = 1.0
      else
         ws.p[i] = p
         ws.one[i] = p * v
      end
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
# `s` before it. Returns `false` if the column has no feasible completion — only
# possible when structural zeros forbid enough ones — so the caller can reject.
function _transitions!(ws::SISWorkspace, k::Int)
   m = length(ws.p)
   gnext, gcur, S = ws.gnext, ws.gcur, ws.S
   fill!(gnext, 0.0)
   fill!(gcur, 0.0)
   gnext[k+1] = 1.0                       # base case: all rows placed, sum == k
   lonext, hinext = k + 1, k + 1          # index range where gnext is nonzero
   locur, hicur = 1, 0                    # gcur is empty (all zero)
   @inbounds for i in m:-1:1
      lo = i == 1 ? 0 : ws.lo[i-1]        # band on the partial sum before row i
      hi = i == 1 ? 0 : ws.hi[i-1]
      one = ws.one[i]                     # weight of a one (odds one : zero = one : 1-p)
      zero = 1 - ws.p[i]                  # weight of a zero
      for t in locur:hicur               # clear only gcur's stale band
         gcur[t] = 0.0
      end
      total = 0.0
      for s in lo:hi
         w1 = one * gnext[s+2]           # row i is a one  -> partial sum s+1
         w0 = zero * gnext[s+1]          # row i is a zero -> partial sum s
         weight = w0 + w1
         gcur[s+1] = weight
         S[s+1, i] = weight > 0 ? w1 / weight : 0.0
         total += weight
      end
      total > 0 || return false          # no completion: a structural-zero dead-end
      invtotal = inv(total)              # rescale to avoid underflow
      for s in lo:hi
         gcur[s+1] *= invtotal
      end
      locur, hicur = lo + 1, hi + 1
      gnext, gcur = gcur, gnext
      lonext, hinext, locur, hicur = locur, hicur, lonext, hinext
   end
   true
end

# Forward pass: walk the rows, drawing each from its transition probability.
# Accumulate the log proposal probability `logq` of the column and, for a weighted
# target, the log target weight `logp = Σ log w` over the ones placed.
function _draw_column!(rows, ws::SISWorkspace, order, k::Int, model::WeightModel, label::Int, rng)
   s = 0
   logq = 0.0
   logp = 0.0
   @inbounds for i in eachindex(ws.p)
      p = ws.S[s+1, i]
      if rand(rng) < p
         row = order[i]
         push!(rows, row)
         s += 1
         logq += log(p)
         logp += _logweight(model, row, label)
         s == k && break
      else
         logq += log(1 - p)
      end
   end
   logq, logp
end

"""
    _sample_column!(rows, res, k, ws, model, pos, label, rng)

Sample the column with label `label` (at sampling position `pos`) and total `k`
into `rows` (original row indices), update the residual problem, and return
`(logq, logp, feasible)`: the log proposal probability, the log target weight, and
whether a completion existed (`false` is a structural-zero dead-end to reject).
"""
function _sample_column!(rows, res::SISResidual, k::Int, ws::SISWorkspace,
                         model::WeightModel, pos::Int, label::Int, rng)
   k == 0 && return 0.0, 0.0, true
   _advance!(res, k)
   _score!(ws, res, Canfield(res.nrows, res.ncols, res.total, res.sumsq), model, pos)
   _band!(ws, res, k)
   _transitions!(ws, k) || return 0.0, 0.0, false
   logq, logp = _draw_column!(rows, ws, res.order, k, model, label, rng)
   _place!(res, rows)
   logq, logp, true
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    _sis(rowsums, colsums, model, rng)

Sample a fixed-margin binary matrix by sequential importance sampling under the
target described by `model` (uniform, or a [`WeightMatrix`]). Returns the row
indices of each column with the log proposal probability `logq` and log target
weight `logp` of the whole matrix; the importance weight of the draw is
`exp(logp - logq)`. Columns are visited in the order the model prescribes (by
decreasing sum), which the authors find improves the approximation. With
structural zeros a draw can dead-end; it is returned with `logq = Inf` so that its
importance weight `exp(logp - logq)` is zero.
"""
function _sis(rowsums::Vector{Int}, colsums::Vector{Int}, model::WeightModel, rng)
   res = SISResidual(rowsums, colsums)
   ws = SISWorkspace(res.nrows, maximum(colsums, init = 0))
   columns = [Int[] for _ in colsums]
   logq = 0.0
   logp = 0.0
   for (pos, label) in enumerate(_columnorder(model, colsums))
      dlogq, dlogp, feasible = _sample_column!(columns[label], res, colsums[label], ws, model, pos, label, rng)
      feasible || return columns, Inf, logp
      logq += dlogq
      logp += dlogp
   end
   columns, logq, logp
end

_sis(rowsums::Vector{Int}, colsums::Vector{Int}, rng) =
   _sis(rowsums, colsums, UniformWeights(), rng)

function _sis!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG)
   rowsums, colsums = _margins(m)
   columns, _, _ = _sis(rowsums, colsums, UniformWeights(), rng)
   _writecols!(m, columns)
end

"""
    SISSampler()

Generator state for the SIS method. SIS draws are independent, so the generator
keeps no precomputed state — each draw resamples the stored matrix from its
margins.
"""
struct SISSampler end

_draw!(m::SparseMatrixCSC{Bool, Int}, rng, ::SISSampler) = _sis!(m, rng)
