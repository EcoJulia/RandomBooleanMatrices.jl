# Weighted targets for sequential importance sampling.
#
# The uniform sampler in sis.jl targets every fixed-margin matrix equally.
# Harrison & Miller (2013) generalise this to the weighted distribution
#
#     P*(z) ∝ ∏_{ij} w[i,j]^z[i,j]    over fixed-margin binary z,
#
# of which the uniform case (w ≡ 1) is a special case. The generalisation only
# changes the per-row odds of a one in the column being sampled: those odds are
# multiplied by a factor vᵢ built from the row's remaining weights (eq. 15). The
# factor uses the elementary symmetric polynomials of the weights over the
# not-yet-sampled columns, all carried in logs to keep their range in check.
#
# A draw is then no longer (near-)uniform, so it carries an importance weight
# f(z) = ∏ w^z / Q*(z); callers reweight Monte-Carlo estimates by it.
#
# Two refinements from the paper are included. Structural zeros (w[i,j] == 0, which
# forbid a one at that position) are handled: the symmetric polynomials and vᵢ drop
# those positions automatically (a zero weight is log -∞), and a draw that the
# zeros leave no way to complete is rejected with importance weight zero. And the
# weights are Sinkhorn-balanced into the canonical w̄ ∈ Λ(w) of eq. (16), which P*
# is invariant to but which makes the proposal scale-free and lower-variance.
# (Not yet done: the paper's variance-based column ordering.)

"""
    WeightModel

How the SIS proposal biases a one beyond the combinatorial [`ColumnScorer`].
`UniformWeights` adds nothing (the uniform target); `WeightMatrix` carries an
entrywise weight matrix and the factors needed to bias the proposal toward it.
New targets dispatch [`_weightfactor`](@ref) and [`_logweight`](@ref) here
without touching the dynamic program.
"""
abstract type WeightModel end

struct UniformWeights <: WeightModel end

"""
    WeightMatrix(w, rowsums, colsums; canonicalize = true)

The target P*(z) ∝ ∏ w[i,j]^z[i,j], for nonnegative weights `w` (zeros forbid a
one at that position). Stores the log weights (for the importance weight) and,
with columns in sampling order, the per-row count of available (positive)
positions and the log elementary symmetric polynomials
`loge[k+1, i, pos] = log eₖ(positive weights of row i from position pos onward)`,
from which the per-row factor vᵢ is read. The proposal is built from the
Sinkhorn-canonical `w̄` unless `canonicalize = false`; the importance weight always
uses the original `w`.
"""
struct WeightMatrix <: WeightModel
   logw::Matrix{Float64}      # log original weights [row, col]    (the importance weight)
   logwbar::Matrix{Float64}   # log canonical weights [row, pos]   (columns in sampling order)
   navail::Matrix{Int}        # navail[row, pos] = #positive weights at positions pos:end
   order::Vector{Int}         # column sampling order (column labels)
   loge::Array{Float64, 3}    # loge[k+1, row, pos] = log eₖ(weights at positions pos:end)
end

function WeightMatrix(w::AbstractMatrix, rowsums::Vector{Int}, colsums::Vector{Int};
                      canonicalize::Bool = true)
   all(>=(0), w) || throw(ArgumentError("weights must be nonnegative"))
   any(>(0), w)  || throw(ArgumentError("weights must have a positive entry"))
   logw = Matrix{Float64}(log.(float.(w)))                  # original weights, for the weight
   wbar = canonicalize ? _canonical(w) : Matrix{Float64}(float.(w))
   order = sortperm(colsums, rev = true)
   logwbar = log.(wbar)[:, order]                           # proposal weights, in sampling order
   navail = _suffixcount(logwbar)
   loge = _logesym(logwbar, maximum(rowsums, init = 0))
   WeightMatrix(logw, logwbar, navail, order, loge)
end

# log(exp(a) + exp(b)), with -Inf absorbing as the log of zero.
function _logaddexp(a::Float64, b::Float64)
   a == -Inf && return b
   b == -Inf && return a
   m = max(a, b)
   m + log1p(exp(-abs(a - b)))
end

# Number of positive weights in each row over column suffixes: `navail[i, pos]`
# counts positions pos:n. A weight is positive exactly when its log is finite.
function _suffixcount(logwbar::Matrix{Float64})
   m, n = size(logwbar)
   navail = zeros(Int, m, n + 1)
   @inbounds for pos in n:-1:1, i in 1:m
      navail[i, pos] = navail[i, pos+1] + (logwbar[i, pos] > -Inf)
   end
   navail
end

# Elementary symmetric polynomials of each row's weights over column suffixes, in
# logs. Built from the empty suffix backwards using eₖ(j:n) = eₖ(j+1:n) +
# w[j]·eₖ₋₁(j+1:n); `loge[k+1, i, pos]` covers columns at positions pos:n. A zero
# weight (log -∞) contributes nothing, so structural zeros drop out naturally.
function _logesym(logwbar::Matrix{Float64}, maxk::Int)
   m, n = size(logwbar)
   loge = fill(-Inf, maxk + 1, m, n + 1)
   loge[1, :, :] .= 0.0                         # e₀ ≡ 1 over any suffix
   @inbounds for pos in n:-1:1, i in 1:m
      lw = logwbar[i, pos]
      for k in 1:maxk
         loge[k+1, i, pos] = _logaddexp(loge[k+1, i, pos+1], lw + loge[k, i, pos+1])
      end
   end
   loge
end

"""
    _canonical(w; tol = 1e-10, maxiter = 1000)

Sinkhorn-balance `w` into the canonical `w̄ ∈ Λ(w)` of Harrison & Miller (eq. 16):
the scaling `w̄ = αᵢ βⱼ wᵢⱼ` whose row and column sums equal the number of positive
entries in that row/column. `P*` is invariant to this scaling, but it makes the
proposal scale-invariant and tends to reduce its variance.
"""
function _canonical(w::AbstractMatrix; tol::Float64 = 1e-10, maxiter::Int = 1000)
   m, n = size(w)
   wbar = Matrix{Float64}(float.(w))
   ni = [count(>(0), view(wbar, i, :)) for i in 1:m]      # positive entries per row
   mj = [count(>(0), view(wbar, :, j)) for j in 1:n]      # positive entries per column
   for _ in 1:maxiter
      for i in 1:m                                        # scale each row to sum nᵢ
         s = sum(view(wbar, i, :))
         s > 0 || continue
         f = ni[i] / s
         @inbounds for j in 1:n
            wbar[i, j] *= f
         end
      end
      drift = 0.0
      for j in 1:n                                        # scale each column to sum mⱼ
         s = sum(view(wbar, :, j))
         s > 0 || continue
         drift = max(drift, abs(s - mj[j]))
         f = mj[j] / s
         @inbounds for i in 1:m
            wbar[i, j] *= f
         end
      end
      drift < tol && break
   end
   wbar
end

# ---------------------------------------------------------------------------
# Interface consumed by the sampler (sis.jl). Plain-Int arguments keep these
# independent of the residual/workspace types.
# ---------------------------------------------------------------------------

# Order in which to visit the columns (by decreasing sum).
_columnorder(::UniformWeights, colsums) = sortperm(colsums, rev = true)
_columnorder(model::WeightMatrix, colsums) = model.order

# Factor multiplying a row's odds of a one in the column at sampling position
# `pos`, given the row's residual sum `ρ`. `0` forbids a one (structural zero);
# `Inf` forces one (the remaining positive weights leave no alternative).
_weightfactor(::UniformWeights, row, ρ, pos) = 1.0
function _weightfactor(model::WeightMatrix, row, ρ, pos)
   ρ == 0 && return 1.0
   logw = model.logwbar[row, pos]
   logw == -Inf && return 0.0                   # structural zero: no one here
   logden = model.loge[ρ+1, row, pos+1]         # log eρ over the columns after pos
   logden == -Inf && return Inf                 # eρ == 0: the row is forced to a one here
   lognum = model.loge[ρ, row, pos+1]           # log eρ₋₁
   (model.navail[row, pos+1] - ρ + 1) / ρ * exp(logw + lognum - logden)
end

# Contribution of a placed one at (row, column) to log ∏ w^z.
_logweight(::UniformWeights, row, label) = 0.0
_logweight(model::WeightMatrix, row, label) = model.logw[row, label]
