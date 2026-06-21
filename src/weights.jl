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
# f(z) = ∏ w^z / Q*(z); callers reweight Monte-Carlo estimates by it. Structural
# zeros (w[i,j] == 0), weight canonicalisation, and the variance-based column
# ordering of the paper are not handled here yet (see README); none affect the
# correctness of the importance weights, only the efficiency of the sampler.

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
    WeightMatrix(w, rowsums, colsums)

The target P*(z) ∝ ∏ w[i,j]^z[i,j]. Stores the log weights (for the importance
weight), and, with columns put in sampling order, the log elementary symmetric
polynomials `loge[k+1, i, pos] = log eₖ(w[i, columns from pos onward])` from which
the per-row factor vᵢ is read. Requires strictly positive weights.
"""
struct WeightMatrix <: WeightModel
   logw::Matrix{Float64}      # log original weights [row, col]   (the importance weight)
   logwbar::Matrix{Float64}   # log weights [row, position]       (columns in sampling order)
   order::Vector{Int}         # column sampling order (column labels)
   loge::Array{Float64, 3}    # loge[k+1, row, pos] = log eₖ(weights at positions pos:end)
end

function WeightMatrix(w::AbstractMatrix, rowsums::Vector{Int}, colsums::Vector{Int})
   all(>(0), w) || throw(ArgumentError("weighted SIS currently requires strictly positive weights"))
   logw = Matrix{Float64}(log.(float.(w)))
   order = sortperm(colsums, rev = true)
   logwbar = logw[:, order]
   loge = _logesym(logwbar, maximum(rowsums, init = 0))
   WeightMatrix(logw, logwbar, order, loge)
end

# log(exp(a) + exp(b)), with -Inf absorbing as the log of zero.
function _logaddexp(a::Float64, b::Float64)
   a == -Inf && return b
   b == -Inf && return a
   m = max(a, b)
   m + log1p(exp(-abs(a - b)))
end

# Elementary symmetric polynomials of each row's weights over column suffixes, in
# logs. Built from the empty suffix backwards using eₖ(j:n) = eₖ(j+1:n) +
# w[j]·eₖ₋₁(j+1:n); `loge[k+1, i, pos]` covers columns at positions pos:n.
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

# ---------------------------------------------------------------------------
# Interface consumed by the sampler (sis.jl). Plain-Int arguments keep these
# independent of the residual/workspace types.
# ---------------------------------------------------------------------------

# Order in which to visit the columns (by decreasing sum).
_columnorder(::UniformWeights, colsums) = sortperm(colsums, rev = true)
_columnorder(model::WeightMatrix, colsums) = model.order

# Factor multiplying a row's odds of a one in the column at sampling position
# `pos`, given the row's residual sum `ρ` and the number of columns `after` it.
_weightfactor(::UniformWeights, row, ρ, after, pos) = 1.0
function _weightfactor(model::WeightMatrix, row, ρ, after, pos)
   ρ == 0 && return 1.0
   logden = model.loge[ρ+1, row, pos+1]         # log eρ over the columns after pos
   logden == -Inf && return Inf                 # eρ == 0: the row is forced to a one here
   lognum = model.loge[ρ, row, pos+1]           # log eρ₋₁
   (after - ρ + 1) / ρ * exp(model.logwbar[row, pos] + lognum - logden)
end

# Contribution of a placed one at (row, column) to log ∏ w^z.
_logweight(::UniformWeights, row, label) = 0.0
_logweight(model::WeightMatrix, row, label) = model.logw[row, label]
