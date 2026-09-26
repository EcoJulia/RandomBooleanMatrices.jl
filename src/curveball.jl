
function sortmerge!(v1, v2, ret, iend, jend)
   retind = 0
   i,j = firstindex(v1), firstindex(v2)
   @inbounds while i <= iend || j <= jend
      if j > jend
         ret[retind += 1] = v1[i]
         i+=1
         continue
      elseif i > iend
         ret[retind += 1] = v2[j]
         j+=1
         continue
      end
      if v1[i] == v2[j]
         error("The two vectors are not supposed to have overlapping values")
      elseif j > lastindex(v2) || v1[i] < v2[j]
         ret[retind += 1] = v1[i]
         i+=1
      elseif i > lastindex(v1) || v1[i] > v2[j]
         ret[retind += 1] = v2[j]
         j+=1
      end
   end
end


function _interdif!(v1, v2, inter, dif)
   nshared, ndiff = 0, 0
   i,j = firstindex(v1), firstindex(v2)
   @inbounds while i <= lastindex(v1) || j <= lastindex(v2)
      if j > lastindex(v2)
         dif[ndiff += 1] = v1[i]
         i+=1
         continue
      elseif i > lastindex(v1)
         dif[ndiff += 1] = v2[j]
         j+=1
         continue
      end
      if v1[i] == v2[j]
         inter[nshared += 1] = v1[i]
         i += 1
         j += 1
      elseif j > lastindex(v2) || v1[i] < v2[j]
         dif[ndiff += 1] = v1[i]
         i+=1
      elseif i > lastindex(v1) || v1[i] > v2[j]
         dif[ndiff += 1] = v2[j]
         j+=1
      end
   end
   ndiff, nshared
end

# The columns that can take part in a trade. A pair involving an empty column is a
# no-op and column sums never change, so this set is fixed along the chain; drawing
# pairs only from it leaves the target distribution unchanged, and stops a sparse
# matrix with many empty columns from spending nearly all its trades on nothing.
_tradingcolumns(m::SparseMatrixCSC) = findall(>(0), diff(SparseArrays.getcolptr(m)))

# The default number of trades per call: five per column that can trade.
_defaulttrades(m::SparseMatrixCSC) = 5 * length(_tradingcolumns(m))

function _curveball!(m::SparseMatrixCSC{Bool, Int}, rng = Random.GLOBAL_RNG, trades::Int = _defaulttrades(m))
   cols = _tradingcolumns(m)
   C = length(cols)
   C < 2 && return m
   mcs = min(2maximum(diff(m.colptr)), size(m, 1))
   not_shared, shared = Vector{Int}(undef, mcs), Vector{Int}(undef, mcs)
   newa, newb = Vector{Int}(undef, mcs), Vector{Int}(undef, mcs)

   for rep ∈ 1:trades
	   A, B = cols[rand(rng, 1:C)], cols[rand(rng, 1:C)]

      # use views directly into the sparse matrix to avoid copying
	   a, b = view(m.rowval, m.colptr[A]:m.colptr[A+1]-1), view(m.rowval, m.colptr[B]:m.colptr[B+1]-1)
	   l_a, l_b = length(a), length(b)

      # an efficient algorithm since both a and b are sorted
      l_dif, l_ab = _interdif!(a, b, shared, not_shared)

	   if !(l_ab ∈ (l_a, l_b))
            L = l_a - l_ab

			   sample!(rng, view(not_shared, Base.OneTo(l_dif)), view(newa,Base.OneTo(L)), replace = false, ordered = true)
            L2,_ = _interdif!(view(newa, 1:L), view(not_shared, Base.OneTo(l_dif)), newa, newb)

			   sortmerge!(shared, newa, a, l_ab, L)
			   sortmerge!(shared, newb, b, l_ab, L2)
	   end
   end

   return m
end

"""
    CurveballSampler(trades)

Generator state for the curveball method: each draw advances the stored matrix by
`trades` curveball trades. The matrix is itself a running Markov chain, so the
generator warm-starts it from an independent SIS draw (see `matrixrandomizer`) to
decorrelate the first sample from the input, and `trades` controls how far the
chain moves between successive draws (thinning).
"""
struct CurveballSampler
   trades::Int
end

_draw!(m::SparseMatrixCSC{Bool, Int}, rng, s::CurveballSampler) = _curveball!(m, rng, s.trades)
