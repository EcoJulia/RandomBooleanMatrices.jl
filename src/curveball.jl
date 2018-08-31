using Random, SparseArrays

function _interdif(v1, v2)
   inter, dif = Int[], Int[]
   i,j = firstindex(v1), firstindex(v2)
   @inbounds while i <= lastindex(v1) || j <= lastindex(v2)
      if j > lastindex(v2)
         push!(dif, v1[i])
         i+=1
         continue
      elseif i > lastindex(v1)
         push!(dif, v2[j])
         j+=1
         continue
      end
      if v1[i] == v2[j]
         push!(inter, v1[i])
         i += 1
         j += 1
      elseif j > lastindex(v2) || v1[i] < v2[j]
         push!(dif, v1[i])
         i+=1
      elseif i > lastindex(v1) || v1[i] > v2[j]
         push!(dif, v2[j])
         j+=1
      end
   end
   inter, dif
end

function _curveball!(m::SparseMatrixCSC{Bool, Int})
   R, C = size(m)

   for rep ∈ 1:5C
	   A, B = rand(1:C,2)

      # use views directly into the sparse matrix to avoid copying
	   a, b = view(m.rowval, m.colptr[A]:m.colptr[A+1]-1), view(m.rowval, m.colptr[B]:m.colptr[B+1]-1)
	   l_a, l_b = length(a), length(b)

      # an efficient algorithm since both a and b are sorted
      shared, not_shared = _interdif(a, b)
	   l_ab = length(shared)

	   if !(l_ab ∈ (l_a, l_b))
			   shuffle!(not_shared)
			   L     = l_a - l_ab
			   a .= sort!([shared; not_shared[1:L]])
			   b .= sort!([shared; not_shared[L+1:end]])
	   end
   end

   return m
end
