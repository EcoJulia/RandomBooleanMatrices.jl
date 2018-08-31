# RandomBooleanMatrices
Create random boolean matrices that maintain row and column sums. This is a very common use case for null models in ecology.
Status](https://travis-ci.org/EcoJulia/SpatialEcology.jl.svg?branch=master)](https://travis-ci.org/EcoJulia/SpatialEcology.jl)
used different forms of swapping, or some alternative approaches like the `quasiswap` algorithm in R's vegan package. These
methods are neither efficient, nor are they guaranteed to sample the possible distribution of boolean vectors with a given row
and column sum equally.

Currently, the package only offers an implementation of the `curveball` algorithm of Strona et al. (2014). There are two forms: 
a `randomize_matrix!` function that will randomize a sparse boolean `Matrix` in-place, and a generator form:

```julia
using SparseArrays, RandomBooleanMatrices

# in-place
m = sprand(Bool, 1000, 1000, 0.1)
randomize_matrix!(m)

# genenrator
m = sprand(Bool, 1000, 1000, 0.1)
rmg = random_matrices(m)
m1 = rmg() # creates a new random matrix
m2 = rmg()
```

### References
Strona, G., Nappo, D., Boccacci, F., Fattorini, S. & San-Miguel-Ayanz, J. (2014) 
A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals. 
Nature Communications, 5, 4114.
