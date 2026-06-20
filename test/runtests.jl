using RandomBooleanMatrices
using SparseArrays
using Random
using Test

Random.seed!(1337)

# the row and column sums of a matrix, as plain vectors
margins(m) = (vec(sum(m, dims = 1)), vec(sum(m, dims = 2)))

# brute-force count of the m×n binary matrices with the given margins
function countmatrices(rowsums, colsums)
    m, n = length(rowsums), length(colsums)
    count = 0
    for bits in 0:(2^(m * n) - 1)
        A = [(bits >> ((j - 1) * m + (i - 1))) & 1 for i in 1:m, j in 1:n]
        vec(sum(A, dims = 2)) == rowsums && vec(sum(A, dims = 1)) == colsums && (count += 1)
    end
    count
end

@testset "curveball" begin
    m = sprand(Bool, 8, 6, 0.2)
    m_old = copy(m)
    rsm = sum(m, dims = 1)
    csm = sum(m, dims = 2)
    randomize_matrix!(m, method = curveball)

    @test rsm == sum(m, dims = 1)
    @test csm == sum(m, dims = 2)
    @test m_old != m

    m2 = rand(0:1, 6, 5)
    rsm = sum(m2, dims = 1)
    csm = sum(m2, dims = 2)
    rmg = matrixrandomizer(m2, method = curveball)
    m3 = rand(rmg)

    @test rsm == sum(m3, dims = 1)
    @test csm == sum(m3, dims = 2)

    m4 = rand(rmg)

    @test m3 != m4
end

@testset "sis" begin
    m = sprand(Bool, 8, 6, 0.3)
    csm, rsm = margins(m)
    randomize_matrix!(m, method = sis)

    @test margins(m) == (csm, rsm)

    m2 = rand(0:1, 6, 5)
    csm, rsm = margins(m2)
    rmg = matrixrandomizer(m2, method = sis)
    m3 = rand(rmg)
    m4 = rand(rmg)

    @test margins(m3) == (csm, rsm)   # margins preserved
    @test margins(m4) == (csm, rsm)
    @test m3 != m4                     # independent draws

    # a fully determined problem has a single solution, which must be returned
    determined = sparse(Bool[1 1 1; 1 1 0; 1 0 0])   # unique matrix with margins (3,2,1)
    @test countmatrices([3, 2, 1], [3, 2, 1]) == 1
    @test rand(matrixrandomizer(determined, method = sis)) == determined
end

@testset "exact" begin
    # counting agrees with brute-force enumeration
    for (rs, cs) in [([2, 1, 1], [1, 2, 1]), ([2, 2, 1, 1], [2, 1, 2, 1]),
                     ([3, 1, 1, 1], [2, 2, 1, 1]), ([2, 2, 2], [2, 2, 2])]
        @test RandomBooleanMatrices._exact_counts(rs, cs).total == countmatrices(rs, cs)
    end

    m = sprand(Bool, 6, 5, 0.4)
    csm, rsm = margins(m)
    randomize_matrix!(m, method = exact)
    @test margins(m) == (csm, rsm)

    m2 = rand(0:1, 5, 6)
    csm, rsm = margins(m2)
    rmg = matrixrandomizer(m2, method = exact)   # counts once, reused below
    m3 = rand(rmg)
    m4 = rand(rmg)

    @test margins(m3) == (csm, rsm)
    @test margins(m4) == (csm, rsm)
    @test m3 != m4

    # a fully determined problem returns its single solution
    determined = sparse(Bool[1 1 1; 1 1 0; 1 0 0])
    @test rand(matrixrandomizer(determined, method = exact)) == determined

    # the exact sampler is uniform: every matrix with these margins appears, and
    # frequencies are close to equal (seeded rng keeps this reproducible)
    rs, cs = [2, 2, 1, 1], [2, 1, 2, 1]
    seed = sparse(Bool[1 0 1 0; 1 1 0 0; 0 0 1 0; 0 0 0 1])
    @test margins(seed) == (cs, rs)
    n = countmatrices(rs, cs)
    rmg = matrixrandomizer(seed, MersenneTwister(20240620), method = exact)
    samples = [rand(rmg) for _ in 1:200n]
    counts = collect(values(Dict(s => count(==(s), samples) for s in unique(samples))))
    @test length(counts) == n                          # full support covered
    @test maximum(counts) < 2 * minimum(counts)        # roughly uniform
end
