using RandomBooleanMatrices
using SparseArrays
using Random
using Test

mean(x) = sum(x) / length(x)

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

# brute-force weighted count κ = Σ_z ∏ w^z over the binary matrices with the margins
function weightedcount(rowsums, colsums, w)
    m, n = length(rowsums), length(colsums)
    κ = 0.0
    for bits in 0:(2^(m * n) - 1)
        A = [(bits >> ((j - 1) * m + (i - 1))) & 1 for i in 1:m, j in 1:n]
        if vec(sum(A, dims = 2)) == rowsums && vec(sum(A, dims = 1)) == colsums
            κ += prod(w[i, j]^A[i, j] for i in 1:m, j in 1:n)
        end
    end
    κ
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

    # the trade count is controllable; zero trades leaves the matrix untouched
    m5 = sprand(Bool, 8, 6, 0.3)
    @test randomize_matrix!(copy(m5), method = curveball, trades = 0) == m5
    @test matrixrandomizer(m5, method = curveball, trades = 7).state.trades == 7

    # the generator warm-starts from an independent draw and preserves margins
    csm5, rsm5 = margins(m5)
    @test margins(rand(matrixrandomizer(m5, method = curveball, trades = 10))) == (csm5, rsm5)
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

    m2 = rand(MersenneTwister(2718), 0:1, 5, 6)
    csm, rsm = margins(m2)
    rmg = matrixrandomizer(m2, MersenneTwister(2719), method = exact)   # counts once, reused
    draws = [rand(rmg) for _ in 1:10]

    @test all(d -> margins(d) == (csm, rsm), draws)
    @test length(unique(draws)) > 1                    # independent draws vary

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

@testset "weighted sis" begin
    m0 = sparse(Bool[1 0 1; 1 1 0; 0 1 0])             # margins (2,2,1) / (2,2,1)
    cs, rs = margins(m0)
    w = [1.0 2.0 0.5; 1.5 1.0 2.0; 0.5 1.0 1.0]

    sampler = importance_sampler(m0, w, MersenneTwister(11))
    draws = [rand(sampler) for _ in 1:100_000]

    # every draw is a fixed-margin matrix
    @test all(d -> margins(d.matrix) == (cs, rs), draws)
    @test draws[1].matrix != draws[2].matrix           # independent draws

    # the mean importance weight is an unbiased estimate of the weighted count κ
    weights = [exp(d.logweight) for d in draws]
    @test isapprox(mean(weights), weightedcount(rs, cs, w), rtol = 0.03)

    # uniform weights reduce to the plain (unweighted) matrix count
    uniform = importance_sampler(m0, ones(size(m0)), MersenneTwister(11))
    uweights = [exp(rand(uniform).logweight) for _ in 1:50_000]
    @test isapprox(mean(uweights), countmatrices(rs, cs), rtol = 0.02)

    # canonicalisation changes the proposal but not the (unbiased) estimate
    off = importance_sampler(m0, w, MersenneTwister(11); canonicalize = false)
    offweights = [exp(rand(off).logweight) for _ in 1:100_000]
    @test isapprox(mean(offweights), weightedcount(rs, cs, w), rtol = 0.03)

    # structural zeros: a zero weight forbids a one there, and κ is still recovered
    wz = [0.0 2.0 1.0; 1.5 0.0 2.0; 1.0 0.5 0.0]       # zero diagonal
    derange = sparse(Bool[0 1 0; 0 0 1; 1 0 0])         # margins (1,1,1), off-diagonal
    zsampler = importance_sampler(derange, wz, MersenneTwister(3))
    zdraws = [rand(zsampler) for _ in 1:100_000]
    @test all(d -> all(i -> !d.matrix[i, i], 1:3), zdraws)   # never a one on a zero
    zweights = [exp(d.logweight) for d in zdraws]
    @test isapprox(mean(zweights), weightedcount([1, 1, 1], [1, 1, 1], wz), rtol = 0.05)

    # input validation
    @test_throws DimensionMismatch importance_sampler(m0, ones(2, 2))
    @test_throws ArgumentError importance_sampler(m0, fill(-1.0, 3, 3))   # negative weight
    @test_throws ArgumentError importance_sampler(m0, zeros(3, 3))        # no positive entry
end
