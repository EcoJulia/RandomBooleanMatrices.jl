using SparseArrays, RandomBooleanMatrices, Random

Random.seed!(1337)

@testset "curveball" begin
    m = sprand(Bool, 8, 6, 0.2)
    rsm = sum(m, dims = 1)
    csm = sum(m, dims = 2)
    randomize_matrix!(m, method = curveball)

    @test rsm == sum(m, dims = 1)
    @test csm == sum(m, dims = 2)

    m2 = rand(0:1, 6, 5)
    rsm = sum(m2, dims = 1)
    csm = sum(m2, dims = 2)
    rmg = random_matrices(m2, method = curveball)
    m3 = rmg()

    @test rsm == sum(m3, dims = 1)
    @test csm == sum(m3, dims = 2)

    m4 = rmg()

    @test m3 != m4
end
