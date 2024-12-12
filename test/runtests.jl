using Test
using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    using DimensionalData
    using Statistics
    using BenchmarkTools
    using DSP
    using Associations
    using StatsBase
    using TimeseriesFeatures

    X = randn(1000, 5)
    μ = Feature(mean, :mean, ["distribution"], "μ")
    σ = Feature(std, :std, ["distribution"], "σ")
    𝒇₁ = FeatureSet([sum, length], [:sum, :length], [["distribution"], ["sampling"]],
                    ["∑x¹", "∑x⁰"])
    𝒇 = FeatureSet([μ, σ]) + 𝒇₁
    𝒇₂ = FeatureSet([μ, σ])
    X = randn(100, 2)
    𝒇₃ = 𝒇₁ + 𝒇₂
end

@testitem "FeatureArray stability" setup=[Setup] begin
    x = randn(10)
    d = Feat(DimensionalData.Categorical(Symbol.(1:length(x));
                                         order = DimensionalData.Unordered()))
    @inferred FeatureArray(x, DimensionalData.format((d,), x), (), DimensionalData.NoName(),
                           DimensionalData.NoMetadata())
    @inferred FeatureArray(x, DimensionalData.format((d,), x))
    f = @inferred FeatureArray(x, (d,))
    f = @inferred FeatureArray(x, Symbol.(1:length(x)))
end

# @testset "Feature stability" begin
#     x = randn(1000) .|> Float32
#     @inferred getmethod(μ)(x)
#     @inferred μ(x)
# end

@testitem "FeatureSet" setup=[Setup] begin
    @test 𝒇₃ isa FeatureSet
    @inferred 𝒇₁(X)
    @inferred 𝒇₃(X)
    @test getnames(𝒇₃) == [:sum, :length, :mean, :std]
    @inferred 𝒇₃[:sum]
    @test getname(𝒇₃[:sum]) == :sum
    @test all([getname(𝒇₃[x]) == x for x in getnames(𝒇₃)])
    @inferred 𝒇₃(X)[:sum, :]
    @test 𝒇₃(X)[:sum] == 𝒇₃(X)[:sum, :]

    @test hcat(eachslice(𝒇₃(X), dims = 2)...) isa FeatureArray # Check rebuild is ok (does not convert to DimArray

    F = 𝒇₃(X)[:, 1]
    𝑓 = [:sum, :length]
    @inferred getindex(F, 𝑓[1])

    F = 𝒇₃(X)
    @inferred getindex(F, 𝑓[1])
    @inferred getindex(F, 1:2)
    # @inferred getindex(F, 𝑓) # Not typestable

    # @inferred 𝒇₃(X)[[:sum, :length], :]
    @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :]
    @test 𝒇₁ == 𝒇₃ \ 𝒇₂ == setdiff(𝒇₃, 𝒇₂)
    @test 𝒇₃ \ 𝒇₂ isa FeatureSet
    @test 𝒇₃ == 𝒇₁ ∪ 𝒇₂ == union(𝒇₁, 𝒇₂)
    @test 𝒇₁ ∪ 𝒇₂ isa FeatureSet
    @test 𝒇₂ == 𝒇₃ ∩ 𝒇₂ == intersect(𝒇₃, 𝒇₂)
    @test 𝒇₃ ∩ 𝒇₂ isa FeatureSet

    @test 𝒇₁ + μ isa FeatureSet
    @test μ + 𝒇₁ isa FeatureSet
end

@testitem "Multidimensional arrays" setup=[Setup] begin
    X = rand(100, 3, 3)
    @test_nowarn 𝒇₁(X)
    @test_nowarn 𝒇₃(X)
    @test_nowarn 𝒇₃[:sum]
    @test_nowarn 𝒇₃(X)[:sum, :, :]
    @test 𝒇₃(X)[:sum] == 𝒇₃(X)[:sum, :, :]
    @test_nowarn 𝒇₃(X)[[:sum, :length], :, :]
    @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :, :]

    F = @test_nowarn μ(X)
    @test F isa Array{<:Float64, 2} # Extra dims are dropped
    @test size(F) == (3, 3)
end

@testitem "Vector of vectors" setup=[Setup] begin
    X = [randn(100) for _ in 1:9]
    @test_nowarn 𝒇₁(X)
    @test_nowarn 𝒇₃(X)
    @test_nowarn 𝒇₃[:sum]
    @test_nowarn 𝒇₃(X)[:sum, :, :]
    @test 𝒇₃(X)[:sum] == 𝒇₃(X)[:sum, :, :]
    @test_nowarn 𝒇₃(X)[[:sum, :length], :, :]
    @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :, :]
end

@testitem "FeatureArray indexing" setup=[Setup] begin
    𝑓s = [:mean, :std]
    𝑓 = FeatureSet([μ, σ])

    X = randn(1000)
    F = 𝒇(X)
    @test F[𝑓] == F[𝑓s]
    @test F[𝑓] == F[1:2]
    @test all(F[𝑓s] .== F[1:2]) # Importantly, F[𝑓s, :] is NOT SUPPORTED

    X = randn(1000, 200)
    F = 𝒇(X)
    @test F[𝑓] == F[𝑓s]
    @test F[𝑓] == F[𝑓, :] == F[1:2, :]
    @test F[𝑓s] == F[𝑓s, :] == F[1:2, :]

    X = randn(1000, 20, 20)
    F = 𝒇(X)
    @test F[𝑓] == F[𝑓s]
    @test F[𝑓] == F[𝑓, :, :] == F[1:2, :, :]
    @test F[𝑓s] == F[𝑓s, :, :] == F[1:2, :, :]
end

@testitem "SuperFeatures" setup=[Setup] begin
    x = rand(1000, 2)
    @test_nowarn TimeseriesFeatures.zᶠ(x)
    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series",
                     super = TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series";
                     super = TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])
    @test all(isapprox.(𝒇(x), [0.0 0.0; 1.0 1.0]; atol = 1e-9))

    x = randn(1000)
    @test 𝒇(x) isa AbstractFeatureVector
    X = randn(1000, 2000)
    z = 𝒇(X)
    @test z isa AbstractFeatureMatrix

    X = collect.(eachcol(X)) # Vector of vectors
    @test 𝒇(X) isa AbstractFeatureMatrix
    @test z == 𝒇(X)

    if Threads.nthreads() ≥ 8 # This will only be faster if the machine has a solid number of threads
        a = @benchmark 𝒇($X)
        _X = eachcol(X)
        b = @benchmark 𝒇.($_X)
        @test median(a.times) ≤ median(b.times) # Check mutlithreading works
        @test a.allocs ≤ b.allocs
    end

    # @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :]

    @test vcat(𝒇, 𝒇) isa SuperFeatureSet
    @test vcat(𝒇, 𝒇₁) isa SuperFeatureSet

    @inferred setdiff(𝒇, 𝒇)
    @inferred setdiff(𝒇, 𝒇₁)
    @test setdiff(𝒇, 𝒇₁) isa SuperFeatureSet

    @test union(𝒇, 𝒇) isa SuperFeatureSet
    @test union(𝒇, 𝒇₁) isa SuperFeatureSet
    @test union(𝒇₁, 𝒇₁) isa FeatureSet

    @test intersect(𝒇, 𝒇) isa SuperFeatureSet
    @test intersect(𝒇, 𝒇₁) isa SuperFeatureSet
    @test isempty(intersect(𝒇, 𝒇₁))
    @test intersect(𝒇₁, 𝒇₁) isa FeatureSet
    @test intersect(union(𝒇, 𝒇₁), 𝒇₁) == 𝒇₁
    @test intersect(union(𝒇, 𝒇₁), 𝒇) == 𝒇

    @test setdiff(𝒇₃, 𝒇₂) == 𝒇₃[1:2]
    @test setdiff(𝒇₃, 𝒇₂) isa FeatureSet
    @test setdiff(𝒇 + 𝒇₂, 𝒇₂) isa SuperFeatureSet

    @test SuperFeatureSet(𝒇₁) isa SuperFeatureSet
    @test 𝒇 \ 𝒇[[1]] == 𝒇[[2]] == 𝒇 \ 𝒇[1]
    @test 𝒇₁ == 𝒇₃ \ 𝒇₂ == setdiff(𝒇₃, 𝒇₂)
    @test 𝒇₃ \ 𝒇₂ isa FeatureSet
    @test 𝒇₃ == 𝒇₁ ∪ 𝒇₂ == union(𝒇₁, 𝒇₂)
    @test 𝒇₁ ∪ 𝒇₂ isa FeatureSet
    @test 𝒇₂ == 𝒇₃ ∩ 𝒇₂ == intersect(𝒇₃, 𝒇₂)
    @test 𝒇₃ ∩ 𝒇₂ isa FeatureSet
end

@testitem "DimArrays" setup=[Setup] begin
    μ = Feature(mean, :mean, ["distribution"], "μ")
    σ = Feature(std, :std, ["distribution"], "σ")
    𝒇₁ = FeatureSet([sum, length], [:sum, :length], [["distribution"], ["sampling"]],
                    ["∑x¹", "∑x⁰"])
    𝒇 = FeatureSet([μ, σ]) + 𝒇₁

    m = Dict(:a => "yolo")
    n = "Bert"
    x = DimArray(randn(100), (Dim{:x}(1:100),); metadata = m, name = n)
    @test σ(x) == σ(x |> vec)
    @test 𝒇(x) == 𝒇(x |> vec)
    @inferred 𝒇(x)
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n

    x = DimArray(rand(100, 2), (Dim{:x}(1:100), Dim{:var}(1:2)); name = n, metadata = m)
    @test σ(x) == σ(x |> Matrix)
    @test 𝒇(x).data == 𝒇(x |> Matrix).data
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n

    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series",
                     super = TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series";
                     super = TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])

    F = @test_nowarn σ(x)
    @test F isa DimArray{<:Float64}
    @test all(F .≈ 1.0)
    F = @test_nowarn μ(x)
    @test all(abs.(F) .< 1e-10)

    F = 𝒇(x)
    @test F isa FeatureArray{<:Float64}
    @test F ≈ [0 0; 1 1]

    x = DimArray(rand(100, 2, 2), (Dim{:x}(1:100), Dim{:var}(1:2), Y(1:2)); name = n,
                 metadata = m)
    @test σ(x) == σ(x |> Array)
    @test 𝒇(x).data == 𝒇(x |> Array).data
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n

    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series",
                     super = TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series";
                     super = TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])

    F = @test_nowarn σ(x)
    @inferred σ(x)
    @test all(F .≈ 1.0)
    @test F isa FeatureArray{<:Float64}
    F = @test_nowarn μ(x)
    @test F isa FeatureArray{<:Float64}
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n

    F = @inferred 𝒇(x)
    y = parent(x)
    @inferred 𝒇(y)

    @test F isa FeatureArray{<:Float64}
    @test F ≈ cat([0 0; 1 1], [0 0; 1 1], dims = 3)
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n
end

@testitem "ACF and PACF" setup=[Setup] begin
    X = randn(1000, 10)
    _acf = mapslices(x -> autocor(x, TimeseriesFeatures.ac_lags; demean = true), X;
                     dims = 1)
    @test all(AC(X) .== _acf)
    _pacf = mapslices(x -> pacf(x, TimeseriesFeatures.ac_lags; method = :regression), X;
                      dims = 1)
    @test all(Partial_AC(X) .== _pacf)
end

@testitem "PACF superfeatures" setup=[Setup] begin
    X = randn(1000, 10)
    lags = TimeseriesFeatures.ac_lags
    AC_slow = FeatureSet([x -> autocor(x, [ℓ]; demean = true)[1]::Float64 for ℓ in lags],
                         Symbol.(["AC_$ℓ" for ℓ in lags]),
                         [["correlation"] for ℓ in lags],
                         ["Autocorrelation at lag $ℓ" for ℓ in lags])
    AC_partial_slow = FeatureSet([x -> pacf(x, [ℓ]; method = :regression)[1]::Float64
                                  for ℓ in lags],
                                 Symbol.(["Partial_AC_$ℓ" for ℓ in lags]),
                                 [["correlation"] for ℓ in lags],
                                 ["Partial autocorrelation at lag $ℓ (regression method)"
                                  for ℓ in lags])

    @test all(AC_slow(X) .== AC(X))
    @test all(AC_partial_slow(X) .== Partial_AC(X))
    println("\nFeature autocorrelation: ")
    @time AC_slow(X)
    println("\nSuperFeature autocorrelation: ")
    @time AC(X)
    println("\nFeature partial autocorrelation: ")
    @time AC_partial_slow(X)
    println("\nSuperfeature partial autocorrelation: ")
    @time Partial_AC(X)
end

@testitem "RAD" setup=[Setup] begin
    x = sin.(0.01:0.01:10)
    r = autocor(x, 1:(length(x) - 1))
    τ = TimeseriesFeatures.firstcrossingacf(x)
    @test 161 < τ < 163
    @test_nowarn CR_RAD(x)
end

@testitem "PairwiseFeatures" setup=[Setup] begin
    X = randn(1000, 5)
    𝑓 = Pearson
    f = @test_nowarn 𝑓(X)

    X = DimArray(randn(100, 2), (Dim{:x}(1:100), Dim{:var}(1:2)))
    f = @test_nowarn 𝑓(X)
    @test dims(f, 1) == dims(X, 2) == dims(f, 2)

    𝒇 = FeatureSet([Pearson, Covariance])
    @test 𝒇(X) isa FeatureArray
end

@testitem "MultivariateFeatures" setup=[Setup] begin
    X = DimArray(randn(100000, 20), (Dim{:x}(1:100000), Dim{:var}(1:20)))
    @test all(isapprox.(Covariance_svd(X), Covariance(X), atol = 1e-4))
    @time f1 = Covariance(X) # Much faster
    @time f2 = Covariance_svd(X) # Much faster
    @time cov(X) # Faster again
end

@testitem "AssociationsExt" setup=[Setup] begin
    X = randn(1000, 2)
    F = @test_nowarn MI_Kraskov_NN_20(X)
    @test F[2] < 0.1

    x = sin.(0.01:0.01:10) .^ 2
    y = cos.(0.01:0.01:10) .^ 3
    F = @test_nowarn MI_Kraskov_NN_20([x y])
    @test F[2] > 3
end

@testitem "Super" setup=[Setup] begin
    using StatsBase, TimeseriesFeatures, Test
    𝐱 = rand(1000, 2)
    μ = Feature(mean, :μ, ["0"], "Mean value of the time series")
    σ = Feature(std, :σ, ["1"], "Standard deviation of the time series")
    μ_z = @test_nowarn Super(μ, TimeseriesFeatures.zᶠ)
    σ_z = @test_nowarn Super(σ, TimeseriesFeatures.zᶠ)
    @test μ_z isa Super
    @test μ_z(𝐱)≈[0, 0] atol=1e-13
    𝒇 = SuperFeatureSet([μ_z, σ_z])
    @test all(isapprox.(𝒇(𝐱), [0.0 0.0; 1.0 1.0]; atol = 1e-9))

    # Check speed
    μ = [Feature(mean, Symbol("μ_$i"), ["0"], "Mean value of the time series")
         for i in 1:100]
    superfeature = @test_nowarn SuperFeatureSet(Super.(μ, [TimeseriesFeatures.zᶠ]))
    feature = [Feature(x -> (zscore(x)), Symbol("μ_$i"), ["0"],
                       "Mean value of the time series") for i in 1:100]

    a = @benchmark superfeature(𝐱) setup=(superfeature = SuperFeatureSet(Super.(μ,
                                                                                [
                                                                                    TimeseriesFeatures.zᶠ
                                                                                ]));
                                          𝐱 = rand(1000, 2))
    b = @benchmark [f(𝐱) for f in feature] setup=(feature = [Feature(x -> (zscore(x)),
                                                                     Symbol("μ_$i"), ["0"],
                                                                     "Mean value of the time series")
                                                             for i in 1:100];
                                                  𝐱 = rand(1000, 2))
    @test median(a.times) < median(b.times) / 2

    # using PProf
    # using Profile
    # Profile.clear()
    # # @profile 𝒇(𝐱)
    # # pprof()
    # @profile superfeature(𝐱)
    # pprof()
end

@testitem "PPC" setup=[Setup] begin
    using DimensionalData, DSP, Test, TimeseriesFeatures
    X = randn(1000, 2)
    F = @test_nowarn PPC_Analytic_Phase(X)

    X = DimArray(randn(1000, 2), (Ti(1:1000), Dim{:var}(1:2)))
    F = @test_nowarn Analytic_Phase(X)
    F = @test_nowarn PPC_Analytic_Phase(X)

    x = 0.01:0.01:100
    X = [sin.(x) cos.(x)]
    F = PPC_Analytic_Phase(X)
    @test F≈[1 1; 1 1] rtol=1e-3

    @test false # This needs more tests
    # (𝑓::AbstractPairwiseFeature)(x::AbstractVector) = getmethod(𝑓)(x, x)
    # function (𝑓::AbstractPairwiseFeature)(X::AbstractArray)
    #     idxs = CartesianIndices(size(X)[2:end])
    #     idxs = Iterators.product(idxs, idxs)
    #     f = i -> getmethod(𝑓)(X[:, first(i)], X[:, last(i)])
    #     f.(idxs)
    # end
    # function (𝑓::AbstractPairwiseFeature)(X::DimensionalData.AbstractDimMatrix)
end

@testitem "TimeseriesToolsExt" setup=[Setup] begin
    using StatsBase
    using TimeseriesTools
    x = colorednoise(0.1:0.1:10000)
    @test_nowarn TimeseriesTools.timescale(x; method = :ac_crossing)

    x = set(x, sin.(times(x) ./ 2π))
    τ = TimeseriesTools.timescale(x) # This is 1/4 the period; i.e. the time shift it requires to become anti-phase
    y = TimeseriesTools.Operators.ℬ(x, Int(τ ÷ step(x)))
    @test cor(x, y)≈0 atol=0.05
end

@testitem "Type stability" setup=[Setup] begin
    𝒇s = SuperFeature.(𝒇₃) |> SuperFeatureSet
    x = randn(1000) .|> Float32
    xx = [randn(1000) for _ in 1:10]
    X = randn(1000, 10)

    # * Features
    @inferred getmethod(μ)(x)
    @inferred μ(x)
    @inferred μ(xx)
    @inferred μ(X)

    # * Super Features
    @inferred SuperFeature(μ, TimeseriesFeatures.zᶠ)
    𝑓 = SuperFeature(μ, TimeseriesFeatures.zᶠ)
    @test 𝑓(rand(1000))≈0.0 atol=1e-10
    @inferred getmethod(𝑓)(x)
    @inferred getsuper(𝑓)(x)
    @inferred 𝑓(x)
    @inferred 𝑓(xx)
    @test all(abs.(𝑓(xx)) .< 1e-10)
    @inferred 𝑓(X)
    @test all(abs.(𝑓(X)) .< 1e-10)

    # * FeatureSets (x, xx, X)

    # * SuperFeatureSets (x, xx, X)
end
