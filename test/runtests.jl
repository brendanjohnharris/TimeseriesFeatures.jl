using DSP
using CausalityTools
using StatsBase
using TimeseriesFeatures
using Test
using DimensionalData
using Statistics
using BenchmarkTools

X = randn(1000, 5)

μ = Feature(mean, :mean, ["distribution"], "μ")
σ = Feature(std, :std, ["distribution"], "σ")
𝒇₁ = FeatureSet([sum, length], [:sum, :length], [["distribution"], ["sampling"]],
                ["∑x¹", "∑x⁰"])
𝒇 = FeatureSet([μ, σ]) + 𝒇₁
@testset "FeatureSet" begin
    𝒇₂ = @test_nowarn FeatureSet([μ, σ])
    X = randn(100, 2)
    𝒇₃ = 𝒇₁ + 𝒇₂
    @test_nowarn 𝒇₁(X)
    @test_nowarn 𝒇₃(X)
    @test getnames(𝒇₃) == [:sum, :length, :mean, :std]
    @test_nowarn 𝒇₃[:sum]
    @test getname(𝒇₃[:sum]) == :sum
    @test all([getname(𝒇₃[x]) == x for x in getnames(𝒇₃)])
    @test_nowarn 𝒇₃(X)[:sum, :]
    @test 𝒇₃(X)[:sum] == 𝒇₃(X)[:sum, :]
    @test_nowarn 𝒇₃(X)[[:sum, :length], :]
    @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :]
    @test 𝒇₁ == 𝒇₃ \ 𝒇₂ == setdiff(𝒇₃, 𝒇₂)
    @test 𝒇₃ == 𝒇₁ ∪ 𝒇₂
    @test 𝒇₂ == 𝒇₃ ∩ 𝒇₂
end

@testset "FeatureArray indexing" begin
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

@testset "SuperFeatures" begin
    x = rand(1000, 2)
    @test_nowarn TimeseriesFeatures.zᶠ(x)
    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series",
                     super = TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series";
                     super = TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])
    @test all(isapprox.(𝒇(x), [0.0 0.0; 1.0 1.0]; atol = 1e-9))
end

@testset "DimArrays" begin
    μ = Feature(mean, :mean, ["distribution"], "μ")
    σ = Feature(std, :std, ["distribution"], "σ")
    𝒇₁ = FeatureSet([sum, length], [:sum, :length], [["distribution"], ["sampling"]],
                    ["∑x¹", "∑x⁰"])
    𝒇 = FeatureSet([μ, σ]) + 𝒇₁

    x = DimArray(randn(100), (Dim{:x}(1:100),))
    @test σ(x) == σ(x |> vec)
    @test 𝒇(x) == 𝒇(x |> vec)

    x = DimArray(randn(100, 2), (Dim{:x}(1:100), Dim{:var}(1:2)))
    @test σ(x) == σ(x |> Matrix)
    @test 𝒇(x).data == 𝒇(x |> Matrix).data

    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series",
                     super = TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series";
                     super = TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])

    F = @test_nowarn σ(x)
    @test all(F .≈ 1.0)
    @test F isa FeatureArray{<:Float64}
    F = @test_nowarn μ(x)
    @test F isa FeatureArray{<:Float64}

    F = 𝒇(x)
    @test F isa FeatureArray{<:Float64}
    @test F ≈ [0 0; 1 1]
end

@testset "ACF and PACF" begin
    X = randn(1000, 10)
    _acf = mapslices(x -> autocor(x, TimeseriesFeatures.ac_lags; demean = true), X;
                     dims = 1)
    @test all(AC(X) .== _acf)
    _pacf = mapslices(x -> pacf(x, TimeseriesFeatures.ac_lags; method = :regression), X;
                      dims = 1)
    @test all(Partial_AC(X) .== _pacf)
end

@testset "PACF superfeatures" begin
    X = randn(1000, 10)
    lags = TimeseriesFeatures.ac_lags
    AC_slow = FeatureSet([x -> autocor(x, [ℓ]; demean = true)[1]::Float64 for ℓ in lags],
                         Symbol.(["AC_$ℓ" for ℓ in lags]),
                         [["correlation"] for ℓ in lags],
                         ["Autocorrelation at lag $ℓ" for ℓ in lags])
    AC_partial_slow = FeatureSet([x -> pacf(x, [ℓ]; method = :regression)[1]::Float64
                                  for ℓ in lags],
                                 Symbol.(["AC_partial_$ℓ" for ℓ in lags]),
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

@testset "RAD" begin
    x = sin.(0.01:0.01:10)
    r = autocor(x, 1:(length(x) - 1))
    τ = TimeseriesFeatures.firstcrossingacf(x)
    @test 161 < τ < 163
    @test_nowarn CR_RAD(x)
end

@testset "PairwiseFeatures" begin
    X = randn(1000, 5)
    𝑓 = Pearson
    f = @test_nowarn 𝑓(X)

    X = DimArray(randn(100, 2), (Dim{:x}(1:100), Dim{:var}(1:2)))
    f = @test_nowarn 𝑓(X)
    @test dims(f, 1) == dims(X, 2) == dims(f, 2)

    𝒇 = FeatureSet([Pearson, Covariance])
    @test 𝒇(X) isa FeatureArray
end

@testset "MultivariateFeatures" begin
    X = DimArray(randn(100000, 20), (Dim{:x}(1:100000), Dim{:var}(1:20)))
    @test all(isapprox.(Covariance_svd(X), Covariance(X), atol = 1e-4))
    @time f1 = Covariance(X) # Much faster
    @time f2 = Covariance_svd(X) # Much faster
    @time cov(X) # Faster again
end

@testset "CausalityToolsExt" begin
    X = randn(1000, 2)
    F = @test_nowarn MI_Lord_NN_20(X)
    @test F[2] < 0.5

    x = sin.(0.01:0.01:10)
    y = cos.(0.01:0.01:10)
    F = @test_nowarn MI_Lord_NN_20([x y])
    @test F[2] > 7
end

@testset "Super" begin
    using StatsBase, TimeseriesFeatures, Test
    𝐱 = rand(1000, 2)
    μ = Feature(mean, :μ, ["0"], "Mean value of the time series")
    σ = Feature(std, :σ, ["1"], "Standard deviation of the time series")
    μ_z = @test_nowarn Super(μ, TimeseriesFeatures.zᶠ)
    σ_z = @test_nowarn Super(σ, TimeseriesFeatures.zᶠ)
    @test μ_z isa Super
    @test μ_z(𝐱)≈[0 0] atol=1e-13
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
                                                                                    TimeseriesFeatures.zᶠ,
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

@testset "PPC" begin
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
end
