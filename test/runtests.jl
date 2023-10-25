using TimeseriesFeatures
using Test
using DimensionalData
using Statistics
using StatsBase

X = randn(1000, 5)


μ = Feature(mean, :mean, ["distribution"], "μ")
σ = Feature(std, :std, ["distribution"], "σ")
𝒇₁ = FeatureSet([sum, length], [:sum, :length], [["distribution"], ["sampling"]], ["∑x¹", "∑x⁰"])
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

@testset "DimArrays" begin
    x = DimArray(randn(100), (Dim{:x}(1:100),))
    @test σ(x)[:std] == σ(x |> vec)
    @test 𝒇(x) == 𝒇(x |> vec)
end

@testset "SuperFeatures" begin
    𝐱 = rand(1000, 2)
    @test_nowarn TimeseriesFeatures.zᶠ(𝐱)
    μ = SuperFeature(mean, :μ, ["0"], "Mean value of the z-scored time series", super=TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, ["1"], "Standard deviation of the z-scored time series"; super=TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])
    @test all(isapprox.(𝒇(𝐱), [0.0 0.0; 1.0 1.0]; atol=1e-9))
end

@testset "ACF and PACF" begin
    X = randn(1000, 10)
    _acf = mapslices(x -> autocor(x, TimeseriesFeatures.ac_lags; demean=true), X; dims=1)
    @test all(ac(X) .== _acf)
    _pacf = mapslices(x -> pacf(x, TimeseriesFeatures.ac_lags; method=:regression), X; dims=1)
    @test all(partial_ac(X) .== _pacf)
end

@testset "PACF superfeatures" begin
    X = randn(1000, 10)
    lags = TimeseriesFeatures.ac_lags
    AC_slow = FeatureSet([x -> autocor(x, [ℓ]; demean=true)[1]::Float64 for ℓ ∈ lags],
        Symbol.(["AC_$ℓ" for ℓ ∈ lags]),
        [["correlation"] for ℓ ∈ lags],
        ["Autocorrelation at lag $ℓ" for ℓ ∈ lags])
    AC_partial_slow = FeatureSet([x -> pacf(x, [ℓ]; method=:regression)[1]::Float64 for ℓ ∈ lags],
        Symbol.(["AC_partial_$ℓ" for ℓ ∈ lags]),
        [["correlation"] for ℓ ∈ lags],
        ["Partial autocorrelation at lag $ℓ (regression method)" for ℓ ∈ lags])

    @test all(AC_slow(X) .== ac(X))
    @test all(AC_partial_slow(X) .== partial_ac(X))
    println("\nFeature autocorrelation: ")
    @time AC_slow(X)
    println("\nSuperFeature autocorrelation: ")
    @time ac(X)
    println("\nFeature partial autocorrelation: ")
    @time AC_partial_slow(X)
    println("\nSuperfeature partial autocorrelation: ")
    @time partial_ac(X)
end

@testset "RAD" begin
    x = sin.(0.01:0.01:10)
    r = autocor(x, 1:length(x)-1)
    τ = TimeseriesFeatures.firstcrossing(x)
    @test 161 < τ < 163
    @test_nowarn CR_RAD(x)
end
