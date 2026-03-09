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
    using TimeseriesFeatures.SuperFeatures.MoreMaps

    x = rand(1000)
    xx = [rand(1000) for _ in 1:10]
    X = rand(1000, 10)
    XX = rand(1000, 3, 4)
    xX = [rand(1000, 3) for _ in 1:4]
    Xd = DimArray(rand(100, 2), (Dim{:x}(1:100), Dim{:var}(1:2)); name = "name")
    μ = Feature(mean, :mean, "μ", ["distribution"])
    σ = Feature(std, :std, "σ", ["distribution"])
    slow = Feature(x -> (sleep(1); sum(x)), :slow, "Slow feature", ["distribution"])
    flow = FeatureSet([μ, σ, slow])
    _fast1 = Feature(x -> 1.0, :fast1, "Fast feature", ["distribution"])
    _fast2 = Feature(x -> 2.0, :fast2, "Fast feature", ["distribution"])
    timestwo = Feature(x -> 2x, :timestwo, "Fast feature", ["distribution"])
    fast = SuperFeatureSet([SuperFeature(timestwo, _fast1), SuperFeature(timestwo, _fast2)])
    𝒇₁ = FeatureSet([sum, length], [:sum, :length], ["∑x¹", "∑x⁰"],
                    [["distribution"], ["sampling"]])
    𝒇 = FeatureSet([μ, σ]) + 𝒇₁
    𝒇₂ = FeatureSet([μ, σ])
    𝒇₃ = 𝒇₁ + 𝒇₂
    𝒇s = FeatureSet([SuperFeature(μ), σ])

    Ts = [Int32, Int64, Float32, Float64]
end

@testitem "Features" setup=[Setup] begin
    using Statistics, TimeseriesFeatures
    μ = @inferred Feature(mean, :mean, "μ", ["distribution"])
    @test μ isa Feature{typeof(mean)}

    _μ = @inferred Feature(mean, :mean, "_μ", ["distribution"]) # * Alternate constructor
    @test isequal(μ, _μ) # Compares names
    @test !==(μ, _μ) # Compares fields
    @test unique([μ, _μ]) == [μ] # Uses isequal

    _μ = @inferred Feature(mean, :_mean, "μ", ["distribution"])
    @test !isequal(μ, _μ) # Compares names
    @test !==(μ, _μ) # Compares fields
    @test unique([μ, _μ]) == [μ, _μ] # Uses isequal

    @test [μ, SuperFeature(μ)] isa Vector{SuperFeature} # * Vector causes a promotion
    for μ in (μ, SuperFeature(μ))
        @inferred Feature(μ)
        @test Feature(μ) == μ

        # * Calculate features
        if VERSION ≥ v"1.12.0"
            f = @inferred μ(x)
        else
            f = μ(x)
        end
        @test f isa Float64
        @test f≈0.5 atol=0.05

        if VERSION ≥ v"1.12.0"
            f = @inferred μ(xx)
        else
            f = μ(xx)
        end
        @test f isa Vector{Float64}
        @test length(f) == 10

        if VERSION ≥ v"1.12.0"
            f = @inferred μ(X)
        else
            f = μ(X)
        end
        @test f isa AbstractVector{Float64}
        @test length(f) == 10

        if VERSION ≥ v"1.12.0"
            f = @inferred μ(XX)
        else
            f = μ(XX)
        end
        @test f isa AbstractMatrix{Float64}
        @test size(f) == (3, 4)

        if VERSION ≥ v"1.12.0"
            f = @inferred μ(xX)
        else
            f = μ(xX)
        end
        @test f isa AbstractVector{<:AbstractVector{Float64}}
        @test length(f) == 4
        @test length(f[1]) == 3

        if VERSION ≥ v"1.12.0"
            map(Ts) do T
                @inferred μ(convert.(T, round.(x)))
                @inferred μ([convert.(T, round.(x)) for x in xx])
                Y = convert.(T, round.(X))
                @inferred μ(Y)
                @inferred μ(convert.(T, round.(XX)))
                yY = [convert.(T, round.(x)) for x in xX]
                @inferred map(μ, yY)
                @inferred μ(yY)
            end
        end
    end
end

@testitem "FeatureArrays" setup=[Setup] begin
    x = randn(10)
    d = Feat(DimensionalData.Categorical(Symbol.(1:length(x));
                                         order = DimensionalData.Unordered()))
    @inferred FeatureArray(x, DimensionalData.format((d,), x), (), DimensionalData.NoName(),
                           DimensionalData.NoMetadata())
    @inferred FeatureArray(x, DimensionalData.format((d,), x))
    f = @inferred FeatureArray(x, (d,))
    f = @inferred FeatureArray(x, Symbol.(1:length(x)))

    𝑓s = [:mean, :std]
    𝑓 = @inferred FeatureSet([μ, σ])

    F = @inferred 𝒇(X)
    @inferred F[1:2]
    @test F[𝑓] == F[𝑓s]
    @test F[𝑓] == F[1:2, :]
    @test all(F[𝑓s, :] .== F[1:2, :])

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

    @test Array(F) == parent(F)
end

@testitem "SuperFeatures" setup=[Setup] begin
    mu = @inferred SuperFeature(μ)
    @inferred SuperFeature(μ, σ)

    @inferred getsuper(mu)
    @inferred getfeature(mu)
end

@testitem "FeatureSet" setup=[Setup] begin
    @test 𝒇₃ isa FeatureSet

    @inferred getfeatures(𝒇₃)
    _ms = [(@inferred getmethod(f)) for f in 𝒇₃]
    ms = @inferred getmethods(𝒇₃)
    @test _ms == ms
    @inferred getnames(𝒇₃)
    @inferred getkeywords(𝒇₃)
    @inferred getdescriptions(𝒇₃)
    @inferred size(𝒇₃)

    @inferred 𝒇₁(X)
    @inferred 𝒇₃(X)
    @test getnames(𝒇₃) == [:sum, :length, :mean, :std]

    ff = @inferred 𝒇₃[[:mean, :sum]]
    𝒈 = deepcopy(𝒇)
    @test_nowarn 𝒈[3] = σ

    # ff = @inferred getfeatures(𝒇)
    # @inferred ff[[3]]
    # @inferred 𝒇[[3]]
    # @inferred 𝒇₃[1]
    # @inferred 𝒇₃[:sum]

    @test getname(𝒇₃[:sum]) == :sum
    @test all([getname(𝒇₃[x]) == x for x in getnames(𝒇₃)])
    # @inferred 𝒇₃(X)[:sum, :] # ! Not type stable!
    @test 𝒇₃(X)[:sum] == 𝒇₃(X)[:sum, :]

    @test hcat(eachslice(𝒇₃(X), dims = 2)...) isa FeatureArray # Check rebuild is ok (does not convert to DimArray

    F = 𝒇₃(X)[:, 1]
    𝑓 = [:sum, :length]
    @inferred getindex(F, 𝑓[1])

    F = 𝒇₃(X)
    # @inferred getindex(F, 𝑓[1]) # ! Not type stable
    @inferred getindex(F, 1:2)
    # @inferred getindex(F, 𝑓) # ! Not typestable

    # @inferred 𝒇₃(X)[[:sum, :length], :]
    @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :]
    @inferred TimeseriesFeatures.SuperFeatures.promote_eltype(𝒇₃, 𝒇₂)
    @test 𝒇₁ == 𝒇₃ \ 𝒇₂ == setdiff(𝒇₃, 𝒇₂)
    @test 𝒇₃ \ 𝒇₂ isa FeatureSet
    @test 𝒇₃ == 𝒇₁ ∪ 𝒇₂ == union(𝒇₁, 𝒇₂)
    @test 𝒇₁ ∪ 𝒇₂ isa FeatureSet
    @test 𝒇₂ == 𝒇₃ ∩ 𝒇₂ == intersect(𝒇₃, 𝒇₂)
    @test 𝒇₃ ∩ 𝒇₂ isa FeatureSet

    # @inferred vcat(𝒇₃, 𝒇₂)
    # @inferred 𝒇₃ + 𝒇₂
    # @inferred 𝒇₃ + μ
    # @inferred 𝒇₃ \ 𝒇₂
    # @inferred 𝒇₃ ∪ 𝒇₂
    # @inferred 𝒇₃ ∩ 𝒇₂

    # @inferred vcat(𝒇₂, 𝒇s)
    # @inferred 𝒇₂ + 𝒇s
    # @inferred 𝒇₂ + μ
    # @inferred 𝒇₃ \ 𝒇s
    # @inferred 𝒇₂ ∪ 𝒇s
    # @inferred 𝒇₂ ∩ 𝒇s

    @test 𝒇₁ + μ isa FeatureSet
    @test μ + 𝒇₁ isa FeatureSet

    @test FeatureSet(mean, :mean, "μ", ["distribution"]) isa FeatureSet
    @test_nowarn show(𝒇₁)
    @test_nowarn display(𝒇₁)
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
    @test size(F) == (3, 3) # Extra dims are dropped
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

@testitem "SuperFeatures" setup=[Setup] begin
    x = rand(1000, 2)
    @test_nowarn TimeseriesFeatures.zᶠ(x)
    μ = SuperFeature(mean, :μ, "Mean value of the z-scored time series", ["0"],
                     TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, "Standard deviation of the z-scored time series", ["1"],
                     TimeseriesFeatures.zᶠ)
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

    # if Threads.nthreads() ≥ 8 # This will only be faster if the machine has a solid number of threads
    #     Z = randn(500000, 1000)
    #     Z = eachcol(Z)
    #     a = @benchmark 𝒇($Z)
    #     b = @benchmark map(𝒇, $Z)
    #     @test median(a.times) ≤ median(b.times) # Check multithreading works
    #     @test a.allocs ≤ b.allocs

    #     c = @benchmark map(𝒇, Chart(), Z)
    # end

    # @test 𝒇₃(X)[[:sum, :length]] == 𝒇₃(X)[[:sum, :length], :]

    @test vcat(𝒇, 𝒇) isa SuperFeatureSet
    @test vcat(𝒇, 𝒇₁) isa SuperFeatureSet

    # @inferred setdiff(𝒇, 𝒇)
    # @inferred setdiff(𝒇, 𝒇₁)
    @test setdiff(𝒇, 𝒇₁) isa SuperFeatureSet

    @test union(𝒇, 𝒇) isa SuperFeatureSet
    @test union(𝒇, 𝒇₁) isa SuperFeatureSet
    @test union(𝒇₁, 𝒇₁) isa FeatureSet

    @test intersect(𝒇, 𝒇) isa SuperFeatureSet
    @test intersect(𝒇, 𝒇₁) isa SuperFeatureSet
    @test isempty(intersect(𝒇, 𝒇₁))
    @test intersect(𝒇₁, 𝒇₁) isa FeatureSet
    @test !(intersect(union(𝒇, 𝒇₁), 𝒇₁) == 𝒇₁) # One's superfeatures, one is features...
    @test isequal(intersect(union(𝒇, 𝒇₁), 𝒇₁), 𝒇₁) # ...but they have the same names
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

    @test 𝒇s isa SuperFeatureSet
    @inferred getfeatures(𝒇s)
    _ms = [(@inferred getmethod(f)) for f in 𝒇s]
    ms = @inferred getmethods(𝒇s)
    @test _ms == ms
    @inferred getnames(𝒇s)
    @inferred getkeywords(𝒇s)
    @inferred getdescriptions(𝒇s)
    @inferred size(𝒇s)
end

@testitem "DimArrays" setup=[Setup] begin
    μ = Feature(mean, :mean, "μ", ["distribution"])
    σ = Feature(std, :std, "σ", ["distribution"])
    𝒇₁ = FeatureSet([sum, length], [:sum, :length], ["∑x¹", "∑x⁰"],
                    [["distribution"], ["sampling"]])
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
    @test dims(𝒇(x), 2) == dims(x, 2)

    μ = SuperFeature(mean, :μ, "Mean value of the z-scored time series", ["0"],
                     TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, "Standard deviation of the z-scored time series", ["1"],
                     TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])

    F = @test_nowarn σ(x)
    @test F isa DimArray{<:Float64}
    @test all(F .≈ 1.0)
    F = @test_nowarn μ(x)
    @test all(abs.(F) .< 1e-10)

    F = 𝒇(x)
    @test F isa FeatureArray{<:Float64}
    @test F ≈ [0 0; 1 1]
    @test dims(F, 2) == dims(x, 2)

    x = DimArray(rand(100, 2, 2), (Dim{:x}(1:100), Dim{:var}(1:2), Y(1:2)); name = n,
                 metadata = m)
    @test σ(x) == σ(x |> Array)
    @test 𝒇(x).data == 𝒇(x |> Array).data
    @test DimensionalData.metadata(𝒇(x)) == m
    @test DimensionalData.name(𝒇(x)) == n

    μ = SuperFeature(mean, :μ, "Mean value of the z-scored time series", ["0"],
                     TimeseriesFeatures.zᶠ)
    σ = SuperFeature(std, :σ, "Standard deviation of the z-scored time series", ["1"],
                     TimeseriesFeatures.zᶠ)
    𝒇 = SuperFeatureSet([μ, σ])

    F = @test_nowarn σ(x)
    if VERSION ≥ v"1.12.0"
        @inferred σ(x)
    end
    @test all(F .≈ 1.0)
    @test F isa DimArray{Float64}
    @test dims(F, 1) isa Dim{:var}
    @test dims(F, 2) isa Y
    F = @test_nowarn μ(x)
    @test F isa DimArray{Float64}
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
                         ["Autocorrelation at lag $ℓ" for ℓ in lags],
                         [["correlation"] for ℓ in lags])
    AC_partial_slow = FeatureSet([x -> pacf(x, [ℓ]; method = :regression)[1]::Float64
                                  for ℓ in lags],
                                 Symbol.(["Partial_AC_$ℓ" for ℓ in lags]),
                                 ["Partial autocorrelation at lag $ℓ (regression method)"
                                  for ℓ in lags],
                                 [["correlation"] for ℓ in lags])

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
    using LinearAlgebra
    X = randn(100000, 5)
    𝑓 = Pearson
    @test 1.0 == @inferred 𝑓(randn(100))
    f = @test_nowarn 𝑓(X)
    @inferred 𝑓(X)

    # @inferred SuperFeature(𝑓)
    @inferred 𝑓(X)
    @test 𝑓(X)≈I(5) atol=0.05

    X = DimArray(randn(100000, 2), (Dim{:x}(1:100000), Dim{:var}(1:2)))
    f = @test_nowarn 𝑓(X)
    @test dims(f, 1) == dims(X, 2) == dims(f, 2)

    𝒇 = FeatureSet([Pearson, Covariance])
    f = @inferred 𝒇(X)
    f = 𝒇(eachcol(X))
    f = 𝒇(X)
    @test 𝒇(X) isa FeatureArray
    @test 𝒇(X)[1] isa DimArray # * Then stack() them to get a 3D array
    @test stack(𝒇(X)) isa AbstractFeatureArray
    @test cat(𝒇(X); dims = 1) isa AbstractFeatureArray
    @test stack(𝒇(X)) isa FeatureArray
    @test stack(𝒇(X))[:Pearson] isa DimArray
    @test stack(𝒇(X))[:Pearson]≈[1 0; 0 1] atol=0.05

    X = randn(10000, 5)
    f = @inferred 𝒇(eachcol(X))
    f = @inferred 𝒇(X)
    @test f[:Pearson]≈I(5) atol=0.1
    @test f[:Covariance]≈cov(X) atol=1e-3

    # ! Now SuperPairwiseFeatureSets
    𝒇 = SuperFeature.(𝒇, [timestwo]) |> PairwiseSuperFeatureSet
    @test 𝒇 isa SuperPairwiseFeatureSet
    # f = @inferred 𝒇(x, x)
    f = @test_nowarn 𝒇(x, x)
    @test f[:Covariance] == Covariance(2.0 * x) # SuperFeature is applied
    f = 𝒇(X)
    @test 𝒇(X) isa FeatureArray

    X = DimArray(randn(100000, 2), (Dim{:x}(1:100000), Dim{:var}(1:2)))
    @test 𝒇(X)[1] isa DimArray # * Then stack() them to get a 3D array
    @test stack(𝒇(X)) isa FeatureArray
    @test stack(𝒇(X))[:Pearson] isa DimArray
    @test stack(𝒇(X))[:Pearson]≈[1 0; 0 1] atol=0.1

    X = randn(100000, 5)
    f = @inferred 𝒇(eachcol(X))
    # f = @inferred 𝒇(X)
    @test f[:Pearson]≈I(5) atol=0.05
    @test f[:Covariance]≈cov(2.0 .* X) atol=1e-4
end

# @testitem "MultivariateFeatures" setup=[Setup] begin
#     X = DimArray(randn(100000, 20), (Dim{:x}(1:100000), Dim{:var}(1:20)))
#     @test all(isapprox.(Covariance_svd(X), Covariance(X), atol = 1e-4))
#     @time f1 = Covariance(X) # Much faster
#     @time f2 = Covariance_svd(X) # Much faster
#     @time cov(X) # Faster again
# end

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
    μ = Feature(mean, :μ, "Mean value of the time series", ["0"])
    σ = Feature(std, :σ, "Standard deviation of the time series", ["1"])
    μ_z = @test_nowarn SuperFeature(μ, TimeseriesFeatures.zᶠ; merge = true)
    σ_z = @test_nowarn SuperFeature(σ, TimeseriesFeatures.zᶠ; merge = true)
    @test μ_z isa SuperFeature
    @test μ_z(𝐱)≈[0, 0] atol=1e-13
    𝒇 = SuperFeatureSet([μ_z, σ_z])
    @test all(isapprox.(𝒇(𝐱), [0.0 0.0; 1.0 1.0]; atol = 1e-9))

    # Check speed
    μ = [Feature(mean, Symbol("μ_$i"), "Mean value of the time series", ["0"])
         for i in 1:100]
    superfeature = @test_nowarn SuperFeatureSet(SuperFeature.(μ, [TimeseriesFeatures.zᶠ];
                                                              merge = true))
    feature = [Feature(x -> (zscore(x)), Symbol("μ_$i"),
                       "Mean value of the time series", ["0"]) for i in 1:100]

    setup = (superfeature = SuperFeatureSet(SuperFeature.(μ, [TimeseriesFeatures.zᶠ];
                                                          merge = true));
             𝐱 = rand(1000, 2))
    a = @benchmark superfeature(𝐱; chart = Chart()) setup=setup

    setup = (feature = [Feature(x -> (zscore(x)), Symbol("μ_$i"),
                                "Mean value of the time series", ["0"])
                        for i in 1:100];
             𝐱 = rand(1000, 2))
    b = @benchmark [f(𝐱) for f in feature] setup=setup
    @test median(a.times) < median(b.times) / 1.5

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
    @test F == PPC_Analytic_Phase(eachcol(X))

    PFS = PairwiseSuperFeatureSet([PPC_Analytic_Phase, PLV_Analytic_Phase])
    @test PFS isa SuperPairwiseFeatureSet
    @test PFS(x)[1] == PPC_Analytic_Phase(x) == PFS[1](x)
end

@testitem "TimeseriesToolsExt" setup=[Setup] begin
    using StatsBase
    using TimeseriesTools
    x = colorednoise(0.1:0.1:10000)
    @test_nowarn TimeseriesTools.timescale(x; method = :ac_crossing)

    x = set(x, sin.(times(x) ./ 2π))
    τ = TimeseriesTools.timescale(x) # This is 1/4 the period; i.e. the time shift it requires to become anti-phase
    y = TimeseriesTools.TimeseriesBase.Operators.ℬ(x, Int(τ ÷ step(x)))
    @test cor(x, y)≈0 atol=0.05
end

@testitem "Calculation type stability" setup=[Setup] begin
    𝒇s = SuperFeature.(𝒇₃) |> SuperFeatureSet

    # * Features
    @inferred getmethod(μ)(x)
    if VERSION ≥ v"1.12.0"
        f = @inferred μ(x)
        @inferred μ(x)
        @inferred μ(xx)
        @inferred μ(X)
    end

    # * Super Features
    @inferred SuperFeature(μ, TimeseriesFeatures.zᶠ)
    𝑓 = SuperFeature(μ, TimeseriesFeatures.zᶠ)
    @test 𝑓(rand(1000))≈0.0 atol=1e-10
    if VERSION ≥ v"1.12.0"
        @inferred getsuper(𝑓)(x)

        @inferred 𝑓(x)
        @inferred 𝑓(xx)
        @inferred 𝑓(X)
    end

    @test all(abs.(𝑓(xx)) .< 1e-10)
    @test all(abs.(𝑓(X)) .< 1e-10)

    # * FeatureSets (x, xx, X)
    if VERSION ≥ v"1.12.0"
        @inferred 𝒇₃(x)
        @inferred 𝒇₃(X)
        @inferred 𝒇₃(XX)
        @inferred 𝒇₃(xx)

        @inferred 𝒇₃(x, Any)
        @inferred 𝒇₃(X, Any)
        @inferred 𝒇₃(XX, Any)
        @inferred 𝒇₃(xx, Any)

        # * SuperFeatureSets (x, xx, X)
        @inferred fast(x)
        @inferred fast(X)
        @inferred fast(xx)
        @inferred fast(XX)
    end
end
