# TimeseriesFeatures.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brendanjohnharris.github.io/TimeseriesFeatures.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brendanjohnharris.github.io/TimeseriesFeatures.jl/dev/)
[![Build Status](https://github.com/brendanjohnharris/TimeseriesFeatures.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/TimeseriesFeatures.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/TimeseriesFeatures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/TimeseriesFeatures.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14969549.svg)](https://doi.org/10.5281/zenodo.10039292)

A Julia package providing the core types for representing, composing, and evaluating time-series features. TimeseriesFeatures.jl functionality underpins packages such as [Catch22.jl](https://github.com/brendanjohnharris/Catch22.jl), and is built around three ideas: a `Feature` is a named function with a description and keywords; a `FeatureSet` is a collection of features that can be evaluated against one or many time series in a single call; and a `FeatureArray` is the labelled output, annotated with feature names along its first dimension (a subtype of [`AbstractDimArray`](https://github.com/rafaqz/DimensionalData.jl)).

The package also defines `SuperFeature`s for features that share a common preprocessing step (e.g. z-scoring) so that the shared computation is performed once per time series. All feature-set evaluations are dispatched through [MoreMaps.jl](https://github.com/brendanjohnharris/MoreMaps.jl), which provides configurable backends for sequential, threaded, distributed, and Dagger-based execution, as well as progress logging.

Below we provide a getting-started guide. For complete API details see the [package documentation](https://brendanjohnharris.github.io/TimeseriesFeatures.jl/stable/).

<br>

# Usage
## Installation
```julia
using Pkg
Pkg.add("TimeseriesFeatures")
using TimeseriesFeatures
```

## Defining a feature
A `Feature` is a function annotated with a name (a `Symbol`), a description, and a vector of keyword strings. The wrapped method should accept at minimum an `AbstractVector` of numbers:
```julia
𝑓 = Feature(sum, :sum, "Sum of time-series values", ["distribution"])
𝑓(1:10)              # 55, equivalent to sum(1:10)
getname(𝑓)           # :sum
getdescription(𝑓)    # "Sum of time-series values"
getkeywords(𝑓)       # ["distribution"]
```
Features are callable like ordinary functions. When called on a `Matrix` the wrapped method is applied column-wise (each column is treated as a time series), regardless of any matrix method the underlying function might define.

## Building a feature set
A `FeatureSet` groups features so they can be evaluated together. It can be constructed from a vector of `Feature`s, or from parallel vectors of methods, names, descriptions, and keywords:
```julia
𝒇 = FeatureSet([sum, length],
               [:sum, :length],
               ["∑x¹", "∑x⁰"],
               [["distribution"], ["sampling"]])
```
`FeatureSet`s support indexing by integer, by symbol, or by a vector of symbols, returning either a single `Feature` or a sub-`FeatureSet`:
```julia
𝒇[:sum]              # the :sum Feature
𝒇[[:sum, :length]]   # a 2-feature FeatureSet
```
They also support array-style set operations: `+` concatenates, `\` (or `setdiff`) removes, and `∩`/`∪` intersect/union by name. Two features are considered equal if their names match.

## Evaluating features
A single time series is provided as a `Vector{<:Number}`; multiple time series are stacked into the columns of a `Matrix{<:Number}` (or, more generally, into a higher-dimensional array where the first dimension indexes within a time series). For example:
```julia
𝐱 = randn(1000)        # one time series
X = randn(1000, 10)    # ten time series, one per column
```
Calling a feature set returns a `FeatureArray`:
```julia
𝐟 = 𝒇(𝐱)              # a FeatureVector of length 2
F = 𝒇(X)              # a 2×10 FeatureMatrix
```
A `FeatureArray` behaves like an `Array` but is annotated with feature names along its first dimension. Names can be retrieved with `getnames(F)` and used directly for indexing:
```julia
F[:sum]               # row of :sum values across the 10 time series
F[:sum, 3]            # :sum value for the third time series
```

## SuperFeatures: sharing preprocessing
When several features share an expensive preprocessing step — z-scoring, detrending, computing a power spectrum — it is wasteful to repeat that step for every feature. A `SuperFeature` pairs a feature with a `super` feature representing that preprocessing, and a `SuperFeatureSet` collects them. When evaluated, each unique super is computed once per time series and its result is reused across every feature that shares it:
```julia
zᶠ = TimeseriesFeatures.zᶠ                # built-in z-score SuperFeature
𝑓₁ = SuperFeature(mean, zᶠ; merge = true) # mean of z-scored input
𝑓₂ = SuperFeature(std,  zᶠ; merge = true) # std of z-scored input
𝒈  = SuperFeatureSet([𝑓₁, 𝑓₂])
𝒈(𝐱)  # z-score is computed once, then reused for both features
```
`SuperFeature`s can be nested (a super may itself be a `SuperFeature`); the display tree groups features by their root super for inspection.

## PairwiseFeatures
A `PairwiseFeature` represents a measure defined between two time series, such as a correlation or covariance. The two built-ins are `Pearson` and `Covariance`:
```julia
Pearson(𝐱, randn(1000))        # scalar Pearson correlation
Pearson(X)                     # 10×10 matrix of pairwise Pearson correlations
```
Bundle pairwise features into a `PairwiseFeatureSet` to evaluate several together. A `SuperPairwiseFeature` lets pairwise measures share preprocessing in the same way as univariate `SuperFeature`s.

<br>

# Parallelism and progress with MoreMaps.jl

Evaluating a feature set across many time series is dispatched through [MoreMaps.jl](https://github.com/brendanjohnharris/MoreMaps.jl). Pass a `chart` keyword to control the execution backend and progress reporting. The default is `Chart(Threaded(), ProgressLogger())`, i.e. multithreaded with `ProgressLogging.jl` output:
```julia
using MoreMaps
F = 𝒇(X)                                            # default: Threaded() + ProgressLogger()
F = 𝒇(X; chart = Chart())                           # plain sequential, no progress
F = 𝒇(X; chart = Chart(Sequential(), NoProgress())) # explicit sequential
```
Available backends include `Sequential()`, `Threaded()`, `Distributed()`, and `Daggermap()`; available progress loggers include `NoProgress()`, `LogLogger(n)`, `ProgressLogger()`, and `TermLogger()`. For example, to run distributed across worker processes with `@info`-style updates every 10 items:
```julia
using Distributed
addprocs(4)
@everywhere using TimeseriesFeatures
F = 𝒇(X; chart = Chart(Distributed(), LogLogger(10)))
```

<br>

# Extensions

TimeseriesFeatures.jl loads optional functionality via package extensions when supporting packages are present in the active environment:

- **`StatsBase`** — additional distributional features built on `StatsBase.jl`.
- **`DSP`** — spectral and filter-based features (loaded automatically when `DSP.jl` is imported).
- **`Associations`** — information-theoretic measures from `Associations.jl` (loaded automatically when `Associations.jl` is imported).
- **`TimeseriesTools`** — interoperability with `TimeseriesTools.jl` time-series types.

<br>

# Related packages

- [HCTSA.jl](https://github.com/brendanjohnharris/HCTSA.jl) — Julia wrapper for the [*pyhctsa*](https://github.com/DynamicsAndNeuralSystems/pyhctsa) feature set.
- [Catch22.jl](https://github.com/brendanjohnharris/Catch22.jl) — the 22 [*catch22*](https://github.com/DynamicsAndNeuralSystems/catch22) features wrapped as a `SuperFeatureSet` over this package.
- [CatchaMouse16.jl](https://github.com/brendanjohnharris/CatchaMouse16.jl) — the [*catchaMouse16*](https://github.com/DynamicsAndNeuralSystems/catchaMouse16) features for fMRI time-series analysis.
- [MoreMaps.jl](https://github.com/brendanjohnharris/MoreMaps.jl) — the mapping framework used to dispatch feature evaluation across backends.
- [DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl) — the labelled-array foundation that `FeatureArray` extends.
