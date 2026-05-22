# Changelog

## `0.7.0`

### Added
- **Breaking**: MoreMaps.jl threading backend; `chart` keyword on feature-set evaluation selects sequential/threaded/distributed/Dagger execution and progress logging.
- The default `Chart` is `Sequential()`, so unlike earlier versions, multithreading is now opt-in via `chart = Chart(Threaded())`.

### Changed
- `LabelledFeatureArray` API: `(F, 𝒇; x = X)` → `(X, F, 𝒇)`.
- `SuperFeatureSet` accepts any `AbstractVector{<:AbstractFeature}` and lifts via `SuperFeature.()`.
- Reworked `FeatureSet` `show`.

### Fixed
- `+`/`\`/`∩`/`∪` when mixing `FeatureSet` and `SuperFeatureSet`.
- `FeatureArray` rebuild via `stack` on Julia 1.12.
- Pairwise feature evaluation paths.

## v0.6.1 — 2025-03-05

### Added
- `CR_RAD_raw` (uncentred RAD variant); both now call `RAD(x, τ, doAbs)` explicitly.
- `RAD` strips `AbstractDimArray` to `parent(z)`.
- Julia 1.10 in CI matrix.

### Changed
- Tightened `FeatureArray` promote rules (PR #14, @KristofferC).
- `Feature(::Matrix)` signature tweak for Julia nightly.
- Rounding-conversion fix for Julia 1.10.

### Fixed
- `firstcrossingacf` rounding on 1.10; macOS test tolerance relaxed.

### Dependencies
- `codecov/codecov-action` 4 → 5.

## v0.6.0 — 2024-12-16

**Breaking.** Type-stability overhaul (closes #10).

### Added
- `Identity` feature exported from `Features`.
- Parametric `Feature{F}` (method as type parameter).
- Parametric `SuperFeature{F,G}` storing `feature::F` and `super::G`; `SuperFeature(feature, super; merge)` constructor.
- `SuperPairwiseFeature` alias and `PairwiseSuperFeatureSet` constructor.
- `maybe_autocor` / `maybe_pacf` / `maybe_median` stubs in `src/StatsBase.jl`, filled in by `StatsBaseExt`.
- Tuple input `NTuple{2, AbstractVector{<:Number}}` for pairwise features.
- `return_type` argument on pairwise feature-set calls.

### Changed (breaking)
- `FeatureArray` strongly typed; feature dim is now a custom `FeatDim` (`@dim Feat FeatDim "Feature"`), replacing `Dim{:feature}`. `getdim`/`setdim` exports dropped.
- `(::Feature)(::AbstractArray)` uses `eachslice(X; dims=2:ndims(X), drop)` instead of `mapslices`.
- `==` on features replaced by `Base.isequal`.
- `MultivariateFeatures` module removed (subsumed by `PairwiseFeature` paths).
- `StatsBaseExt` is now a proper extension module; autocorrelation/RAD feature definitions moved into `src/StatsBase.jl`.
- New `TimeseriesToolsExt` extension; `TimeseriesTools` added as weak dep (paired with `StatsBase`).
- `Pearson` / `Covariance` use `cor(collect(x), collect(y))` for `AbstractVector` compatibility.

### Fixed
- Multithreaded progress logging in `SuperFeatureSet`.
- Numerous type-instability hotspots in `FeatureArray` construction and feature evaluation.

## v0.5.3 — 2024-12-10

### Fixed
- Multithreaded progress logging.

## v0.5.2 — 2024-12-10

### Added
- `Base.sum` on `Feature` / `FeatureSet`.

### Fixed
- Various `SuperFeatures` evaluation paths surfaced by the new test runner.

## v0.5.1 — 2024-12-10

### Added
- TestItemRunner.jl integration (`@testitem`, `@run_package_tests`).

## v0.5.0 — 2024-12-10

### Added (breaking)
- Julia 1.9 package extensions replace `Requires.jl`: `StatsBaseExt`, `TimeseriesToolsExt` (plus `AssociationsExt` from v0.4.4).
- `src/StatsBase.jl` holds feature definitions; `maybe_*` stubs activate when StatsBase is loaded.
- `Identity` super feature.
- `SuperFeatureSet` redefined as `const SuperFeatureSet = FeatureSet{<:AbstractSuperFeature}`.

### Changed (breaking)
- `FeatureSet` arithmetic (`+`/`\`/`∩`/`∪`) generalised over `AbstractFeatureSet` with `promote_rule`/`convert` lifting `Feature` → `SuperFeature`; mixed feature/feature-set ops supported.
- `getindex(𝒇::AbstractFeatureSet, I)` returns a `SuperFeatureSet` for non-scalar `I`.

### Fixed
- Eltype promotion mixing `Feature` and `SuperFeature`.

## v0.4.4 — 2024-11-05

### Changed
- CausalityTools → Associations migration; `MI_Lord_NN_20` now uses `Associations.mutualinfo(LordEstimator(k=20), …)`. `Associations = "4"` in compat.

## v0.4.3 — 2024-08-19

### Fixed
- DimensionalData `name` behaviour change.
- Multithreading correctness for `SuperFeatures` (multithread CI tests disabled while validating).
- `dimconstructor` on `FeatDim` returns `FeatureArray` (not plain `DimArray`).

## v0.4.2 — 2024-08-13

### Changed
- DimensionalData compat → v0.28.

## v0.4.1 — 2024-06-19

### Changed
- DimensionalData compat → v0.29.
- Exploratory Distances.jl integration.

### Dependencies
- `actions/checkout` 3 → 4; `julia-actions/setup-julia` 1 → 2; `codecov/codecov-action` 3 → 4.

## v0.4.0 — 2024-01-22

### Added
- `FeatureSet` set operations (`+`/`\`/`∩`/`∪`) with consistent naming.
- `(::Feature)(::AbstractArray)` dispatch for vectors-of-vectors and arbitrary-dim arrays.

### Fixed
- Multidimensional array support.
- `DimArray` indexing/construction matches plain `Array`.
- `rebuild` when dims are `nothing`.

## v0.3.0 — 2024-01-22

### Added
- Tests for `PairwiseFeatures` and `MultivariateFeatures`.
- `MultivariateFeature` constructor returns a `Feature`.

## v0.2.0 — 2024-01-22

First substantive release after v0.1.0.

### Added
- `PairwiseFeatures`: `AbstractPairwiseFeature`, `PairwiseFeature`, `SPI` alias, `PairwiseFeatureSet`, `SPISet`; built-in `Pearson`, `Covariance`; composes with `Super`.
- `MultivariateFeatures`: `MultivariateFeature`, `MultivariateFeatureSet`, `PairwiseOrMultivariate`; built-in `Covariance_svd`, `Pearson_svd`.
- `StatsBaseExt`: rewritten from old `src/Autocorrelations.jl`; adds `ACF`, `AC`, `PACF`, `Partial_AC`, `firstcrossing`, `firstcrossingacf`, `RAD`, `CR_RAD`.
- `CausalityToolsExt`: `MI_Lord_NN_20` (Lord nearest-neighbour mutual info).
- `DSPExt`: `Analytic_Signal`, `Analytic_Phase`, `Analytic_Amplitude`; `pairwisephaseconsistency` (Vinck 2010, threaded); `phaselockingvalue`; built-ins `PPC`, `PPC_Analytic_Phase`, `PLV`, `PLV_Analytic_Phase`.
- `.JuliaFormatter.toml`, `dependabot.yml`, `Register.yml`, `test/Project.toml`.

### Removed
- `src/Autocorrelations.jl`, `src/Pairwise.jl` placeholder.

### Changed
- Top-level `__init__` `@require`s `StatsBase`, `CausalityTools`, `DSP`.
- `FeatureArrays` and `SuperFeatures` heavily refactored.

## v0.1.0 — 2023-10-25

### Added
- Zenodo DOI badge.

## v0.0.1 — 2023-10-25

Initial release, forked from Catch22.jl and split via PkgTemplates.

### Added
- `Features`, `FeatureSets`, `FeatureArrays`, `SuperFeatures`.
- `Autocorrelations` (loaded via Requires from StatsBase): `ACF`, `ac`, `PACF`, `partial_ac`, `firstcrossing`.
- `zᶠ` z-score feature.
- Docs scaffold; CI / CompatHelper / TagBot workflows.
