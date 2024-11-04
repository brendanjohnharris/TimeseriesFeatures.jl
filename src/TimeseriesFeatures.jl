module TimeseriesFeatures
using DimensionalData
using Reexport
using Requires
using LinearAlgebra
import Statistics: mean, std, cov

function __init__()
    @require StatsBase="2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" begin
        @eval include("../ext/StatsBaseExt.jl")
    end
    @require CausalityTools="5520caf5-2dd7-5c5d-bfcb-a00e56ac49f7" begin
        @eval include("../ext/CausalityToolsExt.jl")
    end
    @require DSP="717857b8-e6f2-59f4-9121-6e50c889abd2" begin
        @eval include("../ext/DSPExt.jl")
    end
    @require Distances="b4f34e82-e78d-54a5-968a-f98e89d6e8f7" begin
        @eval include("../ext/DistancesExt.jl")
    end
end

include("Features.jl")
include("FeatureSets.jl")
include("FeatureArrays.jl")
include("SuperFeatures.jl")
include("PairwiseFeatures.jl")
include("MultivariateFeatures.jl")

z_score(𝐱::AbstractVector) = (𝐱 .- mean(𝐱)) ./ (std(𝐱))
zᶠ = Feature(TimeseriesFeatures.z_score, :z_score, ["normalization"], "𝐱 → (𝐱 - μ(𝐱))/σ(𝐱)")

end
