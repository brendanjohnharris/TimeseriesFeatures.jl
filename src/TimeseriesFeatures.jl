module TimeseriesFeatures
using DimensionalData
using Reexport
using Requires
using LinearAlgebra
import Statistics: mean, std, cov

function __init__()
    @require StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91" begin
        @eval include("Autocorrelations.jl")
    end
end

include("Features.jl")
include("FeatureSets.jl")
include("FeatureArrays.jl")
include("SuperFeatures.jl")

z_score(𝐱::AbstractVector) = (𝐱 .- mean(𝐱)) ./ (std(𝐱))
zᶠ = Feature(TimeseriesFeatures.z_score, :z_score, ["normalization"], "𝐱 → (𝐱 - μ(𝐱))/σ(𝐱)")


end
