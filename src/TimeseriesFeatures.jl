module TimeseriesFeatures
using DimensionalData
using LinearAlgebra
using MoreMaps
import Statistics: mean, std, cov

include("Features.jl")
include("FeatureSets.jl")
include("FeatureArrays.jl")
include("SuperFeatures.jl")
include("PairwiseFeatures.jl")

using .Features, .FeatureSets, .FeatureArrays, .SuperFeatures, .PairwiseFeatures
include("StatsBase.jl")
include("DSP.jl")
include("Associations.jl")

z_score(𝐱::AbstractVector) = (𝐱 .- mean(𝐱)) ./ (std(𝐱))
const zᶠ = Feature(
    TimeseriesFeatures.z_score, :z_score, "𝐱 → (𝐱 - μ(𝐱))/σ(𝐱)",
    ["normalization"]
)

export AbstractFeature, Feature, getmethod, getname, getkeywords, getdescription, Identity
export AbstractFeatureArray, AbstractFeatureVector, AbstractFeatureMatrix,
    FeatureArray, FeatureVector, FeatureMatrix, FeatDim, Feat
export AbstractFeatureSet, FeatureSet, getfeatures, getmethods, getnames, getkeywords,
    getdescriptions
export PairwiseFeature, PairwiseFeatureSet, AbstractPairwiseFeature, SuperPairwiseFeature,
    SuperPairwiseFeatureSet, PairwiseSuperFeatureSet
export SuperFeature, SuperFeatureSet, Super, AbstractSuper, getsuper, getfeature
export Pearson, Covariance

end
