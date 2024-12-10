module TimeseriesFeatures
using DimensionalData
using Requires
using LinearAlgebra
import Statistics: mean, std, cov

function __init__()
    @require Associations="614afb3a-e278-4863-8805-9959372b9ec2" begin
        @eval include("../ext/AssociationsExt.jl")
    end
    @require DSP="717857b8-e6f2-59f4-9121-6e50c889abd2" begin
        @eval include("../ext/DSPExt.jl")
    end
end

include("Features.jl")
include("FeatureSets.jl")
include("FeatureArrays.jl")
include("SuperFeatures.jl")
include("PairwiseFeatures.jl")
include("MultivariateFeatures.jl")

using .Features, .FeatureSets, .FeatureArrays, .SuperFeatures, .PairwiseFeatures,
      .MultivariateFeatures
include("StatsBase.jl")

z_score(𝐱::AbstractVector) = (𝐱 .- mean(𝐱)) ./ (std(𝐱))
zᶠ = Feature(TimeseriesFeatures.z_score, :z_score, ["normalization"], "𝐱 → (𝐱 - μ(𝐱))/σ(𝐱)")

export AbstractFeatureArray, AbstractFeatureVector, AbstractFeatureMatrix,
       FeatureArray, FeatureVector, FeatureMatrix, getdim, setdim, FeatDim, Feat
export AbstractFeature, Feature, getmethod, getname, getkeywords, getdescription
export AbstractFeatureSet, FeatureSet, getfeatures, getmethods, getnames, getkeywords,
       getdescriptions
export MultivariateFeature, MultivariateFeatureSet, AbstractMultivariateFeature,
       PairwiseOrMultivariate
export SPI, PairwiseFeature, SPISet, PairwiseFeatureSet, AbstractPairwiseFeature
export SuperFeature, SuperFeatureSet, Super, AbstractSuper, getsuper, getfeature
export Pearson, Covariance
export Covariance_svd, Pearson_svd

end
