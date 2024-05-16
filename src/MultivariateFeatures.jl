@reexport module MultivariateFeatures
using Statistics
using LinearAlgebra
import ..Features: AbstractFeature, Feature, getmethod, getname
import ..FeatureSets: AbstractFeatureSet, FeatureSet
import ..FeatureArrays: _featuredim
import ..PairwiseFeatures: AbstractPairwiseFeature
using ..DimensionalData
export MultivariateFeature, MultivariateFeatureSet, AbstractMultivariateFeature,
       PairwiseOrMultivariate
export Covariance_svd, Pearson_svd

abstract type AbstractMultivariateFeature <: AbstractFeature end

Base.@kwdef struct MultivariateFeature <: AbstractMultivariateFeature
    method::Function # For an MultivariateFeature, this should be X -> f(X), X is a matrix
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
function MultivariateFeature(method::Function, name = Symbol(method),
                             keywords::Vector{String} = [""], description::String = "")
    Feature(; method, name, keywords, description)
end
function MultivariateFeature(method::Function, name, description::String,
                             keywords::Vector{String} = [""])
    Feature(; method, name, keywords, description)
end

(𝑓::AbstractMultivariateFeature)(X::AbstractMatrix) = getmethod(𝑓)(X)

function (𝑓::AbstractMultivariateFeature)(X::AbstractArray)
    idxs = CartesianIndices(size(X)[3:end])
    idxs = Iterators.product(idxs, idxs)
    f = i -> getmethod(𝑓)(X[:, first(i)], X[:, last(i)])
    f.(idxs)
end
function (𝑓::AbstractMultivariateFeature)(X::DimensionalData.AbstractDimMatrix)
    DimArray(𝑓(X.data), (dims(X, 2), dims(X, 2)))
end

const PairwiseOrMultivariate = Union{<:AbstractMultivariateFeature,
                                     <:AbstractPairwiseFeature}
const MultivariateFeatureSet = FeatureSet{<:PairwiseOrMultivariate}

function (𝒇::MultivariateFeatureSet)(x::AbstractMatrix)
    DimArray(permutedims((cat(FeatureVector([𝑓(x) for 𝑓 in 𝒇], 𝒇)...; dims = ndims(x) + 1)),
                         [ndims(x) + 1, (1:ndims(x))...]),
             (_featuredim(getnames(𝒇)), DimensionalData.AnonDim(),
              DimensionalData.AnonDim())) |> FeatureArray
end
function (𝒇::MultivariateFeatureSet)(x::DimensionalData.AbstractDimMatrix)
    DimArray(permutedims((cat(FeatureVector([𝑓(x |> collect) for 𝑓 in 𝒇], 𝒇)...;
                              dims = ndims(x) + 1)), [ndims(x) + 1, (1:ndims(x))...]),
             (_featuredim(getnames(𝒇)), dims(x, 2), dims(x, 2))) |> FeatureArray
end

# function svdcovariance(X)
#     U, S, V = svd(X')
#     S = Diagonal(S)
#     (U * S * S' * U') / (size(X, 1) - 1)
# end
Covariance_svd = MultivariateFeature(X -> cov(X), :Covariance_svd, "Sample covariance",
                                     ["covariance"])
Pearson_svd = MultivariateFeature(X -> cor(X), :Pearson_svd,
                                  "Pearson correlation coefficient", ["correlation"])

end
