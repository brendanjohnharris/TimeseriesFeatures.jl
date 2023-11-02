using Statistics
using LinearAlgebra
export MultivariateFeature, MultivariateFeatureSet
export SVDCovariance

abstract type AbstractMultivariateFeature <: AbstractFeature end

Base.@kwdef struct MultivariateFeature <: AbstractMultivariateFeature
    method::Function # For an MultivariateFeature, this should be X -> f(X), X is a matrix
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
MultivariateFeature(method::Function, name=Symbol(method), keywords::Vector{String}=[""], description::String="") = Feature(; method, name, keywords, description)
MultivariateFeature(method::Function, name, description::String, keywords::Vector{String}=[""]) = Feature(; method, name, keywords, description)


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

const MultivariateFeatureSet = FeatureSet{<:AbstractMultivariateFeature}

# function (𝒇::MultivariateFeatureSet)(x::AbstractMatrix)
#     DimArray(
#         permutedims((cat(FeatureVector([𝑓(x) for 𝑓 ∈ 𝒇], 𝒇)...; dims=ndims(x) + 1)), [3, 1, 2]),
#         (Dim{:feature}(getnames(𝒇)), DimensionalData.AnonDim(), DimensionalData.AnonDim())) |> FeatureArray
# end
# function (𝒇::PairwiseFeatureSet)(x::DimensionalData.AbstractDimMatrix)
#     DimArray(
#         permutedims((cat(FeatureVector([𝑓(x) for 𝑓 ∈ 𝒇], 𝒇)...; dims=ndims(x) + 1)), [3, 1, 2]),
#         (Dim{:feature}(getnames(𝒇)), dims(x, 2), dims(x, 2))) |> FeatureArray
# end

function svdcovariance(X)
    U, S, V = svd(X')
    S = Diagonal(S)
    (U * S * S' * U') / (size(X, 1) - 1)
end
SVDCovariance = MultivariateFeature(X -> svdcovariance(X), :SVDCovariance, "Sample covariance calculated with the singular-value decomposition", ["covariance"])
