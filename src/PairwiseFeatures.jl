module PairwiseFeatures
using Statistics
export SPI, PairwiseFeature, SPISet, PairwiseFeatureSet, AbstractPairwiseFeature
import ..Features: AbstractFeature, Feature
import ..FeatureSets: AbstractFeatureSet, FeatureSet, getnames, getname
import ..FeatureArrays: FeatureArray, FeatureVector, _featuredim
import ..SuperFeatures: AbstractSuper, Super, getsuper, getmethod
using ..DimensionalData
export Pearson, Covariance

abstract type AbstractPairwiseFeature <: AbstractFeature end

Base.@kwdef struct PairwiseFeature <: AbstractPairwiseFeature
    method::Function # For an SPI, this should be (x, y) -> f(x, y)
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
const SPI = PairwiseFeature
function PairwiseFeature(method::Function, name = Symbol(method),
                         keywords::Vector{String} = [""], description::String = "")
    PairwiseFeature(; method, name, keywords, description)
end
function PairwiseFeature(method::Function, name, description::String,
                         keywords::Vector{String} = [""])
    PairwiseFeature(; method, name, keywords, description)
end

(𝑓::AbstractPairwiseFeature)(x::AbstractVector) = getmethod(𝑓)(x, x)
function (𝑓::AbstractPairwiseFeature)(X::AbstractArray)
    idxs = CartesianIndices(size(X)[2:end])
    idxs = Iterators.product(idxs, idxs)
    f = i -> getmethod(𝑓)(X[:, first(i)], X[:, last(i)])
    f.(idxs)
end
function (𝑓::AbstractPairwiseFeature)(X::DimensionalData.AbstractDimMatrix)
    DimArray(𝑓(X.data), (dims(X, 2), dims(X, 2)))
end
function (𝑓::AbstractPairwiseFeature)(X::AbstractVector{<:AbstractVector})
    # D = _featuredim([getname(𝑓)])
    idxs = CartesianIndices(X)
    idxs = Iterators.product(idxs, idxs)
    f = i -> getmethod(𝑓)(X[first(i)], X[last(i)])
    f.(idxs)
end
function (𝑓::AbstractPairwiseFeature)(X::AbstractDimVector{<:AbstractVector})
    D = dims(X, 1) # _featuredim([getname(𝑓)])
    DimArray(𝑓(parent(X)), (D, D))
end

const PairwiseFeatureSet = FeatureSet{<:AbstractPairwiseFeature}
const SPISet = FeatureSet{<:AbstractPairwiseFeature}

function (𝒇::PairwiseFeatureSet)(x::AbstractMatrix)
    DimArray(permutedims((cat(FeatureVector([𝑓(x) for 𝑓 in 𝒇], 𝒇)...; dims = ndims(x) + 1)),
                         [ndims(x) + 1, 1:ndims(x)]),
             (_featuredim(getnames(𝒇)), DimensionalData.AnonDim(),
              DimensionalData.AnonDim())) |> FeatureArray
end
function (𝒇::PairwiseFeatureSet)(x::DimensionalData.AbstractDimMatrix)
    DimArray(permutedims((cat(FeatureVector([𝑓(x) for 𝑓 in 𝒇], 𝒇)...; dims = ndims(x) + 1)),
                         [3, 1, 2]),
             (_featuredim(getnames(𝒇)), dims(x, 2), dims(x, 2))) |> FeatureArray
end

# TODO Write tests for this

Pearson = SPI((x, y) -> cor(x, y), :Pearson, "Pearson correlation coefficient",
              ["correlation"])
Covariance = SPI((x, y) -> cov(x, y), :Pearson, "Sample covariance", ["covariance"])

# function (𝑓::AbstractSuper{F, S})(x::AbstractVector) where {F <: AbstractPairwiseFeature,
#                                                             S <: AbstractFeature}
#     y = getsuper(𝑓)(x)
#     getfeature(𝑓)(y, y)
# end
# function (𝒇::SuperFeatureSet)(x::AbstractVector{<:Number})::FeatureVector
#     ℱ = getsuper.(𝒇) |> unique |> SuperFeatureSet
#     supervals = ℱ(x)
#     FeatureVector([superloop(𝑓, supervals) for 𝑓 ∈ 𝒇], 𝒇)
# end
# (𝑓::AbstractSuper{F,S})(X::AbstractArray) where {F<:AbstractPairwiseFeature,S<:AbstractFeature}
# (𝑓::AbstractSuper{F,S})(X::AbstractDimArray) where {F<:AbstractPairwiseFeature,S<:AbstractFeature} = _construct(𝑓, mapslices(getmethod(𝑓) ∘ getsuper(𝑓), X; dims=1))

end # module
