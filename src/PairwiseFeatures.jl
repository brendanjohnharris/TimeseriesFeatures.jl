using Statistics
export SPI, PairwiseFeature, SPISet, PairwiseFeatureSet
export Pearson, Covariance

abstract type AbstractPairwiseFeature <: AbstractFeature end

Base.@kwdef struct PairwiseFeature <: AbstractPairwiseFeature
    method::Function # For an SPI, this should be (x, y) -> f(x, y)
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
const SPI = PairwiseFeature
PairwiseFeature(method::Function, name=Symbol(method), keywords::Vector{String}=[""], description::String="") = Feature(; method, name, keywords, description)
PairwiseFeature(method::Function, name, description::String, keywords::Vector{String}=[""]) = Feature(; method, name, keywords, description)


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

const PairwiseFeatureSet = FeatureSet{<:AbstractPairwiseFeature}
const SPISet = FeatureSet{<:AbstractPairwiseFeature}

function (𝒇::PairwiseFeatureSet)(x::AbstractMatrix)
    DimArray(
        permutedims((cat(FeatureVector([𝑓(x) for 𝑓 ∈ 𝒇], 𝒇)...; dims=ndims(x) + 1)), [3, 1, 2]),
        (Dim{:feature}(getnames(𝒇)), DimensionalData.AnonDim(), DimensionalData.AnonDim())) |> FeatureArray
end
function (𝒇::PairwiseFeatureSet)(x::DimensionalData.AbstractDimMatrix)
    DimArray(
        permutedims((cat(FeatureVector([𝑓(x) for 𝑓 ∈ 𝒇], 𝒇)...; dims=ndims(x) + 1)), [3, 1, 2]),
        (Dim{:feature}(getnames(𝒇)), dims(x, 2), dims(x, 2))) |> FeatureArray
end

# TODO Write tests for this

Pearson = SPI((x, y) -> cor(x, y), :Pearson, "Pearson correlation coefficient", ["correlation"])
Covariance = SPI((x, y) -> cov(x, y), :Pearson, "Sample covariance", ["covariance"])
