module PairwiseFeatures
using Statistics
using ProgressLogging
export PairwiseFeature, PairwiseFeatureSet, AbstractPairwiseFeature, SuperPairwiseFeature,
    SuperPairwiseFeatureSet, PairwiseSuperFeatureSet
import ..Features: AbstractFeature, Feature, getmethod, getname, getkeywords,
    getdescription, Identity
import ..FeatureSets: AbstractFeatureSet, FeatureSet, getnames, getname
import ..FeatureArrays: FeatureArray, FeatureVector, _featuredim, LabelledFeatureArray
import ..SuperFeatures: AbstractSuperFeature, SuperFeature, getsuper, getmethod, getfeature,
    SuperFeatureSet
using ..DimensionalData
export Pearson, Covariance

abstract type AbstractPairwiseFeature <: AbstractFeature end

Base.@kwdef struct PairwiseFeature{F} <: AbstractPairwiseFeature where {F <: Function}
    method::F # For an SPI, this should be (x, y) -> f(x, y)
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
const SuperPairwiseFeature = SuperFeature{<:AbstractPairwiseFeature}
const PairwiseUnion = Union{PairwiseFeature, SuperPairwiseFeature}
SuperFeature(f::PairwiseFeature) = SuperFeature(f, Identity)

# * AbstractFeature interface
getmethod(𝑓::PairwiseFeature) = 𝑓.method
getname(𝑓::PairwiseFeature) = 𝑓.name
getnames(𝑓::PairwiseFeature) = [𝑓.name]
getkeywords(𝑓::PairwiseFeature) = 𝑓.keywords
getdescription(𝑓::PairwiseFeature) = 𝑓.description

# * PairwiseFeature calculations
(𝑓::PairwiseFeature)(x::AbstractVector{<:Number}) = getmethod(𝑓)(x, x)
function (𝑓::PairwiseFeature)(x::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    return getmethod(𝑓)(x, y)
end
function (𝑓::PairwiseFeature)(X::AbstractArray{<:AbstractArray})
    return map(𝑓, Iterators.product(X, X))
end
function (𝑓::AbstractPairwiseFeature)(X::AbstractArray{T}) where {T <: Number}
    dims = NTuple{ndims(X) - 1, Int}(2:ndims(X))
    return 𝑓(eachslice(X; dims, drop = true))
end

# * SuperPairwiseFeature calculations
function (𝑓::SuperPairwiseFeature)(x::AbstractVector{<:Number})
    y = getsuper(𝑓)(x)
    return (getmethod ∘ getfeature)(𝑓)(y, y)
end
function (𝑓::SuperPairwiseFeature)(x::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    _x = getsuper(𝑓)(x)
    _y = getsuper(𝑓)(y)
    return (getmethod ∘ getfeature)(𝑓)(_x, _y)
end
function (𝑓::SuperPairwiseFeature)(X::AbstractArray{<:AbstractArray})
    Y = getsuper(𝑓)(X)
    return map(getfeature(𝑓), Iterators.product(Y, Y))
end
function (𝑓::PairwiseUnion)(xy::NTuple{2, AbstractVector{<:Number}})
    return 𝑓(first(xy), last(xy))
end
function (𝑓::PairwiseUnion)(xs::AbstractMatrix{<:Number})
    return 𝑓(eachcol(xs))
end

# * PairwiseFeatureSet calculations
const PairwiseFeatureSet = FeatureSet{<:AbstractPairwiseFeature}
function (𝒇::PairwiseFeatureSet)(
        x::AbstractVector{<:T},
        y::AbstractVector{<:T},
        return_type::Type = Float64
    ) where {T <: Number}
    y = [𝑓(x, y) for 𝑓 in 𝒇]
    y = convert(Vector{return_type}, y)
    return FeatureArray(y, 𝒇)
end
function (𝒇::PairwiseFeatureSet)(
        xy::NTuple{2, AbstractVector{<:Number}},
        return_type::Type = Float64
    )
    return 𝒇(first(xy), last(xy), return_type)
end

function (𝒇::PairwiseFeatureSet)(
        X::AbstractArray{<:AbstractVector},
        return_type::Type = Array{Float64}
    ) # ! We should parallelize this at some point
    F = convert(Vector{return_type}, [𝑓(X) for 𝑓 in 𝒇])
    return LabelledFeatureArray(X, F, 𝒇)
end
function (𝒇::PairwiseFeatureSet)(
        X::AbstractArray{<:AbstractDimVector},
        return_type::Type = DimArray{Float64}
    ) # ! We should parallelize this at some point
    F = convert(Vector{return_type}, [𝑓(X) for 𝑓 in 𝒇])
    return LabelledFeatureArray(X, F, 𝒇)
end
function (𝒇::PairwiseFeatureSet)(X::AbstractArray{<:Number}, args...; kwargs...)
    dims = NTuple{ndims(X) - 1, Int}(2:ndims(X))
    F = 𝒇(eachslice(X; dims, drop = true), args...; kwargs...)
    return LabelledFeatureArray(X, F, 𝒇)
end

# * SuperPairwiseFeatureSet calculations
const SuperPairwiseFeatureSet = FeatureSet{SuperPairwiseFeature}
const _SuperPairwiseFeatureSet = Vector{SuperFeature{<:AbstractPairwiseFeature, T} where {T}}

function LabelledFeatureArray(
        x::AbstractDimArray, F::AbstractDimArray,
        f::Union{PairwiseFeatureSet, SuperPairwiseFeatureSet};
        kwargs...
    )
    return F
end

function PairwiseSuperFeatureSet(f::Vector{<:AbstractSuperFeature})
    f = _SuperPairwiseFeatureSet(f)
    return FeatureSet(f)::SuperPairwiseFeatureSet
end
function (𝒇::SuperPairwiseFeatureSet)(
        x::AbstractVector{<:T},
        y::AbstractVector{<:T},
        return_type::Type = Float64
    ) where {T <: Number}
    supers = getsuper.(𝒇)
    ℱ = supers |> unique |> FeatureSet

    superxs = [f(x) for f in ℱ]
    superys = [f(y) for f in ℱ]
    idxs = indexin(supers, ℱ)

    y = [(getmethod ∘ getfeature)(𝑓)(superxs[i], superys[i]) for (i, 𝑓) in zip(idxs, 𝒇)]
    y = convert(Vector{return_type}, y)
    return y = FeatureArray(y, 𝒇)
end
function (𝒇::SuperPairwiseFeatureSet)(
        x::AbstractVector{<:T},
        return_type::Type = Float64
    ) where {T <: Number}
    return 𝒇(x, x, return_type)
end
function (𝒇::SuperPairwiseFeatureSet)(
        X::AbstractArray{<:AbstractVector},
        return_type::Type = Array{Float64}
    )
    F = convert(Vector{return_type}, [𝑓(X) for 𝑓 in 𝒇])
    return LabelledFeatureArray(X, F, 𝒇)
end
function (𝒇::SuperPairwiseFeatureSet)(
        X::AbstractArray{<:AbstractDimVector},
        return_type::Type = DimArray{Float64}
    )
    F = convert(Vector{return_type}, [𝑓(X) for 𝑓 in 𝒇])
    return LabelledFeatureArray(X, F, 𝒇)
end
function (𝒇::SuperPairwiseFeatureSet)(
        X::AbstractArray{T, N}, args...;
        kwargs...
    ) where {T <: Number, N}
    dims = NTuple{ndims(X) - 1, Int}(2:ndims(X))
    F = 𝒇(eachslice(X; dims, drop = true), args...; kwargs...)
    return LabelledFeatureArray(X, F, 𝒇)
end

Pearson = PairwiseFeature(
    (x, y) -> cor(collect(x), collect(y)), :Pearson,
    "Pearson correlation coefficient",
    ["correlation"]
)
Covariance = PairwiseFeature(
    (x, y) -> cov(collect(x), collect(y)), :Covariance,
    "Sample covariance",
    ["covariance"]
)

end # module
