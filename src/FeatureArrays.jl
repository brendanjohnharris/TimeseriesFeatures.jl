module FeatureArrays
import ..Features: AbstractFeature, getname, getmethod
import ..FeatureSets: getnames, AbstractFeatureSet, FeatureSet
using ProgressLogging
using MoreMaps
using DimensionalData
import DimensionalData: dims, refdims, data, name, metadata, rebuild, parent,
                        AbstractDimArray, NoName, Categorical, Unordered
import DimensionalData.Dimensions: AnonDim, format, NoMetadata
import Base: Array, getindex, setindex!

export AbstractFeatureArray, AbstractFeatureVector, AbstractFeatureMatrix,
       FeatureArray, FeatureVector, FeatureMatrix, FeatDim, Feat
abstract type FeatDim{T} <: DimensionalData.DependentDim{T} end
DimensionalData.@dim Feat FeatDim "Feature"

abstract type AbstractFeatureArray{T, N, D, A} <: AbstractDimArray{T, N, D, A} end

AbstractFeatureVector = AbstractFeatureArray{T, 1} where {T}
AbstractFeatureMatrix = AbstractFeatureArray{T, 2} where {T}

_featuredim(features) = Feat(Categorical(features; order = Unordered()))

"""
    F = FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol},Vector{Symbol}}, [timeseries::Union{Vector, Tuple}], args...)

Construct a `FeatureArray`, which annotates the array `data` with names of `features` along rows and, optionally, `timeseries` along columns.
Since `FeatureArray <: AbstractFeatureArray <: AbstractDimArray`, further arguments to the `FeatureArray` constructor are passed to the `DimArray` constructor.
To access feature names, use `getnames(F)`.

# Examples
```julia
data = rand(Int, 2, 10) # Some feature matrix with 2 features and 10 timeseries
F = FeatureArray(data, [:sum, :length])
```
"""
struct FeatureArray{T, N,
                    D <:
                    Tuple{<:FeatDim, Vararg},
                    R <: Tuple, A <: AbstractArray{T, N}, Na,
                    Me} <: AbstractFeatureArray{T, N, D, A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me
end

function DimensionalData.dimconstructor(::Tuple{<:FeatDim,
                                                Vararg{DimensionalData.Dimensions.Dimension}})
    FeatureArray
end
DimensionalData.dimconstructor(::Tuple{<:FeatDim, Vararg}) = FeatureArray
DimensionalData.dimconstructor(::FeatDim) = FeatureArray

# * Need to overload stack for cases where the inner arrays are DimArrays
Base.stack(A::FeatureArray) = stack(DimArray(A); dims = 1) |> FeatureArray

function FeatureArray(data::A, dims::Tuple{D, Vararg};
                      refdims::R = (), name::Na = NoName(),
                      metadata::M = NoMetadata()) where {D <: FeatDim,
                                                         R, A, Na, M}
    FeatureArray(data, format(dims, data), refdims, name, metadata)
end
function FeatureArray(data, dims::Tuple;
                      refdims = (), name = NoName(),
                      metadata = NoMetadata())
    DimArray(data, format(dims, data), refdims, name, metadata)
end

function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}};
                      kwargs...)
    FeatureArray(data,
                 (_featuredim(features),
                  fill(AnonDim(), ndims(data) - 1)...);
                 kwargs...)
end
function FeatureArray(data::AbstractVector, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      timeseries::Union{Vector, Tuple}; kwargs...)
    FeatureArray(reshape(data, :, 1),
                 (_featuredim(features), Dim{:timeseries}(timeseries)); kwargs...)
end
function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      timeseries::Union{Vector, Tuple}; kwargs...)
    FeatureArray(data, (_featuredim(features), Dim{:timeseries}(timeseries));
                 kwargs...)
end
function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      otherdims::Union{Pair, Tuple{Pair}, Vector{Pair}}; kwargs...)
    if otherdims isa Pair
        otherdims = [otherdims]
    end
    FeatureArray(data,
                 (_featuredim(features),
                  [Dim{x.first}(x.second[:]) for x in otherdims]...); kwargs...)
end

FeatureArray(D::DimArray) = FeatureArray(D.data, D.dims, D.refdims, D.name, D.metadata)

dims(A::AbstractFeatureArray) = A.dims
refdims(A::AbstractFeatureArray) = A.refdims
data(A::AbstractFeatureArray) = A.data
name(A::AbstractFeatureArray) = A.name
metadata(A::AbstractFeatureArray) = A.metadata
parent(A::AbstractFeatureArray) = data(A)
Base.Array(A::AbstractFeatureArray) = Array(parent(A))

@inline function rebuild(A::AbstractFeatureArray, data::AbstractArray, dims::Tuple,
                         refdims::Tuple, name, metadata)
    if dims isa Tuple
        dims = replace(collect(dims), nothing => AnonDim())
    end
    FeatureArray(data, Tuple(dims); refdims, name, metadata)
end

# * Index with Features and feature names
fidx(𝑓::AbstractFeature) = getname(𝑓)
fidx(𝑓::AbstractFeatureSet) = getnames(𝑓)
fidx(𝑓::Union{Symbol, Vector{Symbol}}) = At(𝑓)
FeatureSetUnion = Union{Vector{Symbol}, AbstractFeatureSet}
FeatureUnion = Union{Symbol, AbstractFeature, FeatureSetUnion}
getindex(A::AbstractFeatureVector, 𝑓::FeatureUnion) = getindex(A, fidx(𝑓))
setindex!(A::AbstractFeatureVector, x, 𝑓::FeatureUnion) = setindex!(A, x, fidx(𝑓))

# * In these cases, just indexing features, we can actually be type stable
function getindex(A::T, 𝑓::FeatureSetUnion)::T where {T <: AbstractFeatureVector}
    getindex(A, fidx(𝑓))
end

function getindex(A::AbstractFeatureArray, 𝑓::FeatureUnion, i, I...)
    getindex(A, fidx(𝑓), i, I...)
end
function getindex(A::AbstractFeatureArray{T, 2, D}, 𝑓::Symbol, i,
                  I...) where {T, D <: Tuple{FeatDim, AnonDim}}
    parent(getindex(A, fidx(𝑓), i, I...))
end
function setindex!(A::AbstractFeatureArray, x, 𝑓::FeatureUnion, i, I...)
    setindex!(A, x, fidx(𝑓), i, I...)
end

# * And with features alone, no other dims. Here we assume features are along the first dim.
function getindex(A::AbstractFeatureArray, 𝑓::FeatureUnion)
    getindex(A, 𝑓, fill(:, ndims(A) - 1)...)
end
function setindex!(A::AbstractFeatureArray, x, 𝑓::FeatureUnion)
    setindex!(A, x, 𝑓, fill(:, ndims(A) - 1)...)
end

"""
    FeatureArray{T, 2} where {T}

An alias to construct a `FeatureArray` for a flat set of timeseries.

# Examples
```julia
data = rand(Int, 2, 3) # Some feature matrix with 2 features and 3 timeseries
F = FeatureMatrix(data, [:sum, :length], [1, 2, 3])
```
"""
FeatureMatrix = FeatureArray{T, 2} where {T}

"""
    FeatureArray{T, 1} where {T}

An alias to construct a `FeatureArray` for a single time series.

# Examples
```julia
data = randn(2) # Feature values for 1 time series
𝐟 = FeatureVector(data, [:sum, :length])
```
"""
FeatureVector = FeatureArray{T, 1} where {T}

function FeatureArray(X::AbstractArray, 𝒇::AbstractFeatureSet, args...; kwargs...)
    FeatureArray(X::AbstractArray, getnames(𝒇), args...; kwargs...)
end

function (FeatureArray{T, N} where {T})(x::AbstractArray{S, N}, args...;
                                        kwargs...) where {S, N}
    FeatureArray(x, args...; kwargs...)
end

"""
    getnames(𝒇::FeatureArray)
Get the names of features represented in the feature vector or array 𝒇 as a vector of symbols.
"""
featureDims(A::AbstractDimArray) = lookup(A, Feat)
getnames(A::AbstractFeatureArray) = featureDims(A)

_name(x) = DimensionalData.NoName()
_name(x::AbstractDimArray) = DimensionalData.name(x)
_name(x::AbstractDimStack) = DimensionalData.name(x)
function LabelledFeatureArray(x::AbstractArray, args...; kwargs...)
    FeatureArray(args...;
                 name = _name(x),
                 metadata = DimensionalData.metadata(x),
                 refdims = DimensionalData.refdims(x), kwargs...)
end
function LabelledFeatureArray(x::AbstractArray, F::AbstractDimArray, f; kwargs...)
    FeatureArray(parent(F), f; # Don't have a DimArray as data
                 name = _name(x),
                 metadata = DimensionalData.metadata(x),
                 refdims = DimensionalData.refdims(x), kwargs...)
end
function LabelledFeatureArray(x::AbstractDimArray, F::AbstractArray, f; kwargs...)
    newdims = (_featuredim(getnames(f)), DimensionalData.dims(x)[2:end]...)
    FeatureArray(parent(F), newdims;
                 name = _name(x),
                 metadata = DimensionalData.metadata(x),
                 refdims = DimensionalData.refdims(x), kwargs...)
end
function LabelledFeatureArray(x::AbstractDimArray{<:AbstractArray}, F::AbstractArray, f;
                              kwargs...)
    newdims = (_featuredim(getnames(f)), DimensionalData.dims(x)[1:end]...)
    FeatureArray(parent(F), newdims;
                 name = _name(x),
                 metadata = DimensionalData.metadata(x),
                 refdims = DimensionalData.refdims(x), kwargs...)
end

function (𝒇::FeatureSet)(x::AbstractVector{<:T},
                         return_type::Type = Float64) where {T <: Number}
    y = [𝑓(x) for 𝑓 in 𝒇]
    y = convert(Vector{return_type}, y)
    y = LabelledFeatureArray(x, y, 𝒇)
end
function (𝒇::FeatureSet)(X::AbstractArray{<:AbstractVector}, return_type::Type = Float64;
                         chart = DEFAULT_CHART())
    F = LabelledFeatureArray(X, Array{return_type}(undef, length(𝒇), size(X)...), 𝒇)
    Fc = map(𝒇, chart, X)
    vec(parent(F)) .= Iterators.flatten(Fc)
    return F
end
function (𝒇::FeatureSet)(X::AbstractArray{T, N}, return_type::Type = Float64;
                         kwargs...) where {T <: Number, N}
    dims = ntuple(i -> i + 1, N - 1)
    F = LabelledFeatureArray(X, Array{return_type}(undef, length(𝒇), size(X)[2:end]...), 𝒇)
    F .= 𝒇(eachslice(X; dims); kwargs...)
    return F
end

(𝑓::AbstractFeature)(𝒳::AbstractDimStack) = map(𝑓, 𝒳)

end # module
