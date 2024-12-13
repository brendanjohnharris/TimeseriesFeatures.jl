module FeatureArrays
import ..Features: AbstractFeature, getname, getmethod
import ..FeatureSets: getnames, AbstractFeatureSet, FeatureSet
using ProgressLogging
using DimensionalData
import DimensionalData: dims, refdims, data, name, metadata, rebuild, parent,
                        AbstractDimArray, NoName, Categorical, Unordered
import DimensionalData.Dimensions: AnonDim, format, LookupArrays.NoMetadata
import Base: Array, getindex, setindex!

export AbstractFeatureArray, AbstractFeatureVector, AbstractFeatureMatrix,
       FeatureArray, FeatureVector, FeatureMatrix, getdim, setdim, FeatDim, Feat
abstract type FeatDim{T} <: DimensionalData.DependentDim{T} end
DimensionalData.@dim Feat FeatDim "Fe"

abstract type AbstractFeatureArray{T, N, D, A} <: AbstractDimArray{T, N, D, A} end

AbstractFeatureVector = AbstractFeatureArray{T, 1} where {T}
AbstractFeatureMatrix = AbstractFeatureArray{T, 2} where {T}

_featuredim(features) = Feat(Categorical(features; order = Unordered()))
_timeseriesdim(timeseries) = Dim{:timeseries}(timeseries)

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
                                                Vararg{<:DimensionalData.Dimension}})
    FeatureArray
end

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
    FeatureArray(data, (_featuredim(features), _timeseriesdim(timeseries));
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
function LabelledFeatureArray(args...; x::AbstractArray, kwargs...)
    FeatureArray(args...;
                 name = _name(x),
                 metadata = DimensionalData.metadata(x),
                 refdims = DimensionalData.refdims(x), kwargs...)
end

FeatureArray(D::DimArray) = FeatureArray(D.data, D.dims, D.refdims, D.name, D.metadata)
# DimensionalData.DimArray(D::FeatureArray) = DimArray(D.data, D.dims, D.refdims, D.name, D.metadata)

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

getdim(X::AbstractDimArray, dim) = dims(X, dim).val

"""
    getnames(𝒇::FeatureArray)
Get the names of features represented in the feature vector or array 𝒇 as a vector of symbols.
"""
featureDims(A::AbstractDimArray) = getdim(A, Feat)
getnames(A::AbstractFeatureArray) = featureDims(A)

timeseriesDims(A::AbstractDimArray) = getdim(A, :timeseries)

function setdim(F::DimArray, dim, vals...)::DimArray
    dimvec = Vector{Dimension}(undef, length(dims(F)))
    [(dimvec[i] = dims(F)[i]) for i in 1:length(dims(F)) if !∈(i, dim)]
    [(dimvec[dim[d]] = Dim{vals[d].first}(vals[d].second)) for d in 1:lastindex(dim)]
    DimArray(F, Tuple(dimvec)) # * Much faster to leave F as a DimArray rather than Array(F)
end
setdim(F::AbstractFeatureArray, args...) = FeatureArray(setdim(DimArray(F), args...))

function sortbydim(F::AbstractDimArray, dim; rev = false)
    sdim = FeatureArrays.getdim(F, dim)
    idxs = sortperm(sdim; rev)
    indx = [collect(1:size(F, i)) for i in 1:ndims(F)]
    indx[dim] = idxs
    return F[indx...]
end

_name(x) = DimensionalData.NoName()
_name(x::AbstractDimArray) = DimensionalData.name(x)
_name(x::AbstractDimStack) = DimensionalData.name(x)
function (𝒇::FeatureSet)(x::AbstractVector{<:T},
                         return_type::Type = Float64) where {T <: Number}
    y = [convert(return_type, 𝑓(x)) for 𝑓 in 𝒇]::Vector{return_type}
    y = LabelledFeatureArray(y, 𝒇; x)
end
function (𝒇::FeatureSet)(X::AbstractArray{<:AbstractVector}, return_type::Type = Float64)
    F = Array{return_type}(undef, (length(𝒇), size(X)...))
    @withprogress name="TimeseriesFeatures" begin
        threadlog = 0
        threadmax = prod(size(F, 2))
        l = Threads.ReentrantLock()
        Threads.@threads for i in eachindex(X)
            F[:, Tuple(i)...] .= 𝒇(X[i])
            if Threads.threadid() == 1
                threadlog += Threads.nthreads()
                @lock l (@logprogress threadlog / threadmax)
            end
        end
    end
    LabelledFeatureArray(F, 𝒇; x = X)
end
function (𝒇::FeatureSet)(X::AbstractArray{<:Number}, args...)
    dims = NTuple{ndims(X) - 1, Int}(2:ndims(X))
    𝒇(eachslice(X; dims, drop = true), args...)
end
# function (𝒇::FeatureSet)(X::AbstractArray)
#     F = Array{Float64}(undef, (length(𝒇), size(X)[2:end]...))
#     threadlog = 0
#     threadmax = prod(size(F)[2:end])
#     l = size(X, 1) > 1000 ? Threads.ReentrantLock() : nothing
#     @withprogress name="TimeseriesFeatures" begin
#         Threads.@threads for i in CartesianIndices(size(F)[2:end])
#             F[:, Tuple(i)...] = vec(𝒇(X[:, Tuple(i)...]))
#             if !isnothing(l)
#                 lock(l)
#                 try
#                     threadlog += 1
#                     @logprogress threadlog / threadmax
#                 finally
#                     unlock(l)
#                 end
#             end
#         end
#     end
#     LabelledFeatureArray(F, 𝒇; x = X)
# end

##  _construct(𝑓::AbstractFeature, X::AbstractDimArray{T,1}) where {T} = 𝑓(X.data)

(𝑓::AbstractFeature)(𝒳::AbstractDimStack) = map(𝑓, 𝒳)
# function _construct(𝑓::AbstractFeature, X::DimensionalData.AbstractDimMatrix)
#     DimArray(𝑓(X.data), (_featuredim([getname(𝑓)]), dims(X)[2:end]...);
#              refdims = refdims(X),
#              name = DimensionalData.name(X), metadata = DimensionalData.metadata(X))
# end
# function (𝑓::AbstractFeature)(X::DimensionalData.AbstractDimMatrix)
#     fullmethod(𝑓)(X)
# function _setconstruct(𝒇::AbstractFeatureSet, X::DimensionalData.AbstractDimArray)
#     FeatureArray(𝒇(X.data), (_featuredim(getnames(𝒇)), dims(X)[2:end]...);
#                  refdims = refdims(X),
#                  name = name(X), metadata = metadata(X))
# end
# function _setconstruct(𝒇::AbstractFeatureSet, X::AbstractArray)
#     FeatureArray(𝒇(X), (_featuredim(getnames(𝒇)), dims(X)[2:end]...);
#                  refdims = refdims(X),
#                  name = name(X), metadata = metadata(X))
# end
# (𝒇::FeatureSet)(X::AbstractDimArray) = _setconstruct(𝒇, X)

end # module
