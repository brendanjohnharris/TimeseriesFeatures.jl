@reexport module FeatureArrays
import ..Features: AbstractFeature, getname, getmethod
import ..FeatureSets: getnames, AbstractFeatureSet
using ProgressLogging
using DimensionalData
import DimensionalData: dims, refdims, data, name, metadata, rebuild, parent,
                        AbstractDimArray, NoName
import DimensionalData.Dimensions: AnonDim, format, LookupArrays.NoMetadata
import Base: Array, getindex, setindex!

export AbstractFeatureArray, AbstractFeatureVector, AbstractFeatureMatrix,
       FeatureArray, FeatureVector, FeatureMatrix,
       getdim, setdim,
       FeatureDim, Fe # This should be set below in a future breaking release
# abstract type FeatureDim{T} <: DimensionalData.DependentDim{T} end
# DimensionalData.@dim Fe FeatureDim "Fe"
const FeatureDim = Dim{:feature, <:DimensionalData.LookupArrays.Categorical}
const Fe = FeatureDim

abstract type AbstractFeatureArray{T, N, D, A} <: AbstractDimArray{T, N, D, A} end

AbstractFeatureVector = AbstractFeatureArray{T, 1} where {T}
AbstractFeatureMatrix = AbstractFeatureArray{T, 2} where {T}

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
                    Tuple{<:FeatureDim, Vararg},
                    R <: Tuple, A <: AbstractArray{T, N}, Na,
                    Me} <:
       AbstractFeatureArray{T, N, D, A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me
end

function FeatureArray(data::A, dims::Tuple{D, Vararg},
                      refdims::R = (), name::Na = NoName(),
                      metadata::M = NoMetadata()) where {D <: Dim{:feature, Vector{Symbol}},
                                                         R, A, Na, M}
    FeatureArray(data, format(dims, data), refdims, name, metadata)
end
function FeatureArray(data::A, dims::Tuple,
                      refdims::R = (), name::Na = NoName(),
                      metadata::M = NoMetadata()) where {R, A, Na, M}
    DimArray(data, format(dims, data), refdims, name, metadata)
end

function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      args...)
    FeatureArray(data, (Dim{:feature}(features), fill(AnonDim(), ndims(data) - 1)...),
                 args...)
end
function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      timeseries::Union{Vector, Tuple}, args...)
    if data isa AbstractVector
        FeatureArray(reshape(data, :, 1),
                     (Dim{:feature}(features), Dim{:timeseries}(timeseries)), args...)
    else
        FeatureArray(data, (Dim{:feature}(features), Dim{:timeseries}(timeseries)), args...)
    end
end
function FeatureArray(data::AbstractArray, features::Union{Tuple{Symbol}, Vector{Symbol}},
                      otherdims::Union{Pair, Tuple{Pair}, Vector{Pair}}, args...)
    if otherdims isa Pair
        otherdims = [otherdims]
    end
    FeatureArray(data,
                 (Dim{:feature}(features),
                  [Dim{x.first}(x.second[:]) for x in otherdims]...), args...)
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
    FeatureArray(data, dims, refdims, name, metadata)
end

# * Index with Features and feature names
fidx(𝑓::AbstractFeature) = getname(𝑓)
fidx(𝑓::AbstractFeatureSet) = getnames(𝑓)
fidx(𝑓::Union{Symbol, Vector{Symbol}}) = At(𝑓)
FeatureUnion = Union{Symbol, Vector{Symbol}, AbstractFeature, AbstractFeatureSet}
getindex(A::AbstractFeatureVector, 𝑓::FeatureUnion) = getindex(A, fidx(𝑓))
setindex!(A::AbstractFeatureVector, x, 𝑓::FeatureUnion) = setindex!(A, x, fidx(𝑓))
function getindex(A::AbstractFeatureArray, 𝑓::FeatureUnion, i, I...)
    getindex(A, fidx(𝑓), i, I...) #A[fidx(𝑓)][:, i, I...]
end
function getindex(A::AbstractFeatureArray{T, 2, D}, 𝑓::Symbol, i,
                  I...) where {T, D <: Tuple{FeatureDim, AnonDim}}
    getindex(A, fidx(𝑓), i, I...).data
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

function FeatureArray(X::AbstractArray, 𝒇::AbstractFeatureSet)
    FeatureArray(X::AbstractArray, getnames(𝒇))
end

function (FeatureArray{T, N} where {T})(x::AbstractArray{S, N}, args...) where {S, N}
    FeatureArray(x, args...)
end

getdim(X::AbstractDimArray, dim) = dims(X, dim).val

"""
    getnames(𝒇::FeatureArray)
Get the names of features represented in the feature vector or array 𝒇 as a vector of symbols.
"""
featureDims(A::AbstractDimArray) = getdim(A, :feature)
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

(𝒇::AbstractFeatureSet)(x::AbstractVector{<:Number}) = FeatureVector([𝑓(x) for 𝑓 in 𝒇], 𝒇)

function (𝒇::AbstractFeatureSet)(X::AbstractVector{<:AbstractVector})
    F = Array{Float64}(undef, (length(𝒇), length(X)))
    @withprogress name="TimeseriesFeatures" begin
        threadlog = 0
        threadmax = prod(size(F, 2))
        l = Threads.ReentrantLock()
        Threads.@threads for i in eachindex(X)
            F[:, Tuple(i)...] .= 𝒇(X[i])
            lock(l)
            try
                threadlog += 1
                @logprogress threadlog / threadmax
            finally
                unlock(l)
            end
        end
    end
    FeatureArray(F, 𝒇)
end

function (𝒇::AbstractFeatureSet)(X::AbstractArray)
    F = Array{Float64}(undef, (length(𝒇), size(X)[2:end]...))
    threadlog = 0
    threadmax = prod(size(F)[2:end]) / Threads.nthreads()
    @withprogress name="TimeseriesFeatures" begin
        Threads.@threads for i in CartesianIndices(size(F)[2:end])
            F[:, Tuple(i)...] = vec(𝒇(X[:, Tuple(i)...]))
            Threads.threadid() == 1 && (threadlog += 1) % 50 == 0 &&
                @logprogress threadlog / threadmax
        end
    end
    FeatureArray(F, 𝒇)
end

# _construct(𝑓::AbstractFeature, X::AbstractDimArray{T,1}) where {T} = 𝑓(X.data)
(𝑓::AbstractFeature)(𝒳::AbstractDimStack) = map(𝑓, 𝒳)
function _construct(𝑓::AbstractFeature, X::DimensionalData.AbstractDimMatrix)
    DimArray(𝑓(X.data), (Dim{:feature}([getname(𝑓)]), dims(X)[2:end]...);
             refdims = refdims(X),
             name = DimensionalData.name(X), metadata = DimensionalData.metadata(X))
end
(𝑓::AbstractFeature)(X::DimensionalData.AbstractDimMatrix) = _construct(𝑓, X)
function _setconstruct(𝒇::AbstractFeatureSet, X::DimensionalData.AbstractDimArray)
    FeatureArray(𝒇(X.data), (Dim{:feature}(getnames(𝒇)), dims(X)[2:end]...), refdims(X),
                 DimensionalData.name(X), DimensionalData.metadata(X))
end
function _setconstruct(𝒇::AbstractFeatureSet, X::AbstractArray)
    FeatureArray(𝒇(X), (Dim{:feature}(getnames(𝒇)), dims(X)[2:end]...))
end
(𝒇::AbstractFeatureSet)(X::AbstractDimArray) = _setconstruct(𝒇, X)

end # module
