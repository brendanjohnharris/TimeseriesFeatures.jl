@reexport module SuperFeatures

import ..getmethod
import ..Features: AbstractFeature, Feature, getmethod, getname, getkeywords, getdescription
import ..FeatureSets: AbstractFeatureSet, FeatureSet, getmethods, getnames, getdescriptions,
                      getkeywords
import ..FeatureArrays: FeatureVector, AbstractDimArray, _construct, _setconstruct,
                        FeatureArray, _featuredim
using ..DimensionalData
using ProgressLogging

export SuperFeature,
       SuperFeatureSet,
       Super, AbstractSuper,
       getsuper, getfeature

abstract type AbstractSuperFeature <: AbstractFeature end

## Univariate features
Base.@kwdef struct SuperFeature <: AbstractSuperFeature
    method::Function
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
    super::AbstractFeature
end
function SuperFeature(method::Function, name = Symbol(method),
                      keywords::Vector{String} = [""], description::String = "";
                      super::AbstractFeature)
    SuperFeature(; super, method, name, keywords, description)
end
function SuperFeature(method::Function, name, description::String,
                      keywords::Vector{String} = [""]; super::AbstractFeature)
    SuperFeature(; super, method, name, keywords, description)
end
getsuper(𝒇::AbstractSuperFeature) = 𝒇.super
getsuper(::AbstractFeature) = ()
getfeature(𝑓::SuperFeature) = Feature(getmethod(𝑓))

(𝑓::SuperFeature)(x::AbstractVector) = x |> getsuper(𝑓) |> getmethod(𝑓)
(𝑓::SuperFeature)(x::DimensionalData.AbstractDimVector) = x |> getsuper(𝑓) |> getmethod(𝑓)

function (𝑓::SuperFeature)(X::DimensionalData.AbstractDimArray)
    FeatureArray(getmethod(𝑓).(getsuper(𝑓)(X)),
                 (_featuredim([getname(𝑓)]), dims(X)[2:end]...); refdims = refdims(X),
                 name = name(X), metadata = metadata(X))
end
function (𝑓::SuperFeature)(X::DimensionalData.AbstractDimMatrix)
    FeatureArray(getmethod(𝑓).(getsuper(𝑓)(X)).data,
                 (_featuredim([getname(𝑓)]), dims(X)[2:end]...); refdims = refdims(X),
                 name = name(X), metadata = metadata(X))
end

struct SuperFeatureSet <: AbstractFeatureSet
    features::AbstractVector
    SuperFeatureSet(features::Vector{T}) where {T <: AbstractFeature} = new(features)
end

# SuperPairwiseFeatureSet = SuperFeatureSet

function SuperFeatureSet(methods::AbstractVector{<:Function}, names::Vector{Symbol},
                         descriptions::Vector{String}, keywords, super)
    SuperFeature.(methods, names, descriptions, keywords, super) |> SuperFeatureSet
end
function SuperFeatureSet(methods::Function, args...)
    [SuperFeature(methods, args...)] |> SuperFeatureSet
end
function SuperFeatureSet(; methods, names, keywords, descriptions, super)
    SuperFeatureSet(methods, names, keywords, descriptions, super)
end
SuperFeatureSet(f::AbstractFeature) = SuperFeatureSet([f])

# SuperFeatureSet(𝒇::Vector{Feature}) = SuperFeatureSet(getmethods(𝒇), getnames(𝒇), getdescriptions(𝒇), getkeywords(𝒇), getsuper(first(𝒇)))
getindex(𝒇::AbstractFeatureSet, I) = SuperFeatureSet(getfeatures(𝒇)[I])
SuperFeatureSet(𝒇::Vector{Feature}) = FeatureSet(𝒇) # Just a regular feature set

function superloop(f::AbstractSuperFeature, supervals, x)
    getfeature(f)(supervals[getname(getsuper(f))])
end
function superloop(f::AbstractFeature, supervals, x)
    f(x) # Just a regular feature of the original time series
end

function (𝒇::SuperFeatureSet)(x::AbstractVector{<:Number}; kwargs...)::FeatureVector
    ℱ = getsuper.(𝒇) |> unique |> SuperFeatureSet
    supervals = Dict(getname(f) => f(x) for f in ℱ)
    FeatureArray(vcat([superloop(𝑓, supervals, x) for 𝑓 in 𝒇]...), 𝒇; kwargs...)
end
function (𝒇::SuperFeatureSet)(X::AbstractArray; kwargs...)
    ℱ = getsuper.(𝒇) |> unique |> SuperFeatureSet
    supervals = Array{Any}(undef, (length(ℱ), size(X)[2:end]...)) # Can we be more specific with the types?
    threadlog = 0
    threadmax = 2.0 .* prod(size(X)[2:end]) / Threads.nthreads()
    @withprogress name="TimeseriesFeatures" begin
        idxs = CartesianIndices(size(X)[2:end])
        Threads.@threads for i in idxs
            supervals[:, i] = vec([f(X[:, i]) for f in ℱ])
            Threads.threadid() == 1 && (threadlog += 1) % 50 == 0 &&
                @logprogress threadlog / threadmax
        end
        supervals = FeatureArray(supervals, ℱ)
        f1 = superloop.(𝒇, [supervals[:, first(idxs)]], [X[:, first(idxs)]]) # Assume same output type for all time series
        F = similar(f1, (length(𝒇), size(X)[2:end]...))
        F[:, first(idxs)] .= f1
        Threads.@threads for i in idxs[2:end]
            F[:, i] .= superloop.(𝒇, [supervals[:, i]], [X[:, i]])
            Threads.threadid() == 1 && (threadlog += 1) % 50 == 0 &&
                @logprogress threadlog / threadmax
        end
        return FeatureArray(F, 𝒇; kwargs...)
    end
end
function (𝒇::SuperFeatureSet)(x::AbstractDimArray; kwargs...)
    F = 𝒇(parent(x))
    FeatureArray(parent(F),
                 (_featuredim(getnames(𝒇)), dims(x)[2:end]...); refdims = refdims(x),
                 name = name(x), metadata = metadata(x), kwargs...)
end

# (𝒇::SuperFeatureSet)(X::AbstractDimArray) = _setconstruct(𝒇, X)

## Pairwise features
abstract type AbstractSuper{F, S} <: AbstractSuperFeature where {F, S} end
struct Super{F, S} <: AbstractSuper{F, S}
    feature::F
    super::S
    name::Symbol
end
Super(feature, super) = Super(feature, super, Symbol(feature.name, "_", super.name))
getmethod(𝑓::AbstractSuper) = 𝑓.feature.method
getname(𝑓::AbstractSuper) = 𝑓.name
getnames(𝑓::AbstractSuper) = [𝑓.name]
getkeywords(𝑓::AbstractSuper) = unique([𝑓.feature.keywords..., 𝑓.super.keywords...])
getdescription(𝑓::AbstractSuper) = 𝑓.feature.description * " [of] " * 𝑓.super.description
getsuper(𝑓::AbstractSuper) = 𝑓.super
getfeature(𝑓::AbstractSuper) = 𝑓.feature

function (𝑓::AbstractSuper{F, S})(x::AbstractVector) where {F <: AbstractFeature,
                                                            S <: AbstractFeature}
    getfeature(𝑓)(getsuper(𝑓)(x))
end
function (𝑓::AbstractSuper{F, S})(x::AbstractArray) where {F <: AbstractFeature,
                                                           S <: AbstractFeature}
    getfeature(𝑓)(getsuper(𝑓)(x))
end
function (𝑓::AbstractSuper{F, S})(x::AbstractDimArray) where {F <: AbstractFeature,
                                                              S <: AbstractFeature}
    getfeature(𝑓)(getsuper(𝑓)(x))
end
function (𝑓::AbstractSuper{F, S})(x::DimensionalData.AbstractDimMatrix) where {
                                                                               F <:
                                                                               AbstractFeature,
                                                                               S <:
                                                                               AbstractFeature
                                                                               }
    getfeature(𝑓)(getsuper(𝑓)(x))
end
# function (𝑓::AbstractSuper{F,S})(x::AbstractArray{<:AbstractArray}) where {F<:AbstractFeature,S<:AbstractFeature}
#     map(getfeature(𝑓) ∘ getsuper(𝑓), x)
# end

end # module
