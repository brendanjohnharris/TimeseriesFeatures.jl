module SuperFeatures

import ..Features: AbstractFeature, Feature, getmethod, getname, getkeywords,
                   getdescription, Identity
import ..FeatureSets: AbstractFeatureSet, FeatureSet, getmethods, getnames, getdescriptions,
                      getkeywords
import ..FeatureArrays: FeatureVector, AbstractDimArray, FeatureArray, _featuredim,
                        LabelledFeatureArray
using ..DimensionalData
import Base: union, intersect, setdiff, convert, promote_rule, promote_eltype, cat, +, \
using ProgressLogging

export SuperFeature,
       SuperFeatureSet,
       Super, AbstractSuper,
       getsuper, getfeature

abstract type AbstractSuperFeature <: AbstractFeature end

## Univariate features
Base.@kwdef struct SuperFeature{F, G} <:
                   AbstractSuperFeature where {F <: AbstractFeature, G <:
                                                                     AbstractFeature}
    feature::F
    super::G
    name::Symbol
    description::String = getdescription(feature)
    keywords::Vector{String} = getkeywords(feature)
end
function SuperFeature(feature::AbstractFeature, super::AbstractFeature;
                      merge = false,
                      kwargs...)
    if merge
        name = Symbol(getname(feature), "_", getname(super))
        description = getdescription(feature) * " [of] " * getdescription(super)
        keywords = unique([getkeywords(feature)..., getkeywords(super)...])
    else
        name = getname(feature)
        description = getdescription(feature)
        keywords = getkeywords(feature)
    end
    SuperFeature(; feature, super, name, description, keywords, kwargs...)
end
function SuperFeature(method::Function, name::Symbol,
                      description::String, keywords::Vector{String},
                      super::AbstractFeature)
    feature = Feature(method, name, description, keywords)
    SuperFeature(feature, super, name, description, keywords)
end

Base.convert(::Type{SuperFeature}, x::Feature) = SuperFeature(x)
SuperFeature(f::Feature) = SuperFeature(f, Identity)
SuperFeature(f::SuperFeature) = f

# * Helper functions
# AbstractSuperFeature interface
getsuper(𝑓::SuperFeature) = 𝑓.super
getfeature(𝑓::SuperFeature) = 𝑓.feature

# AbstractFeature interface
getmethod(𝑓::SuperFeature) = (getmethod ∘ getfeature)(𝑓) ∘ getsuper(𝑓)
getname(𝑓::SuperFeature) = 𝑓.name
getnames(𝑓::SuperFeature) = [𝑓.name]
getkeywords(𝑓::SuperFeature) = 𝑓.keywords
getdescription(𝑓::SuperFeature) = 𝑓.description

const SuperFeatureSet = FeatureSet{<:AbstractSuperFeature}

SuperFeatureSet(𝒇::AbstractVector{<:AbstractSuperFeature}) = FeatureSet(𝒇)
SuperFeatureSet(𝒇::FeatureSet) = SuperFeatureSet(SuperFeature.(𝒇))
function SuperFeatureSet(features::AbstractVector{<:Function}, names::Vector{Symbol},
                         descriptions::Vector{String}, keywords, super)
    SuperFeature.(features, names, descriptions, keywords, super) |> FeatureSet
end
function SuperFeatureSet(features::Feature, args...)
    [SuperFeature(features, args...)] |> FeatureSet
end
function SuperFeatureSet(; features, names, keywords, descriptions, super)
    SuperFeatureSet(features, names, keywords, descriptions, super)
end
SuperFeatureSet(f::AbstractFeature) = SuperFeatureSet([f])

function (𝒇::SuperFeatureSet)(x::AbstractVector{<:T},
                              return_type::Type = Float64) where {T <: Number}
    F = LabelledFeatureArray(Vector{return_type}(undef, length(𝒇)), 𝒇; x)
    supers = getsuper.(𝒇)
    ℱ = supers |> unique |> FeatureSet
    supervals = [f(x) for f in ℱ]
    idxs = indexin(supers, ℱ)
    F .= [(getmethod ∘ getfeature)(f)(supervals[i]) for (i, f) in zip(idxs, 𝒇)]
    return F
end

function (𝒇::SuperFeatureSet)(X::AbstractArray{<:AbstractVector},
                              return_type::Type = Float64)
    supers = getsuper.(𝒇)
    ℱ = supers |> unique |> FeatureSet
    idxs = indexin(supers, ℱ)
    F = LabelledFeatureArray(Array{return_type}(undef, length(𝒇), size(X)...), 𝒇; x = X)
    @withprogress name="TimeseriesFeatures" begin
        threadlog = 0
        threadmax = length(X)
        l = Threads.ReentrantLock()
        Threads.@threads for i in CartesianIndices(X)
            supervals = [f(X[i]) for f in ℱ]
            F[:, i] .= [(getmethod ∘ getfeature)(f)(supervals[i])
                        for (i, f) in zip(idxs, 𝒇)]
            if Threads.threadid() == 1
                threadlog += Threads.nthreads()
                @lock l (@logprogress threadlog / threadmax)
            end
        end
    end
    return F
end

# * Feature set arithmetic
function promote_rule(::Type{<:SuperFeatureSet}, ::Type{<:FeatureSet})
    SuperFeatureSet{SuperFeature}
end
function promote_rule(::Type{SuperFeature{F, G}}, ::Type{<:AbstractFeature}) where {F, G}
    SuperFeature
end
function promote_rule(::Type{AbstractSuperFeature}, ::Type{<:AbstractFeature})
    SuperFeature
end
function promote_rule(::Type{AbstractSuperFeature}, ::Type{<:Feature{<:H}}) where {H}
    SuperFeature
end
function promote_rule(::Type{SuperFeature}, ::Type{<:Feature{<:H}}) where {H}
    SuperFeature
end
function Base.promote_eltype(v1::AbstractFeatureSet, v2::AbstractFeatureSet)
    Base.promote_type(eltype(v1), eltype(v2))
end

# ! None of these are type stable
function Base.vcat(V1::A, V2::B) where {A <: AbstractFeatureSet, B <: AbstractFeatureSet}
    T = Base.promote_eltype(V1, V2)
    FeatureSet(Base.typed_vcat(T, T.(V1), T.(V2)))
end
(+)(𝒇::AbstractFeatureSet, 𝒇′::AbstractFeatureSet) = vcat(𝒇, 𝒇′)
(+)(𝒇::AbstractFeature, 𝒇′::AbstractFeature) = FeatureSet([𝒇, 𝒇′])
function intersect(𝒇::A, 𝒇′::B) where {A <: AbstractFeatureSet, B <: AbstractFeatureSet}
    T = promote_eltype(𝒇, 𝒇′)
    FeatureSet(intersect(T.(𝒇), T.(𝒇′)))
end
function union(𝒇::A, 𝒇′::B) where {A <: AbstractFeatureSet, B <: AbstractFeatureSet}
    T = promote_eltype(𝒇, 𝒇′)
    FeatureSet(union(T.(𝒇), T.(𝒇′)))
end
function setdiff(𝒇::A, 𝒇′::B) where {A <: AbstractFeatureSet, B <: AbstractFeatureSet}
    T = promote_eltype(𝒇, 𝒇′)
    FeatureSet(setdiff(T.(𝒇), T.(𝒇′)))
end
(\)(𝒇::AbstractFeatureSet, 𝒇′::AbstractFeatureSet) = setdiff(𝒇, 𝒇′)

# Allow operations between FeatureSet and Feature by converting the Feature
for p in [:+, :\, :setdiff, :union, :intersect]
    eval(quote
             ($p)(𝒇::AbstractFeatureSet, f::AbstractFeature) = ($p)(𝒇, FeatureSet(f))
             ($p)(f::AbstractFeature, 𝒇::AbstractFeatureSet) = ($p)(FeatureSet(f), 𝒇)
         end)
end

end # module
