module Features
using DimensionalData
import Base: ==, show, hash
export AbstractFeature,
       Feature,
       getmethod,
       getname,
       getkeywords,
       getdescription,
       Identity

"""
`AbstractFeature` has a loose interface. Overload (𝑓::AbstractFeature)(x) for different
functionality. For compatibility with other `TimeseresFeatures` types, define:

## Helper methods:
- `getmethod(𝑓::AbstractFeature)`
- `getname(𝑓::AbstractFeature)
- `getkeywords(𝑓::AbstractFeature)
- `getdescription(𝑓::AbstractFeature)`
"""
abstract type AbstractFeature <: Function end

"""
    𝑓 = Feature([;] method::Function, name=Symbol(method), keywords="", description="")

Construct a `Feature`, which is a function annotated with a `name`, `keywords` and short `description`.
Features can be called as functions while `getname(𝑓)`, `getkeywords(𝑓)` and `getdescription(𝑓)` can be used to access the annotations.
The function should have at minimum a method for `AbstractVector`.
The method on vectors will be applied column-wise to `Matrix` inputs, regardless of the function methods defined for `Matrix`.

# Examples
```julia
𝑓 = Feature(sum, :sum, ["distribution"], "Sum of time-series values")
𝑓(1:10) # == sum(1:10) == 55
getdescription(𝑓) # "Sum of time-series values"
```
"""
Base.@kwdef struct Feature{F} <: AbstractFeature where {F <: Function}
    method::F
    name::Symbol = Symbol(method)
    description::String = ""
    keywords::Vector{String} = [""]
end
Feature(f::AbstractFeature) = f

# * AbstractFeature interface
getmethod(𝑓::Feature) = 𝑓.method
getname(𝑓::Feature) = 𝑓.name
getnames(𝑓::Feature) = [𝑓.name]
getkeywords(𝑓::Feature) = 𝑓.keywords
getdescription(𝑓::Feature) = 𝑓.description

# * Calculate features
(𝑓::AbstractFeature)(x::AbstractVector{<:Number}) = x |> getmethod(𝑓)
(𝑓::AbstractFeature)(X::AbstractArray{<:AbstractArray}) = map(𝑓, X)
function (𝑓::AbstractFeature)(X::AbstractArray{T}) where {T <: Number}
    return selectdim(mapslices(𝑓, X, dims = 1), 1, 1)
end

# * Comparing features
# For indexing: any features with the same name are the same feature
hash(𝑓::AbstractFeature, h::UInt) = hash(𝑓.name, h)
(Base.isequal)(𝑓::AbstractFeature, 𝑓′::AbstractFeature) = Base.isequal(hash(𝑓), hash(𝑓′))

# * Display
function commasep(x)
    if isempty(x)
        return String[]
    else
        y = fill(", ", 2 * length(x) - 1)
        y[1:2:end] .= x
        return y
    end
end
formatshort(𝑓::AbstractFeature) = [string(getname(𝑓)), " $(getdescription(𝑓))"]
function formatlong(𝑓::AbstractFeature)
    [string(typeof(𝑓)) * " ",
     string(getname(𝑓)),
     " with fields:\n",
     "$(repeat(' ', 3))description: ",
     getdescription(𝑓),
     "\n$(repeat(' ', 3))keywords: ",
     "$(commasep(getkeywords(𝑓))...)"]
end
show(𝑓::AbstractFeature) = print(formatlong(𝑓)...)
show(io::IO, 𝑓::AbstractFeature) = print(io, formatlong(𝑓)...)
function show(io::IO, m::MIME"text/plain", 𝑓::AbstractFeature)
    s = formatlong(𝑓)
    printstyled(io, s[1])
    println(io)
    println(io)
    printstyled(io, s[2], color = :light_blue, bold = true)
    printstyled(io, s[3])
    printstyled(io, s[4], color = :magenta)
    printstyled(io, s[5])
    printstyled(io, s[6], color = :yellow)
    printstyled(io, s[7])
end

const Identity = Feature(identity, :identity, "Identity function", ["transformation"])

end # module
