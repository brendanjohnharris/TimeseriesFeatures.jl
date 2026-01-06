module SuperFeatures
using MoreMaps

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
Base.@kwdef struct SuperFeature{F,G} <:
                   AbstractSuperFeature where {F<:AbstractFeature,G<:
    AbstractFeature}
    feature::F
    super::G
    name::Symbol
    description::String = getdescription(feature)
    keywords::Vector{String} = getkeywords(feature)
end
function SuperFeature(feature::AbstractFeature, super::AbstractFeature;
    merge=false,
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
# getmethod(𝑓::SuperFeature) = (getmethod ∘ getfeature)(𝑓) ∘ (getmethod ∘ getsuper)(𝑓)
getname(𝑓::SuperFeature) = 𝑓.name
getnames(𝑓::SuperFeature) = [𝑓.name]
getkeywords(𝑓::SuperFeature) = 𝑓.keywords
getdescription(𝑓::SuperFeature) = 𝑓.description

const SuperFeatureSet = FeatureSet{<:AbstractSuperFeature}

SuperFeatureSet(𝒇::AbstractVector{<:AbstractFeature}) = FeatureSet(map(SuperFeature, 𝒇))
SuperFeatureSet(𝒇::FeatureSet) = SuperFeatureSet(map(SuperFeature, collect(𝒇)))
SuperFeatureSet(𝒇::SuperFeatureSet) = 𝒇

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

function stamp(ℱ, idxs, fs)
    function f(x)
        supervals = ℱ(x, Any)
        return [𝑓(supervals[i]) for (i, 𝑓) in zip(idxs, fs)]
    end
end
function (𝒇::SuperFeatureSet)(x::AbstractVector{<:T},
    return_type::Type=Float64) where {T<:Number}
    F = LabelledFeatureArray(x, Vector{return_type}(undef, length(𝒇)), 𝒇)
    supers = getsuper.(𝒇)
    ℱ = supers |> unique |> FeatureSet
    idxs = indexin(supers, ℱ)
    fs = [(getmethod ∘ getfeature)(f) for f in collect(𝒇)]
    F .= stamp(ℱ, idxs, fs)(x)
    return F
end

function (𝒇::SuperFeatureSet)(X::AbstractArray{<:AbstractVector},
    return_type::Type=Float64;
    chart=Chart(Threaded(), ProgressLogger()))
    supers = getsuper.(𝒇)
    ℱ = supers |> unique |> FeatureSet
    idxs = indexin(supers, ℱ)
    fs = [(getmethod ∘ getfeature)(f) for f in collect(𝒇)]

    F = LabelledFeatureArray(X, Array{return_type}(undef, length(𝒇), size(X)...), 𝒇)
    Fc = map(stamp(ℱ, idxs, fs), chart, X)
    vec(parent(F)) .= Iterators.flatten(Fc)

    return F
end

# * Feature set arithmetic
function promote_rule(::Type{<:SuperFeatureSet}, ::Type{<:FeatureSet})
    SuperFeatureSet{SuperFeature}
end
function promote_rule(::Type{<:SuperFeature}, ::Type{<:AbstractFeature})
    SuperFeature
end
function promote_rule(::Type{AbstractSuperFeature}, ::Type{<:AbstractFeature})
    SuperFeature
end
function promote_rule(::Type{AbstractSuperFeature}, ::Type{<:Feature{<:H}}) where {H}
    SuperFeature
end
function promote_rule(::Type{<:SuperFeature}, ::Type{<:Feature{<:H}}) where {H}
    SuperFeature
end
function Base.promote_eltype(v1::AbstractFeatureSet, v2::AbstractFeatureSet)
    Base.promote_type(eltype(v1), eltype(v2))
end

# ! None of these are type stable
function Base.vcat(V1::A, V2::B) where {A<:AbstractFeatureSet,B<:AbstractFeatureSet}
    vcat(V1..., V2...) |> FeatureSet
end
(+)(𝒇::AbstractFeatureSet, 𝒇′::AbstractFeatureSet) = vcat(𝒇, 𝒇′)
(+)(𝒇::AbstractFeature, 𝒇′::AbstractFeature) = FeatureSet([𝒇, 𝒇′])
function intersect(𝒇::A, 𝒇′::B) where {A<:AbstractFeatureSet,B<:AbstractFeatureSet}
    FeatureSet(intersect(collect(𝒇), collect(𝒇′)))
end
function union(𝒇::A, 𝒇′::B) where {A<:AbstractFeatureSet,B<:AbstractFeatureSet}
    FeatureSet(union(collect(𝒇), collect(𝒇′)))
end
function setdiff(𝒇::A, 𝒇′::B) where {A<:AbstractFeatureSet,B<:AbstractFeatureSet}
    FeatureSet(setdiff(collect(𝒇), collect(𝒇′)))
end
(\)(𝒇::AbstractFeatureSet, 𝒇′::AbstractFeatureSet) = setdiff(𝒇, 𝒇′)

# Allow operations between FeatureSet and Feature by converting the Feature
for p in [:+, :\, :setdiff, :union, :intersect]
    eval(quote
        ($p)(𝒇::AbstractFeatureSet, f::AbstractFeature) = ($p)(𝒇, FeatureSet(f))
        ($p)(f::AbstractFeature, 𝒇::AbstractFeatureSet) = ($p)(FeatureSet(f), 𝒇)
    end)
end

# * Pretty print super feature set

const MAX_TREE_LINES = 100
const TREE_COLORS = [:red, :magenta, :cyan, :light_blue]  # root, mid, leaf, ...

function Base.show(io::IO, m::MIME"text/plain", 𝒇::SuperFeatureSet)
    if length(𝒇) == 0
        printstyled(io, "Empty SuperFeatureSet", color=:light_red, bold=true)
        return
    end

    # Build hierarchical tree structure
    # tree[root_super] = Dict(mid_super => [features...], ...)
    tree = _build_super_tree(𝒇)
    n_roots = length(tree)

    # Calculate display limits to stay under MAX_TREE_LINES
    max_per_leaf, max_mids = _calc_display_limits(tree, MAX_TREE_LINES)

    # Count intermediate supers (mids that are different from their root)
    n_mids = sum(sum(mid != root for (mid, _) in children)
                 for (root, children) in tree)
    has_mids = n_mids > 0

    # === Section 1: Tree summary ===
    printstyled(io, "SuperFeatureSet", color=:green, bold=true)
    print(io, " (")
    printstyled(io, "$n_roots", color=TREE_COLORS[1])
    if has_mids
        print(io, " → ")
        printstyled(io, "$n_mids", color=TREE_COLORS[min(2, length(TREE_COLORS))])
        print(io, " → ")
        printstyled(io, "$(length(𝒇))", color=TREE_COLORS[min(3, length(TREE_COLORS))])
    else
        print(io, " → ")
        printstyled(io, "$(length(𝒇))", color=TREE_COLORS[min(2, length(TREE_COLORS))])
    end
    println(io, " features)")

    for (ri, (root, children)) in enumerate(tree)
        is_last_root = (ri == n_roots)
        root_prefix = is_last_root ? "└─ " : "├─ "
        child_prefix = is_last_root ? "   " : "│  "

        # Count total features under this root
        total = sum(length(feats) for (_, feats) in children)

        print(io, root_prefix)
        printstyled(io, string(getname(root)), color=TREE_COLORS[1], bold=true)
        printstyled(io, " ($total)", color=:light_black)
        println(io)

        # Print children (intermediate supers or direct features)
        n_children = length(children)
        mids_to_show = min(max_mids, n_children)

        for (ci, (mid, features)) in enumerate(children)
            if ci > mids_to_show
                break
            end

            is_last_child = (ci == mids_to_show) && (n_children <= max_mids)
            mid_prefix = is_last_child ? "└─ " : "├─ "
            feat_prefix = is_last_child ? "   " : "│  "

            # Check if mid is same as root (no intermediate super)
            if mid == root
                # Direct features under root (level 2 = index 2)
                n_to_show = min(max_per_leaf, length(features))
                for (fi, f) in enumerate(features[1:n_to_show])
                    is_last_feat = (fi == n_to_show) && (length(features) <= max_per_leaf)
                    print(io, child_prefix, is_last_feat ? "└─ " : "├─ ")
                    printstyled(io, string(getname(f)),
                        color=TREE_COLORS[min(2, length(TREE_COLORS))])
                    println(io)
                end
                if length(features) > max_per_leaf
                    print(io, child_prefix, "└─ ")
                    printstyled(io, "... $(length(features) - max_per_leaf) more",
                        color=:light_black)
                    println(io)
                end
            else
                # Intermediate super with features under it (level 2)
                print(io, child_prefix, mid_prefix)
                printstyled(io, string(getname(mid)),
                    color=TREE_COLORS[min(2, length(TREE_COLORS))])
                printstyled(io, " ($(length(features)))", color=:light_black)
                println(io)

                # Print features under this intermediate super (level 3)
                n_to_show = min(max_per_leaf, length(features))
                for (fi, f) in enumerate(features[1:n_to_show])
                    is_last_feat = (fi == n_to_show) && (length(features) <= max_per_leaf)
                    print(io, child_prefix, feat_prefix, is_last_feat ? "└─ " : "├─ ")
                    printstyled(io, string(getname(f)),
                        color=TREE_COLORS[min(3, length(TREE_COLORS))])
                    println(io)
                end
                if length(features) > max_per_leaf
                    print(io, child_prefix, feat_prefix, "└─ ")
                    printstyled(io, "... $(length(features) - max_per_leaf) more",
                        color=:light_black)
                    println(io)
                end
            end
        end

        # Show "... more branches" if we truncated mid-level branches
        if n_children > max_mids
            hidden_mids = n_children - max_mids
            hidden_feats = sum(length(feats) for (_, feats) in children[(max_mids+1):end])
            print(io, child_prefix, "└─ ")
            printstyled(io, "... $hidden_mids more branches ($hidden_feats features)",
                color=:light_black)
            println(io)
        end
    end

    # === Section 2: Feature list (first 10) ===
    println(io)
    printstyled(io, "Features:", bold=true)
    println(io)
    n_show = min(10, length(𝒇))
    for i in 1:n_show
        printstyled(io, "  [$i] ", color=:light_black)
        printstyled(io, string(getname(𝒇[i])), color=:light_blue, bold=true)
        println(io)
    end
    if length(𝒇) > 10
        printstyled(io, "  ... $(length(𝒇) - 10) more features", color=:light_black)
    end
end

# Calculate max features per leaf and max mids per root to keep total lines under max_lines
# Returns (max_per_leaf, max_mids_per_root)
function _calc_display_limits(tree, max_lines)
    features_section_lines = 12  # header + 10 features + "more"
    header_lines = 1

    # Try different combinations of max_mids and max_per_leaf
    for max_mids in [typemax(Int), 10, 5, 3, 2, 1]
        for max_per_leaf in 5:-1:1
            total = _estimate_lines(tree, max_mids, max_per_leaf) + header_lines +
                    features_section_lines
            if total <= max_lines
                return (max_per_leaf, max_mids)
            end
        end
    end
    return (1, 1)
end

# Estimate total lines for the tree with given limits
function _estimate_lines(tree, max_mids, max_per_leaf)
    total = 0
    for (root, children) in tree
        total += 1  # root line

        n_mids = length(children)
        mids_to_show = min(max_mids, n_mids)

        for (ci, (mid, features)) in enumerate(children)
            if ci > mids_to_show
                break
            end

            if mid != root
                total += 1  # mid line
            end

            # Feature lines
            n_feats = length(features)
            total += min(max_per_leaf, n_feats)
            if n_feats > max_per_leaf
                total += 1  # "... more" line
            end
        end

        if n_mids > max_mids
            total += 1  # "... more branches" line
        end
    end
    return total
end

# Get the root superfeature (deepest in the chain)
function _get_root_super(super::AbstractFeature)
    if super isa SuperFeature
        inner = getsuper(super)
        if inner != Identity
            return _get_root_super(inner)
        end
    end
    return super
end

# Build a nested tree: root -> intermediate -> features
# Returns Vector of (root => Vector of (mid => [features]))
function _build_super_tree(𝒇::SuperFeatureSet)
    # Group by root super first
    root_groups = Dict{AbstractFeature,Vector{eltype(𝒇)}}()
    for f in 𝒇
        root = _get_root_super(getsuper(f))
        if !haskey(root_groups, root)
            root_groups[root] = eltype(𝒇)[]
        end
        push!(root_groups[root], f)
    end

    # For each root, group by immediate super
    result = Pair{AbstractFeature,Vector{Pair{AbstractFeature,Vector{eltype(𝒇)}}}}[]
    for (root, features) in root_groups
        mid_groups = Dict{AbstractFeature,Vector{eltype(𝒇)}}()
        for f in features
            mid = getsuper(f)
            # If mid's parent is root (or mid is root), use mid as the grouping key
            if !haskey(mid_groups, mid)
                mid_groups[mid] = eltype(𝒇)[]
            end
            push!(mid_groups[mid], f)
        end
        # Convert to ordered vector
        mid_vec = Pair{AbstractFeature,Vector{eltype(𝒇)}}[]
        for (mid, feats) in mid_groups
            push!(mid_vec, mid => feats)
        end
        push!(result, root => mid_vec)
    end
    return result
end

end # module
