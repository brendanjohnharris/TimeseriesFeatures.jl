using TimeseriesFeatures
using Documenter

DocMeta.setdocmeta!(TimeseriesFeatures, :DocTestSetup, :(using TimeseriesFeatures); recursive=true)

makedocs(;
    modules=[TimeseriesFeatures],
    authors="brendanjohnharris <brendanjohnharris@gmail.com> and contributors",
    repo="https://github.com/brendanjohnharris/TimeseriesFeatures.jl/blob/{commit}{path}#{line}",
    sitename="TimeseriesFeatures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://brendanjohnharris.github.io/TimeseriesFeatures.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/brendanjohnharris/TimeseriesFeatures.jl",
    devbranch="main",
)
