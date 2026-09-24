module GranularScenes

#################################################################################
# Dependencies
#################################################################################
using Gen
using Rooms
using Graphs
using Random
using Parameters
using FillArrays
using StaticArrays
using DocStringExtensions
using FunctionalCollections


# qualified
using Statistics: mean, std
using LinearAlgebra: norm
using Base.Iterators: product

include("utils/utils.jl")
# include("dgp/dgp.jl")
include("gm/gm.jl")
include("agent/agent.jl")
# include("inference/inference.jl")
# include("planning.jl")

end # module GranularScenes
