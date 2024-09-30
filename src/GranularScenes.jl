module GranularScenes

#################################################################################
# Dependencies
#################################################################################

using Gen
using Rooms

using Graphs
using PyCall
using StaticArrays # REVIEW
using Parameters
using FunctionalCollections
using DocStringExtensions

#################################################################################
# Runtime configuration
#################################################################################

using PyCall
const numpy = PyNULL()
const torch = PyNULL()
const ti = PyNULL()
const vox = PyNULL()

function __init__()
    copy!(numpy, pyimport("numpy"))
    copy!(torch, pyimport("torch"))
    copy!(ti, pyimport("taichi"))

    # REVIEW: needed?
    # ti.init(arch = "gpu")

    copy!(vox, pyimport("pydeps"))

end

include("utils/utils.jl")
include("dgp/dgp.jl")
include("gm/gm.jl")
include("inference/inference.jl")
include("planning.jl")

end # module GranularScenes
