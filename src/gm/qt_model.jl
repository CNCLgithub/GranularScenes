export QuadTreeModel


include("graphics/graphics.jl")

#################################################################################
# Model specification
#################################################################################

"""
Parameters for an instance of the `QuadTreeModel`.
"""
@with_kw struct QuadTreeModel

    #############################################################################
    # Room geometry
    #############################################################################
    #
    dims::Tuple{Int64, Int64}
    # coarsest node is centered at [0,0]
    # and has a span of [1,1]
    center::SVector{2, Float64} = SVector{2, Float64}(0, 0)
    bounds::SVector{2, Float64} = SVector{2, Float64}(1, 1)

    # maximum resolution of each tracker
    max_depth::Int64
    # probablility of splitting node
    # TODO: currently unused (hard coded to 0.5)
    split_prob::Float64 = 0.5

    # coarsest node
    start_node::QTProdNode = QTProdNode(center, bounds, 1, max_depth, 1)

    #############################################################################
    # Planning / Navigation
    #############################################################################
    #
    entrance::Vector{Int64}
    exit::Vector{Int64}
    # weight to balance cost of distance with obstacle occupancy
    obs_cost::Float64 = 1.0

    #############################################################################
    # Graphics
    #############################################################################
    #
    renderer::QuadTreeRenderer
    # minimum variance in prediction
    pixel_var::Float32 = 1.0
end

function QuadTreeModel(gt::GridRoom;
                       render_kwargs::Dict,
                       kwargs...)
    QuadTreeModel(;
        dims = Rooms.steps(gt),
        entrance = entrance(gt),
        exit = exits(gt),
        max_depth = _max_depth(gt),
        renderer = QuadTreeRenderer(; render_kwargs...),
        kwargs...
    )
end

"""
Maximum depth of quad tree
"""
function _max_depth(r::GridRoom)
    @unpack bounds, steps = r
    # FIXME: allow for arbitrary room steps
    @assert all(ispow2.(steps)) "Room not a power of 2"
    convert(Int64, minimum(log2.(steps)) + 1)
end

#################################################################################
# Inference utils
#################################################################################

"""
    ridx_to_leaf(st, idx, c)

Returns

# Arguments

- `st::QuadTreeState`
- `ridx::Int64`: Linear index in room space
- `c::Int64`: Column size of room
"""
function room_to_leaf(qt::QuadTree, ridx::Int64, c::Int64)
    point = idx_to_node_space(ridx, c)
    traverse_qt(qt, point)
end


include("qt_model_gen.jl")
include("planning.jl")
