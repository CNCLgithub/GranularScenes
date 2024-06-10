export QuadTreeModel


include("graphics.jl")

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
    center::SVector{2, Float64} = zeros(2)
    bounds::SVector{2, Float64} = ones(2)

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
    renderer::TaichiScene
    # minimum variance in prediction
    pixel_var::Float32 = 1.0
end

function QuadTreeModel(gt::GridRoom;
                       render_kwargs = Dict(),
                       kwargs...)
    QuadTreeModel(;
        dims = Rooms.steps(gt),
        entrance = entrance(gt),
        exit = exits(gt),
        max_depth = _max_depth(gt),
        renderer = TaichiScene(gt; render_kwargs...),
        kwargs...
    )
end

"""
Maximum depth of quad tree
"""
function _max_depth(r::GridRoom)
    @unpack bounds, steps = r
    # FIXME: allow for arbitrary room steps
    @assert all(ispow2.(steps))
    convert(Int64, minimum(log2.(steps)) + 1)
end


struct QTPath
    g::SimpleGraph
    dm::Matrix{Float64}
    edges::Vector{AbstractEdge}
end

function QTPath(st::QTAggNode)
    g = SimpleGraph(1)
    dm = Matrix{Float64}(undef, 1, 1)
    dm[1] = weight(st) * area(st.node)
    edges = [Graphs.SimpleEdge(1,1)]
    QTPath(g, dm, edges)
end

# struct QuadTreeState
#     qt::QuadTree
#     img_mu::Array{Float64, 3}
#     img_sd::Array{Float64, 3}
#     path::QTPath
# end


#################################################################################
# Graphics
#################################################################################

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

function create_obs(p::QuadTreeModel, r::GridRoom)
    _img = render(p.renderer, r)
    # need to reshape ti.field (m x n) -> array (m x n x 3)
    # for Gen.logpdf(observe_pixels)
    img = @pycall _img.to_numpy()::PyObject
    constraints = Gen.choicemap()
    constraints[:pixels] = img
    constraints
end

include("qt_model_gen.jl")
