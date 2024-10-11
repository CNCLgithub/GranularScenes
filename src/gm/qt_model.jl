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
    @assert all(ispow2.(steps)) "Room not a power of 2"
    convert(Int64, minimum(log2.(steps)) + 1)
end

function apply_changes(qt::QuadTree, changes::AbstractArray{Float64})
    # HACK: does not update parent statistics
    lvs = leaves(qt)
    n = length(lvs)
    @assert length(changes) === n "changes missmatch with qt leaves"
    new_leaves = Vector{QTAggNode}(undef, n)
    @inbounds for i = 1:n
        x = lvs[i]
        c = changes[i]
        # REVIEW: delta based on node size?
        w = c * (1.0 - x.mu) + (1.0 - c) * x.mu
        new_leaves[i] =
            QTAggNode(w, x.u, x.k, x.leaves, x.node, x.children)
    end
    QuadTree(qt.root, new_leaves, qt.mapping)
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

function create_obs(p::QuadTreeModel, first::GridRoom, second::GridRoom)
    _img1 = render(p.renderer, first)
    img1 = @pycall _img1.to_numpy()::PyObject
    _img2 = render(p.renderer, second)
    img2 = @pycall _img2.to_numpy()::PyObject
    constraints = Gen.choicemap()
    constraints[:img_a] = img1
    constraints[:img_b] = img2
    constraints
end

function create_obs(p::QuadTreeModel, r::GridRoom)
    _img = render(p.renderer, r)
    img = @pycall _img.to_numpy()::PyObject
    constraints = Gen.choicemap()
    constraints[:pixels] = img
    constraints
end

include("qt_model_gen.jl")
include("planning.jl")
