export QuadTreeModel, QTTrace


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
    convert(Int64, log2(minimum(steps)) + 1)
end

function change_weights(qt::QuadTree)
    lvs = leaves(qt)
    n = length(lvs)
    ws = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        x = lvs[i]
        w = weight(x)
        log_prop_tiles = 2 * (1 - level(node(x)))
        uncertainty = log(1.0 - w)
        # uncertainty: 2 * abs(w - 0.5) [0, 1]
        # size: ntiles / total tiles [0, 1]
        # ws[i] = fast_sigmoid((2 + log(abs(0.5 - w)) + log_prop_tiles))
        # ws[i] = fast_sigmoid(2 * abs(0.5 - w))
        # ws[i] = log_prop_tiles
        ws[i] = uncertainty
    end
    ws = softmax(ws)
end

function apply_changes(qt::QuadTree, idx::Int)
    idx == 0 && return qt
    # HACK: does not update parent statistics
    lvs = leaves(qt)
    n = length(lvs)
    @assert idx <= n && idx > 0 "changes missmatch with qt leaves"
    @inbounds x = lvs[idx]
    # Assume the change adds an obstacle.
    # Get the amount adding a new obstacle changes the overall
    # density of the node
    max_depth = node(x).max_level
    depth = node(x).level
    delta_mass = exp2(-2 * (max_depth - depth))
    w = min(1.0, x.mu + delta_mass)
    new_leaves = deepcopy(lvs)
    new_leaves[idx] =
        QTAggNode(w, x.u, x.k, x.leaves, x.node, x.children)
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

function create_obs(p::QuadTreeModel, r::GridRoom,
                    key = :img_a)
    _img = render(p.renderer, r)
    img = @pycall _img.to_numpy()::PyObject
    constraints = Gen.choicemap()
    constraints[key] = img
    constraints
end

include("qt_model_gen.jl")


gen_fn(::QuadTreeModel) = qt_model
# const QTModelIR = Gen.get_ir(qt_model)
const QTTrace = Gen.get_trace_type(qt_model)

include("planning.jl")
