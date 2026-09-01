export QTProdNode, QTAggNode, QuadTree

include("spatial_primitives.jl")

#################################################################################
# Production
#################################################################################

"""
A production node in the quad tree. Defines a spatially oriented rectangle.

# Properties

- `bounds`: Axis-aligned bounding box (continuous coordinates)
- `level`: The number of splits with `level==1` denoting no splits
- `max_level`: The maximum number of splits allowed
- `tree_idx`: The Gen-trace index of the node in the production trace
"""
struct QTProdNode
    bounds::AABB2D{Float64}
    level::Int64
    max_level::Int64
    tree_idx::Int64
end

QTProdNode(c::SVector{2, Float64}, d::SVector{2, Float64}, l, ml, ti) =
    QTProdNode(AABB2D(c, d), l, ml, ti)

center(n::QTProdNode) = center(n.bounds)
dims(n::QTProdNode) = SVector{2, Float64}(n.bounds.xmax - n.bounds.xmin,
                                          n.bounds.ymax - n.bounds.ymin)
area(n::QTProdNode) = area(n.bounds)
contains(n::QTProdNode, p::SVector{2, Float64}) = contains(n.bounds, p)
contact(a::QTProdNode, b::QTProdNode) = touches(a.bounds, b.bounds)

Base.length(n::QTProdNode) = n.bounds.xmax - n.bounds.xmin

"""
    max_leaves(n::QTProdNode)

The finest resolutions supported.
"""
max_leaves(n::QTProdNode) = 2^(n.max_level - 1)

"""
    pos_to_idx(pos, n)

Maps R^2 position of QTProdNode to a linear index in nxn.
"""
function pos_to_idx(pos::SVector{2, Float64}, n::Int64)
    g = GridTransform(n)
    c = pos_to_index(g, pos)
    (c[1] - 1) * n + c[2]
end

"""
    node_to_idx(n::QTProdNode, d::Int64)

Maps a node to the linear indices of its covered cells in a `d × d` grid.
"""
function node_to_idx(n::QTProdNode, d::Int64)
    g = GridTransform(d)
    @unpack level, max_level = n
    if level == max_level
        return [pos_to_idx(center(n), d)]
    end
    # offset from each boundary
    fac = 0.5 - (1.0 / exp2(max_level - level + 1))
    lower = center(n) - fac .* dims(n)
    upper = center(n) + fac .* dims(n)
    steps = Int64(exp2(max_level - level))
    # xy ordering to match julia col-wise
    # doesn't actually matter for square scenes
    xs = LinRange(lower[1], upper[1], steps)
    ys = LinRange(upper[2], lower[2], steps)
    idx = Vector{Int64}(undef, steps^2)
    for (i, (y, x)) in enumerate(product(ys, xs))
        idx[i] = pos_to_idx(SVector{2, Float64}([x, y]), d)
    end
    return idx
end

"""
    idx_to_node_space(i, d)

Maps a linear index in `d × d` to the R^2 plane `[-0.5, 0.5]`.
"""
function idx_to_node_space(i::Int64, d::Int64)
    g = GridTransform(d)
    index_to_pos(g, i)
end

dist(x::QTProdNode, y::QTProdNode) = norm(center(x) - center(y))

# REVIEW: Some way to parameterize weights?
function produce_weight(n::QTProdNode)::Float64
    @unpack level, max_level = n
    # level == max_level ? 0. : 0.5
    level == 1 ? 0.99 :
        (level == max_level ? 0. : 0.5)
end

# quadrant offset constants (relative to parent center, scaled by 0.25*dims)
const q1 = SVector{2, Float64}([-1.0,  1.0])
const q2 = SVector{2, Float64}([-1.0, -1.0])
const q3 = SVector{2, Float64}([ 1.0,  1.0])
const q4 = SVector{2, Float64}([ 1.0, -1.0])

"""
Splits the node into 4 children, centered in 4 quadrants
with respect to the center of the parent.
"""
function produce_qt(n::QTProdNode)::SVector{4, QTProdNode}
    @unpack center, level, max_level = n
    dc = 0.25 .* dims(n)  # offset by 1/4 of dims
    new_dims = 0.5 .* dims(n)
    offs = (q1 .* dc, q2 .* dc, q3 .* dc, q4 .* dc)
    SVector{4, QTProdNode}((
        QTProdNode(center + offs[1], new_dims, level + 1, max_level,
                   Gen.get_child(n.tree_idx, 1, 4)),
        QTProdNode(center + offs[2], new_dims, level + 1, max_level,
                   Gen.get_child(n.tree_idx, 2, 4)),
        QTProdNode(center + offs[3], new_dims, level + 1, max_level,
                   Gen.get_child(n.tree_idx, 3, 4)),
        QTProdNode(center + offs[4], new_dims, level + 1, max_level,
                   Gen.get_child(n.tree_idx, 4, 4)),
    ))
end


#################################################################################
# Aggregation
#################################################################################

struct QTAggNode
    mu::Float64
    u::Float64
    k::Int64
    leaves::Int64
    node::QTProdNode
    children::Vector{QTAggNode}
end

weight(st::QTAggNode) = st.mu
dof(st::QTAggNode) = st.u
leaves(st::QTAggNode) = st.leaves
node(st::QTAggNode) = st.node
Base.length(st::QTAggNode) = st.k

"""
Aggregates quad tree production nodes into a tree, keeping track of DOF.
"""
function QTAggNode(n::QTProdNode, y::Float64, children::Vector{QTAggNode})
    if isempty(children)
        u  = 0.0
        k = 1
        l = 1
    else
        # equal area => mean of variance
        u = sqrt(mean(dof.(children).^2))
        k = sum(length.(children)) + 1
        l = sum(leaves.(children))
    end
    QTAggNode(y, u, k, l, n, children)
end

function contains(st::QTAggNode, p::SVector{2, Float64})
    contains(st.node, p)
end

max_leaves(n::QTAggNode) = max_leaves(n.node)

"""
    leaf_vec(s)

Returns all of the leaf `QTAggNode`s from root `s`.
"""
function leaf_vec(s::QTAggNode)::Vector{QTAggNode}
    v = Vector{QTAggNode}(undef, s.leaves)
    add_leaves!(v, s)
    return v
end

function add_leaves!(v::Vector{QTAggNode}, s::QTAggNode)
    heads::Vector{QTAggNode} = [s]
    i::Int64 = 1
    while !isempty(heads)
        head = pop!(heads)
        if isempty(head.children)
            v[i] = head
            i += 1
        else
            append!(heads, head.children)
        end
    end
    return nothing
end

function leaf_mapping(lv::Vector{QTAggNode})::Dict{Int64, Int64}
    mapping = Dict{Int64, Int64}()
    for i = 1:length(lv)
        leaf = @inbounds lv[i]
        mapping[leaf.node.tree_idx] = i
    end
    return mapping
end


#################################################################################
# Tree and traversal
#################################################################################

struct QuadTree
    root::QTAggNode
    leaves::Vector{QTAggNode}
    mapping::Dict{Int64, Int64}
end

function QuadTree(root::QTAggNode)
    lvs = leaf_vec(root)
    mapping = leaf_mapping(lvs)
    nmax = max_leaves(root.node)
    QuadTree(root, lvs, mapping)
end

max_leaves(qt::QuadTree) = max_leaves(qt.root.node)

"""
    get_depth(n::Int64)

Returns the depth of a node in a quad tree,
using Gen's `Recurse` indexing system.
"""
function get_depth(n::Int64)
    head = n
    d::Int64 = 1
    while head > 1
        head = Gen.get_parent(head, 4)
        d += 1
    end
    d
end

"""
    traverse_qt(root, dest)

Returns the quad tree node at index `dest` (Gen `Recurse` indexing).
"""
function traverse_qt(root::QTAggNode, dest::Int64)
    # No children or root is destination
    (isempty(root.children) || dest == 1) && return root
    d = get_depth(dest) - 1
    path = Vector{Int64}(undef, d)
    idx = dest
    @inbounds for i = 1:d
        path[d - i + 1] = Gen.get_child_num(idx, 4)
        idx = Gen.get_parent(idx, 4)
    end
    for i = 1:d
        # in the context of split-merge,
        # for the backward of a merge, t_prime will not
        # have a child at the last step
        isempty(root.children) && break
        root = root.children[path[i]]
    end
    root
end

"""
    traverse_qt(root, dest)

Returns the *smallest* quad tree node that contains `dest`.
"""
function traverse_qt(root::QTAggNode, dest::SVector{2, Float64})
    # assuming root must contain dest
    head = root
    while !isempty(head.children)
        idx = findfirst(s -> contains(s, dest), head.children)
        head = @inbounds head.children[idx]
    end
    return head
end

traverse_qt(qt::QuadTree, dest) = traverse_qt(qt.root, dest)


function project_qt(qt::QuadTree)
    project_qt(qt.leaves, max_leaves(qt))
end

"""
    project_qt(lv, dims)

Projects the quad tree to a nxn matrix

# Arguments
- `lv::Vector{QTAggNode}`: The leaves of a quad tree
- `n`: Maximum number of leaves possible (see `max_leaves`)
"""
function project_qt(lv::Vector{QTAggNode}, n::Int64)
    gs = Matrix{Float32}(undef, n, n)
    project_qt!(gs, lv)
    return gs
end

function project_qt!(gs::Matrix{Float32},
                     lv::Vector{QTAggNode})
    d = size(gs, 1)
    for x in lv
        idx = node_to_idx(x.node, d)
        w = weight(x)
        w = w > 0.025 ? w : 0.0
        for i = idx
            gs[i] = w
        end
    end
    return nothing
end

include("qt_prior_gen.jl")
