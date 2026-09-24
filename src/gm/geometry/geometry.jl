export QuadTree, NodeId, GranularitySchema

include("spatial_primitives.jl")

#################################################################################
# Keys
#################################################################################

"""
A node identity: (depth, Morton code). Parent/child links are pure arithmetic:
children of (d, m) are (d+1, 4m+i), parent of (d, m) is (d-1, m >> 2).
Depth 1 = root (matching QTProdNode's level convention).
"""
struct NodeId
    depth  :: UInt8
    morton :: UInt32
end

Base.:(==)(a::NodeId, b::NodeId) = a.depth == b.depth && a.morton == b.morton
Base.hash(a::NodeId, h::UInt) = hash((a.depth, Int(a.morton)), h)
Base.isless(a::NodeId, b::NodeId) =
    a.depth < b.depth || (a.depth == b.depth && a.morton < b.morton)
Base.show(io::IO, n::NodeId) =
    print(io, "NodeId(d=$(Int(n.depth)), m=0b", bitstring(n.morton), ")")

"""
    morton_code(x, y, d)

Split x,y bit-interleaved into a Morton code at depth `d` (depth-1 grid is 1×1).
X occupies odd bits, Y even bits (matches `node_to_idx`'s x-then-y ordering).
"""
function morton_code(x::Int, y::Int, d::Int)
    m = UInt32(0)
    @inbounds for b in 0:(d-2)
        m |= UInt32(((x >> b) & 1) << (2b + 1))
        m |= UInt32(((y >> b) & 1) << 2b)
    end
    m
end

child_key(n::NodeId, i::Int) = NodeId(n.depth + 1, (n.morton << 2) | UInt32(i - 1))
parent_key(n::NodeId) = n.depth > 1 ? NodeId(n.depth - 1, n.morton >> 2) : nothing

"""
    node_xy(n)

De-interleave a Morton key into (x, y) cell indices at the node's depth.
"""
function node_xy(n::NodeId)
    x = y = 0
    @inbounds for b in 0:(n.depth - 2)
        x |= ((n.morton >> (2b + 1)) & 1) << b
        y |= ((n.morton >> 2b) & 1) << b
    end
    x, y
end

#################################################################################
# Schema and tree
#################################################################################

"""
Invariant granularity schema: max resolution, world extent, and the latent
structure (the leaf set). Split/merge rewrites `leaves`; the rest never changes.
"""
struct QTSchema
    max_level :: Int64
    bounds    :: AABB2D{Float64}
    leaves    :: Vector{NodeId}
end

"""
A linear quadtree keyed by NodeId. Structure is the sorted leaf set;
`weight_map` holds per-leaf occupancy weights.
"""
struct QuadTree
    schema     :: QTSchema
    weight_map :: Dict{NodeId, Float64}
end

# convenience accessors ---------------------------------------------------------

@inline max_level(qt::QuadTree) = qt.schema.max_level
@inline nleaves(qt::QuadTree) = length(qt.schema.leaves)
@inline weight_of(qt::QuadTree, n::NodeId) = get(qt.weight_map, n, 0.0)

"Finest cells per axis."
@inline finest_grid(qt::QuadTree) = 1 << (max_level(qt) - 1)

"Edge length of leaf `i` in finest cells."
@inline leaf_cells_per_side(qt::QuadTree, n::NodeId) =
    1 << (max_level(qt) - n.depth)

"World-space AABB of a node key."
function node_bounds(qt::QuadTree, n::NodeId)
    L = finest_grid(qt)
    s = 1.0 / L   # schema bounds assumed [-0.5, 0.5]^2, as in the renderer
    x, y = node_xy(n)
    sz = leaf_cells_per_side(qt, n)
    lo = SVector((x - L/2) * s, (y - L/2) * s)
    AABB2D(lo, lo .+ sz .* s)
end

center(n::NodeId, qt::QuadTree) = center(node_bounds(qt, n))
dist(a::NodeId, b::NodeId, qt::QuadTree) = norm(center(a, qt) - center(b, qt))
edge_length(n::NodeId, qt::QuadTree) = 2.0 * leaf_cells_per_side(qt, n) / finest_grid(qt)

"""
    find_leaf_index(qt, n) -> position in qt.schema.leaves, or 0 if absent.

Binary search over the sorted leaf vector.
"""
function find_leaf_index(qt::QuadTree, n::NodeId)
    r = searchsortedfirst(qt.schema.leaves, n)
    (r <= length(qt.schema.leaves) && qt.schema.leaves[r] == n) ? r : 0
end

# construction ------------------------------------------------------------------

"Root-only tree with weight μ₀."
function QuadTree(max_level::Int, μ₀::Float64)
    root = NodeId(UInt8(1), UInt32(0))
    schema = QTSchema(max_level, AABB2D(SVector(-0.5, -0.5), SVector(0.5, 0.5)),
                      [root])
    QuadTree(schema, Dict{NodeId,Float64}(root => μ₀))
end

"""
    validate(qt) -> (ok::Bool, msg::String)

Invariants: leaves sorted, unique, no leaf is an ancestor of another leaf,
all weights finite and in [0, 1].
"""
function validate(qt::QuadTree)
    issorted(qt.schema.leaves) || return (false, "leaves not sorted")
    allunique(qt.schema.leaves) || return (false, "duplicate leaves")
    leafset = Set(qt.schema.leaves)
    for n in qt.schema.leaves
        p = parent_key(n)
        while p !== nothing
            p in leafset && return (false, "leaf $n nested inside leaf $p")
            p = parent_key(p)
        end
        0.0 <= weight_of(qt, n) <= 1.0 ||
            return (false, "weight out of range at $n")
    end
    return (true, "")
end

include("graphics_helpers.jl")
