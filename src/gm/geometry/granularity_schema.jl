struct QTSchema
    max_level :: Int64                            # as in QTProdNode
    bounds    :: AABB2D{Float64}                  # root extent
    leaves    :: Vector{NodeId}
end

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
Base.hash(a::NodeId, h::UInt) = hash((a.depth, a.morton), h)
Base.isless(a::NodeId, b::NodeId) = (a.depth, a.morton) < (b.depth, b.morton)

"Split x,y bit-interleaved into a Morton code (x-ordering matches node_to_idx)."
function morton_code(x::Int, y::Int, d::Int)
    @assert 0 <= x < 2^(d-1) && 0 <= y < 2^(d-1)
    m = UInt32(0)
    for b in 0:d-2
        m |= UInt32(((x >> b) & 1) << (2b + 1))   # x on odd bits
        m |= UInt32(((y >> b) & 1) << (2b))       # y on even bits
    end
    m
end

child(n::NodeId, i::Int) = NodeId(n.depth + 1, (n.morton << 2) | UInt32(i - 1))
parent(n::NodeId) = n.depth > 1 ? NodeId(n.depth - 1, n.morton >> 2) : nothing
function xy(n::NodeId)  # de-interleave
    x, y = 0, 0
    for b in 0:n.depth-2
        x |= ((n.morton >> (2b + 1)) & 1) << b
        y |= ((n.morton >> (2b)) & 1) << b
    end
    (Int(x), Int(y))
end

struct QuadTree
    schema::QTSchema
    weight_map::Dict{NodeId, Float64}
end

"""
Returns the linear indices (column-major, li = (c1-1)*d + c2) covered by leaf n,
matching node_to_idx's ordering convention. Pure integer arithmetic, no LinRange.
"""
function leaf_lin_idxs(qt::QuadTree, n::NodeId, d::Int)
    L = 1 << (qt.schema.max_level - 1)   # finest cells per axis
    x, y = xy(n)
    lo = x << (qt.schema.max_level - n.depth)        # in finest units
    sz = L >> (qt.schema.max_level - n.depth)        # leaf extent in finest cells
    # scale finest grid to the render resolution d
    f = d ÷ L
    c1s = (lo*f + 1) : ((lo + sz)*f)                 # X / col (c1)
    c2s = (y*f + 1) : ((y + sz)*f)                   # Y / row (c2)
    # node_to_idx fills xs ascending, ys descending; over d×d col-major grid
    # li = (c1-1)*d + c2, so emit in matching order:
    idxs = Vector{Int64}(undef, length(c1s) * length(c2s))
    k = 0
    for yf in reverse(c2s), xf in c1s                # y descending, x ascending
        idxs[k += 1] = (xf - 1)*d + yf
    end
    return idxs
end
