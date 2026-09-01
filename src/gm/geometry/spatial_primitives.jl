export AABB2D, GridTransform, contains, area, center


"""
    AABB2D{Float64}

An axis-aligned bounding box in 2D continuous coordinates,
stored as `[xmin, ymin, xmax, ymax]`.
"""
struct AABB2D{T}
    xmin::T
    ymin::T
    xmax::T
    ymax::T
end

AABB2D(center::SVector{2, Float64}, dims::SVector{2, Float64}) =
    AABB2D(center[1] - 0.5 * dims[1],
           center[2] - 0.5 * dims[2],
           center[1] + 0.5 * dims[1],
           center[2] + 0.5 * dims[2])


area(b::AABB2D) = (b.xmax - b.xmin) * (b.ymax - b.ymin)

center(b::AABB2D) = SVector{2, Float64}(0.5 * (b.xmin + b.xmax),
                                        0.5 * (b.ymin + b.ymax))

contains(b::AABB2D, p::SVector{2, Float64}) =
    b.xmin <= p[1] <= b.xmax && b.ymin <= p[2] <= b.ymax

"""
    touches(a::AABB2D, b::AABB2D)

True if the two boxes share at least one point (zero-area contact counts).
Equivalent to the old `contact(::QTProdNode, ::QTProdNode)` test.
"""
touches(a::AABB2D, b::AABB2D) =
    a.xmin <= b.xmax && b.xmin <= a.xmax &&
    a.ymin <= b.ymax && b.ymin <= a.ymax


"""
    GridTransform(n)

Maps continuous coordinates in `[-0.5, 0.5]^2` to integer grid indices
`[1, n]^2` via `index(pos, n) = ceil(n * (pos + 0.5))`.

- `pos_to_index(pos)` -> `(col, row)` (Julia column-major order)
- `index_to_pos(i)`    -> continuous center of grid cell `i`
"""
struct GridTransform
    n::Int64

    function GridTransform(n::Int64)
        n > 0 || error("GridTransform requires n > 0")
        new(n)
    end
end

Base.length(g::GridTransform) = g.n

"""
    pos_to_index(g, pos)

Maps `pos ∈ [-0.5, 0.5]^2` to a `(col, row)` grid index in `[1, n]^2`.
"""
function pos_to_index(g::GridTransform, pos::SVector{2, Float64})
    c = @. ceil(Int64, g.n * (pos + 0.5))
    (c[1], c[2])
end

"""
    index_to_pos(g, i)

Maps a linear column-major index `i ∈ [1, n^2]` back to the continuous
center of its grid cell.
"""
function index_to_pos(g::GridTransform, i::Int64)
    n = g.n
    row = ((i - 1) % n) + 1
    col = (i - 1) ÷ n + 1
    SVector{2, Float64}((col - 0.5) / n - 0.5,
                        (row - 0.5) / n - 0.5)
end
