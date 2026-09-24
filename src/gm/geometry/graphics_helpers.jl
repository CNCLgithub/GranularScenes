#################################################################################
# Graphics: dense-grid fill (replaces project_qt! / write_obstacles! inner loop)
#################################################################################


"""
    leaf_lin_idxs!(buf, qt, n, d) -> buf[1:k]

Fill `buf` with the linear indices (column-major, li = (c1-1)*d + c2) covered
by leaf `n` on a `d × d` render grid, in node_to_idx's y-descending, x-ascending
order. Returns the count.
"""
function leaf_lin_idxs!(buf::Vector{Int64}, qt::QuadTree, n::NodeId, d::Int)
    L = finest_grid(qt)
    f = d ÷ L
    x, y = node_xy(n)                 # node-level coords at n.depth
    sz = leaf_cells_per_side(qt, n)
    x0 = x * sz                       # convert to finest-cell coords
    y0 = y * sz
    w  = sz * f                       # leaf side in render cells
    k  = 0
    for j in 0:(w-1), i in 0:(w-1)
        c1 = x0 * f + i + 1           # render column (x)
        c2 = y0 * f + j + 1           # render row (y)
        buf[k += 1] = (c1 - 1) * d + c2
    end
    return k
end

"""
    write_obstacles!(occ, qt, d; threshold=0.025)

Dense d×d occupancy fill. Union semantics (max) as in the renderer.
`buf` must be preallocated with capacity ≥ d² for safety (a root leaf
 covers the whole grid).
"""
function write_obstacles!(occ::Matrix{Float32}, qt::QuadTree, d::Int;
                          buf::Vector{Int64} = Int64[], threshold::Float32 = 0.025f0)
    fill!(occ, 0.0f0)
    maxbuf = d * d                       # a root leaf covers the whole grid
    length(buf) < maxbuf && resize!(buf, maxbuf)
    for (i, n) in enumerate(qt.schema.leaves)
        w = qt.weight_map[n]
        w = w > threshold ? Float32(w) : 0.0f0
        w == 0.0f0 && continue
        k = leaf_lin_idxs!(buf, qt, n, d)
        for li in view(buf, 1:k)
            occ[li] = max(occ[li], w)
        end
    end
    return occ
end
