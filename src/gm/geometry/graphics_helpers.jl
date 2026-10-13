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
    # The quad tree's finest frame (L x L) and the render grid (d x d) need
    # not agree (e.g. a DDP-built qt with max_level 6 rendered at d = 16).
    # Map the leaf's finest-cell span [x0, x0+sz) x [y0, y0+sz) onto the
    # render grid by range division; `write_obstacles!` takes the max over
    # the (possibly overlapping) contributions. Works for L > d, L == d and
    # L < d. Output order is ascending in (c1, c2).
    L = finest_grid(qt)
    x, y = node_xy(n)                 # node-level coords at n.depth
    sz = leaf_cells_per_side(qt, n)
    x0 = x * sz                       # convert to finest-cell coords
    y0 = y * sz
    c1lo = (x0 * d) ÷ L + 1           # inclusive render column range
    c1hi = ((x0 + sz) * d - 1) ÷ L + 1
    c2lo = (y0 * d) ÷ L + 1           # inclusive render row range
    c2hi = ((y0 + sz) * d - 1) ÷ L + 1
    k  = 0
    for c1 in c1lo:c1hi, c2 in c2lo:c2hi
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
        w > threshold || continue
        k = leaf_lin_idxs!(buf, qt, n, d)
        for li in view(buf, 1:k)
            occ[li] = max(occ[li], w)
        end
    end
    return occ
end

function write_obstacles!(m::Matrix{Float32}, gr::GridRoom, d::Int;
                          buf::Vector{Int64} = Int64[], threshold::Float32 = 0.025f0)
    dmat = data(gr)
    size(dmat) == size(m) || throw(ArgumentError("Size missmatch"))
    for i = eachindex(m)
        m[i] = dmat[i] == obstacle_tile
    end
    return nothing
end
