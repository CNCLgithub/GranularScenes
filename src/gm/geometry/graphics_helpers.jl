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
    x, y = node_xy(n)
    sz = leaf_cells_per_side(qt, n)
    k = 0
    for yf in (y+sz-1):-1:y                    # y descending
        for xf in x:(x+sz-1)                   # x ascending
            buf[k += 1] = (xf*f + 1 - 1)*d + (yf*f + 1)
        end
    end
    return k
end

"""
    write_obstacles!(occ, qt, d; threshold=0.025)

Dense d×d occupancy fill. Union semantics (max) as in the renderer.
`buf` must be preallocated with capacity ≥ 4·finest_grid(qt)^2 for safety,
though a single leaf never covers more than sz² ≤ L² cells.
"""
function write_obstacles!(occ::Matrix{Float32}, qt::QuadTree, d::Int;
                          buf::Vector{Int64} = Int64[], threshold::Float32 = 0.025f0)
    fill!(occ, 0.0f0)
    maxbuf = finest_grid(qt)^2          # a single leaf covers at most L² cells
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
