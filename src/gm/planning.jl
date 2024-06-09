function a_star_heuristic(nodes::Vector{QTAggNode}, dest::QTAggNode,
                          scale::Float64)
    src -> dist(nodes[src].node, dest.node) * scale
end
function a_star_heuristic(nodes::Vector{QTAggNode}, dest::Int64, scale::Float64)
    _dest = nodes[dest]
    a_star_heuristic(nodes, _dest)
end

function nav_graph(lv::Vector{QTAggNode}, w::Float64)
    n = length(lv)
    adm = fill(false, (n, n))
    dsm = fill(Inf, (n, n))

    @inbounds for i = 1:(n-1), j = (i+1):n
        x = lv[i]
        y = lv[j]
        # only care when nodes are touching
        contact(x.node, y.node) || continue
        d = dist(x.node, y.node)
        #  work to traverse each node
        work = d + w*(length(x.node) * weight(x) + length(y.node)*weight(y))
        adm[i, j] = adm[j, i] = true
        dsm[i, j] = dsm[j, i] = work
    end
    (adm, dsm)
end

"""
    qt_a_star(qt, d, ent, ext)

Applies `a_star` to the quad tree.

# Arguments
- `qt::QuadTree`: A quad tree to traverse over leaves
- `d::Int64`: The row dimensions of the room
- `ent::Int64`: The entrance tile
- `ext::Int64`: The exit tile

# Returns
A tuple, first element is `QTPath` and the second is a vector
 of the leave nodes in QT.
"""
function qt_a_star(qt::QuadTree, dw::Float64, ent::Int64, ext::Int64)
    @unpack root, leaves, mapping = qt
    length(leaves) == 1 && return QTPath(first(leaves))
    # adjacency, distance matrix
    ad, dm = nav_graph(leaves, dw)

    g = SimpleGraph(ad)

    # map entrance and exit in room to qt
    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = mapping[ent_node.node.tree_idx]

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = mapping[ext_node.node.tree_idx]

    # L2 heuristic used in A*
    heuristic = a_star_heuristic(leaves, ext_node, 1.0)
    # compute path and path grid
    path = a_star(g, ent_idx, ext_idx, dm, heuristic)
    QTPath(g, dm, path)
end
