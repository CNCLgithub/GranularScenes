using DataStructures

export QTPath,
    qt_a_star

struct QTPath
    g::SimpleGraph
    dm::Matrix{Float64}
    edges::Vector{AbstractEdge}
end

function QTPath(st::QTAggNode)
    g = SimpleGraph(1)
    dm = Matrix{Float64}(undef, 1, 1)
    dm[1] = weight(st) * area(st.node)
    edges = [Graphs.SimpleEdge(1,1)]
    QTPath(g, dm, edges)
end



function a_star_heuristic(qt::QuadTree, ent::Int, ext::Int)
    @unpack root, leaves, mapping = qt
    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = mapping[ent_node.node.tree_idx]

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = mapping[ext_node.node.tree_idx]

    a_star_heuristic(leaves, ext_node, 1.0)
end

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

---

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
    # # adjacency, distance matrix
    # ad, dm = nav_graph(leaves, dw)

    # g = SimpleGraph(ad)

    # # map entrance and exit in room to qt
    # row_d::Int64 = max_leaves(qt.root.node)
    # ent_point = idx_to_node_space(ent, row_d)
    # ent_node = traverse_qt(root, ent_point)
    # ent_idx = mapping[ent_node.node.tree_idx]

    # ext_point = idx_to_node_space(ext, row_d)
    # ext_node = traverse_qt(root, ext_point)
    # ext_idx = mapping[ext_node.node.tree_idx]

    # # L2 heuristic used in A*
    # heuristic = a_star_heuristic(leaves, ext_node, 1.0)
    # # compute path and path grid
    # path = a_star(g, ent_idx, ext_idx, dm, heuristic)
    # QTPath(g, dm, path)

    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = ent_node.node.tree_idx

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = ext_node.node.tree_idx

    open_set = PriorityQueue{Int64,Float64}()
    open_set[ent_idx] = 0.0

    closed_set = Set{Int64}()

    g_score = Dict{Int64, Float64}()
    g_score[ent_idx] = 0

    came_from = Dict{Int64, Int64}()
    came_from[ent_idx] = ent_idx


    heuristic = n -> dist(node(n), node(ext_node))

    qt_astar_impl!(
        ext_idx,
        open_set,
        closed_set,
        g_score,
        came_from,
        heuristic,
        qt,
        dw
    )

end


function qt_astar_impl!(
    goal::Int64, # the end vertex
    open_set, # an initialized heap containing the active vertices
    closed_set, # an (initialized) color-map to indicate status of vertices
    g_score, # a vector holding g scores for each node
    came_from, # a vector holding the parent of each node in the A* exploration
    heuristic,
    qt::QuadTree, dw::Float64)

    total_path = Int64[]
    score = Inf
    current = 0
    @inbounds while !isempty(open_set)
        current = dequeue!(open_set)
        cur_state = leaf_from_idx(qt, current)

        if current == goal
            reconstruct_path!(total_path, came_from, current)
            score = g_score[current]
            return score, total_path
        end

        push!(closed_set, current)
        # closed_set[current] = true

        adj = adjacent_leaves(qt, current)
        for neighbor in adj
            in(neighbor, closed_set) && continue
            n_state = leaf_from_idx(qt, neighbor)
            tentative_g_score = g_score[current] +
                traversal_cost(cur_state, n_state, dw)

            if tentative_g_score < get(g_score, neighbor, Inf)
                g_score[neighbor] = tentative_g_score
                priority = tentative_g_score + heuristic(n_state)
                open_set[neighbor] = priority
                came_from[neighbor] = current
            end
        end
    end
    return score, total_path
end

function reconstruct_path!(
    total_path::Vector{Int64}, # a vector to be filled with the shortest path
    came_from, # a vector holding the parent of each node in the A* exploration
    end_idx, # the end vertex
)
    curr_idx = end_idx
    while came_from[curr_idx] != curr_idx
        pushfirst!(total_path, curr_idx)
        curr_idx = came_from[curr_idx]
    end
    pushfirst!(total_path, curr_idx)
    return nothing
end

function traversal_cost(src::QTAggNode, dst::QTAggNode, obs_cost::Float64)
    d = dist(node(dst), node(src))
    # c = obs_cost * (weight(dst) * length(node(dst)) +
    #     weight(src) * length(node(src)))
    c = obs_cost * (weight(dst) + weight(src))
    d * c
end
