"""
    qt_a_star(qt, obs_cost, ent, ext; τ=1.0) ->
        (score, path::Vector{NodeId}, softgrads::Dict{NodeId, Float64})

A* over the leaf adjacency graph. `ent` / `ext` are world-space positions
(SVector{2}) or finest-cell linear indices (converted via idx_to_node_space
semantics). Vertices are NodeIds; all solver state is Dict/Set keyed.

In addition to the optimal path, returns δkπ as a soft-path-gradient:
the gradient of the τ-smoothed A* objective (Boltzmann/soft-min value
function, cf. soft Q-learning, maximum-entropy planning) w.r.t. each
leaf's occupancy weight. Because cost is linear in the weights,
    ∂V_τ(start)/∂w_k = obs_cost · 2^(depth_k−1) · p_k
where p_k is the posterior probability that the smooth path distribution
traverses leaf k. Nonzero on near-optimal corridors, not just the argmin
path — fixing the support collapse of the hard-argmin estimator.

Accumulation rule: every relaxation (u → v) contributes its Boltzmann
mass, exp(−(g+h)/τ) normalized over the open set at that expansion,
weighted by the per-leaf cost coefficient. The normalization makes the
estimator a proper smoothing: at τ → 0 it recovers the indicator of the
optimal path; larger τ spreads mass over the frontier.
"""
function qt_a_star(qt::QuadTree, obs_cost::Float64, ent::S2V, ext::S2V,
                   τ::Float64 = 1.0)
    src  = leaf_at(qt, ent)
    goal = leaf_at(qt, ext)

    open_set = PriorityQueue{NodeId, Float64}()
    open_set[src] = 0.0
    closed_set = Set{NodeId}()
    g_score = Dict{NodeId, Float64}(src => 0.0)
    came_from = Dict{NodeId, NodeId}(src => src)
    heuristic = n -> dist(n, goal, qt)

    # δkπ accumulation: raw per-leaf soft mass, then normalized.
    soft_mass = Dict{NodeId, Float64}()

    neighbor_buf = Vector{NodeId}(undef, max_neighbors(qt))
    found = false
    while !isempty(open_set)
        # Boltzmann mass over the open set at this expansion:
        # p(node expanded) ∝ exp(−(g+h)/τ). Each expansion contributes
        # mass × (per-leaf cost coefficient) to that leaf's δkπ.
        _, fmin = minimum(open_set)
        Z = 0.0
        for (_, f) in open_set
            Z += exp(-(f - fmin) / τ)
        end
        for node in keys(open_set)
            mass = exp(-(open_set[node] - fmin) / τ) / Z
            soft_mass[node] = get(soft_mass, node, 0.0) +
                mass * obs_cost * exp2(node.depth - 1)
        end
        current = dequeue!(open_set)
        current == goal && (found = true; break)
        push!(closed_set, current)
        k = adjacent_leaves!(neighbor_buf, qt, current)
        for t in 1:k
            nb_id = neighbor_buf[t]
            nb_id in closed_set && continue
            tentative = g_score[current] +
                traversal_cost(qt, current, nb_id, obs_cost)
            old = get(g_score, nb_id, Inf)
            if tentative < old
                g_score[nb_id] = tentative
                came_from[nb_id] = current
                open_set[nb_id] = tentative + heuristic(nb_id)
            end
        end
    end
    found || return (Inf, NodeId[], soft_mass)

    # reconstruct
    path = NodeId[]
    cur = goal
    while came_from[cur] != cur
        pushfirst!(path, cur)
        cur = came_from[cur]
    end
    pushfirst!(path, cur)
    return (g_score[goal], path, soft_mass)
end

#################################################################################
# Point lookup
#################################################################################


"""
    leaf_at(qt, p) -> NodeId

Smallest leaf containing world point `p`. Compute the finest cell, then climb
ancestors until a leaf is found (presence test = `weight_map` key). O(D)
dict probes.
"""
function leaf_at(qt::QuadTree, p::SVector{2, Float64})
    L = finest_grid(qt)
    b = qt.schema.bounds
    u = clamp((p[1] - b.xmin) / (b.xmax - b.xmin), 0.0, 1 - 1e-12)
    v = clamp((p[2] - b.ymin) / (b.ymax - b.ymin), 0.0, 1 - 1e-12)
    x = min(Int(floor(u * L)), L - 1)
    y = min(Int(floor(v * L)), L - 1)
    for d in max_level(qt):-1:1
        n = NodeId(UInt8(d), morton_code(x, y, d))
        haskey(qt.weight_map, n) && return n
    end
    error("point outside tree: $p")
end

#################################################################################
# Adjacency — preallocated, no dynamic resizing
#################################################################################


"""
    max_neighbors(qt) -> Int

Upper bound on a leaf's neighbor count: each face of a finest-cell leaf has
1 neighbor, and a coarse leaf has at most sz cells per face × 4 faces.
The global bound is 4·L, the per-depth bound 4·(L >> (D - d)).
"""
max_neighbors(qt::QuadTree) = 4 * finest_grid(qt)

"""
    adjacent_leaves!(out, qt, n) -> count

Face-neighbors of leaf `n`, via per-direction descent (Samet-style neighbor
finding over Morton keys). For each of the 4 directions:

  1. compute the same-size neighbor block with O(1) field arithmetic;
  2. climb ancestors: the first present node at depth ≤ n.depth covers the
     entire face segment — emit it, one neighbor;
  3. otherwise the region is subdivided finer than `n`: recurse over the
     blocks intersecting the face, emitting present leaves (each appears
     once, since children along a face are disjoint).

Dict probes are O(D + k) per face, where k is the number of neighbors on
that face — proportional to output size and independent of the leaf's
extent (vs. a finest-cell walk at O(sz·D)).

`out` must have capacity ≥ max_neighbors(qt).
"""
function adjacent_leaves!(out::Vector{NodeId}, qt::QuadTree, n::NodeId)
    D = max_level(qt)
    L = finest_grid(qt)
    x, y = node_xy(n)
    sz = leaf_cells_per_side(qt, n)
    stack = Vector{NTuple{3, Int64}}(undef, 3 * max_level(qt))
    cnt = 0
    for dir in 1:4                  # 1: −x  2: +x  3: −y  4: +y
        bx, by = dir == 1 ? (x*sz - 1, y*sz) :
                 dir == 2 ? (x*sz + sz, y*sz) :
                 dir == 3 ? (x*sz, y*sz - 1) : (x*sz, y*sz + sz)
        # outside the grid → this face borders the world edge
        (0 <= bx < L) && (0 <= by < L) || continue
        # Step 2: climb from the same-size neighbor cell; the first present
        # ancestor at depth ≤ n.depth covers the whole face segment.
        found = false
        d = n.depth
        while d >= 1
            cell = NodeId(UInt8(d),
                          morton_code(bx >> (D - d), by >> (D - d), Int64(d)))
            if haskey(qt.weight_map, cell)
                out[cnt += 1] = cell
                found = true
                break
            end
            d -= 1
        end
        found && continue
        # Step 3: no same-size or coarser node present → subdivided finer.
        # The block at n.depth is internal; expand its children.
        cnt += _descend_face!(out, cnt, qt, stack, n.depth,
                              bx >> (D - n.depth), by >> (D - n.depth), D)
    end
    return cnt
end

"""
    _descend_face!(out, k, qt, d, cx, cy, D) -> count emitted

Emit the leaves covering the face-segment block of side 2^(D−d) at depth-d
coordinates (cx, cy). Precondition: no present node at any depth < d covers
the block. For each child: present → emit; absent and d+1 < D → expand;
at depth D the block is necessarily a leaf (leaves partition the plane).

Recursion-free: an explicit LIFO stack of (depth, cx, cy) work items,
allocated by the caller (capacity 3·D; each expansion pushes at most 3
items and depth ≤ D). Emission order is depth-first, children pushed in
reverse so the +q child is processed first; the emitted set is identical
to the recursive version.
"""
function _descend_face!(out::Vector{NodeId}, k::Int, qt::QuadTree,
                        stack::Vector{NTuple{3, Int64}},
                        d::Int, cx::Int, cy::Int, D::Int)
    cnt = 0
    sp = 1
    stack[1] = (d, cx, cy)
    while sp > 0
        (dd, xx, yy) = stack[sp]
        sp -= 1
        for q in 3:-1:0
            cxx = (xx << 1) | (q & 1)
            cyy = (yy << 1) | (q >> 1)
            ck = NodeId(UInt8(dd + 1), morton_code(cxx, cyy, dd + 1))
            if haskey(qt.weight_map, ck)
                out[k + cnt + 1] = ck
                cnt += 1
            elseif dd + 1 < D
                sp += 1
                stack[sp] = (dd + 1, cxx, cyy)
            end
        end
    end
    return cnt
end

#################################################################################
# A* navigation over the leaf graph
#################################################################################

"""
    collision_score(qt, n)

Expected number of finest-cell collisions traversing leaf `n`:
weight × 2^(depth-1), as in the original collision_score.
"""
collision_score(qt::QuadTree, n::NodeId) =
    weight_of(qt, n) * exp2(n.depth - 1)

"""
    traversal_cost(qt, n, m, obs_cost)

Cost of traversing between touching leaves `n` and `m`:
distance between centers + obstacle penalty on both nodes.
"""
function traversal_cost(qt::QuadTree, n::NodeId, m::NodeId, obs_cost::Float64)
    obs_cost * (collision_score(qt, n) + collision_score(qt, m)) +
        dist(n, m, qt)
end


"Entry/exit may be world points or finest-cell indices (1-based, row-major)."
_to_point(i::Int, qt::QuadTree) = idx_to_node_space(i, finest_grid(qt))
_to_point(p::SVector{2, Float64}, qt::QuadTree) = p

"""

Maps a linear index in nxn to a R^2 plane [-0.5, 0.5]

# Arguments
- i: linear index
- d: number of columns
"""
function idx_to_node_space(i::Int64, d::Int64)
    index_to_pos(GridTransform(d), i)
end

"""
    room_index_to_point(r::GridRoom, i) -> SVector{2, Float64}

Map a `GridRoom` linear tile index (column-major over `steps(r)`, fast axis =
`steps[1]`) to node space [-0.5, 0.5]^2. Cell centers, matching
`index_to_pos` semantics but with the room's own stride.
"""
function room_index_to_point(r::GridRoom, i::Int)
    nr, nc = steps(r)
    row = (i - 1) % nr + 1      # fast axis = steps[1] = row -> pos[2] (y)
    col = (i - 1) ÷ nr + 1      # slow axis = steps[2] = col -> pos[1] (x)
    SVector{2, Float64}((col - 0.5) / nc - 0.5,
                        (row - 0.5) / nr - 0.5)
end

"""
    room_index_to_qt(r::GridRoom, i, qt) -> Int

Remap a `GridRoom` tile index onto the QT's finest grid: locate the tile
center in node space, then re-linearize at stride `finest_grid(qt)`. Use this
for anything fed to `AStarPlanner` / `qt_a_star` as a linear index.
"""
function room_index_to_qt(r::GridRoom, i::Int, qt::QuadTree)
    L = finest_grid(qt)
    p = room_index_to_point(r, i)
    # leaf_at(qt, p)
    # c = p .+ 0.5
    # col = clamp(ceil(Int, c[1] * L), 1, L)
    # row = clamp(ceil(Int, c[2] * L), 1, L)
    # (col - 1) * L + row
end

