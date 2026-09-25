export AStarPlanner, AStarState, PlanningModule,
    qt_topdown, best_path

@kwdef struct AStarPlanner <: PlanningProtocol
    ent::S2V
    ext::S2V
    nsamples::Int = 100
    obstacle_cost::Float64 = 1.0
    grad_tau::Float64 = 1.0
end


mutable struct AStarState <: MentalState{AStarPlanner}
    scores::Vector{Float64}
    paths::Vector{Vector{NodeId}}
end


function PlanningModule(protocol::AStarPlanner)
    scores = Vector{Float64}(undef, protocol.nsamples)
    fill!(scores, -Inf)
    paths = Vector{Vector{NodeId}}(undef, protocol.nsamples)
    state = AStarState(scores, paths)
    MentalModule{AStarPlanner}(protocol, state)
end


function step_module!(planning::MentalModule{<:AStarPlanner},
                      perception::MentalModule{<:AdaptiveMH})
    protocol, state = mparse(planning)

    _, vstate = mparse(perception)

    mass = logsumexp(vstate.weights)

    n = length(vstate.samples)
    for i = 1:n
        vtrace = vstate.samples[i]
        vweight = vstate.weights[i] - mass
        qt = get_retval(vtrace)
        score, path, dpi =
            qt_a_star(qt,
                      protocol.obstacle_cost,
                      protocol.ent, protocol.ext,
                      protocol.grad_tau)

        state.scores[i] = -score + vweight
        state.paths[i] = path
    end

    return nothing
end


function best_path(planning::MentalModule{<:AStarPlanner})
    protocol, state = mparse(planning)
    idx = argmax(state.scores)
    return idx, state.paths[idx]
end

using Colors: N0f8


"""
    qt_topdown(qt, d = finest_grid(qt); path = NodeId[], max_w = 1.0,
               threshold = 0.025)

Top-down projection of a quadtree on a d×d render grid (same linear-index
convention as `leaf_lin_idxs!`: li = (c1−1)*d + c2, x = column, y = row).

- Occupied leaves (weight above `threshold`) render as grayscale shading of
  their stored weight.
- Leaves along `path` (a `Vector{NodeId}`) are tinted by depth; start is
  green, goal is purple, intermediate leaves are red-tinted.
"""
function qt_topdown(qt,
                    path::Vector{NodeId} = NodeId[],
                    d = finest_grid(qt);
                    max_w::Float64 = 1.0, threshold::Float64 = 0.025)
    # Occupancy via the renderer's own fill routine (float buffer, d×d).
    occ = Matrix{Float32}(undef, d, d)
    write_obstacles!(occ, qt, d; threshold = Float32(threshold))

    img = Matrix{RGB{Float64}}(undef, d, d)
    max_w = max_w > 0 ? max_w : 1.0
    # occ[li] is column-major with li=(x-1)*d + y: occ[x, y]. Julia displays
    # the FIRST index vertically, and room_topdown shows the fast axis
    # (steps[1], the room row) vertically -- so display row = y (c2, fast),
    # column = x (c1, slow): img[y, x] = occ[x, y].
    for x in 1:d, y in 1:d
        v = clamp(occ[x, y] / max_w, 0.0, 1.0)
        img[y, x] = v > 0 ?
            RGB(1.0 - 0.85v, 1.0 - 0.85v, 1.0 - 0.85v) :
            RGB(1.0, 1.0, 1.0)
    end
    isempty(path) && return img

    tint = 0.45
    buf = Int64[]
    for (j, n) in enumerate(path)
        resize!(buf, d * d)
        k = leaf_lin_idxs!(buf, qt, n, d)   # render-cell linear indices
        c = j == 1               ? RGB(0.0, 0.6, 0.0) :
            j == lastindex(path) ? RGB(0.7, 0.0, 0.7) :
                                   RGB(0.9, 0.15, 0.1)
        for idx in 1:k
            li = buf[idx]
            # li = (c1-1)*d + c2 with Julia column-major indexing: the slow
            # index c1 is the display row (y), c2 the column (x).
            r0 = (li - 1) % d + 1   # fast index (y) -> display row
            c0 = (li - 1) ÷ d + 1   # slow index (x) -> display column
            r, g, b = red(img[r0, c0]), green(img[r0, c0]), blue(img[r0, c0])
            img[r0, c0] = RGB((1-tint)*r + tint*red(c),
                              (1-tint)*g + tint*green(c),
                              (1-tint)*b + tint*blue(c))
        end
    end
    img
end

# Bresenham on finest-grid cells, 1-based, y descending (node_to_idx order)
function _line_cells(p, q, L)
    x0, y0 = Tuple(p) .- 1
    x1, y1 = Tuple(q) .- 1
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = x0 < x1 ? 1 : -1
    sy = y0 < y1 ? 1 : -1
    err = dx + dy
    cells = Tuple{Int,Int}[]
    while true
        push!(cells, (x0 + 1, y0 + 1))
        (x0 == x1 && y0 == y1) && break
        e2 = 2err
        e2 >= dy && (err += dy; x0 += sx)
        e2 <= dx && (err += dx; y0 += sy)
    end
    cells
end

