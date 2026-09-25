export AStarPlanner, AStarState, PlanningModule,
    qt_topdown, best_path

@kwdef struct AStarPlanner <: PlanningProtocol
    ent::Int
    ext::Int
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

- Occupied leaves (weight > threshold) in grayscale, brightness 1 − w/max_w,
  coarser leaves shaded lighter so refinement structure is visible.
- `path`: consecutive NodeIds (e.g. the second return of `qt_a_star`).
  Each leaf is painted as its full footprint in red, the first green, the
  last magenta; overlaps later in the path win. Coarse leaves in the path
  are therefore visible as large red blocks, not points.
- `draw_centers = true`: thin blue line between consecutive leaf centers
  (via `center(n, qt)`, mapped with `node_to_idx`) showing the planned
  route under the blocky footprints.
"""
function qt_topdown(qt, path::Vector{NodeId} = NodeId[],
                    d::Int = finest_grid(qt);
                    max_w::Float64 = 1.0, threshold::Float64 = 0.025,
                    draw_centers::Bool = true)
    img = fill(RGB{N0f8}(1, 1, 1), d, d)
    buf = Vector{Int64}(undef, d * d)

    for n in qt.schema.leaves                       # finer leaves paint last
        w = get(qt.weight_map, n, 0.0)
        w > threshold || continue
        t = 1.0 - clamp(w / max_w, 0.0, 1.0)
        shade = 0.15 * (1 - (max_level(qt) - Int(n.depth)) /
                        max(1, max_level(qt) - 1))
        v = clamp(t + shade, 0.0, 1.0)
        k = leaf_lin_idxs!(buf, qt, n, d)
        for li in view(buf, 1:k)
            img[li] = RGB{N0f8}(v, v, v)
        end
    end

    # path footprints
    L = finest_grid(qt)
    f = d ÷ L
    for (j, n) in enumerate(path)
        haskey(qt.weight_map, n) || continue        # skip invalid ids
        k = leaf_lin_idxs!(buf, qt, n, d)
        c = j == 1                    ? RGB{N0f8}(0, 0.6, 0) :
            j == lastindex(path)      ? RGB{N0f8}(0.7, 0, 0.7) :
                                        RGB{N0f8}(0.9, 0.15, 0.1)
        for li in view(buf, 1:k)
            img[li] = c
        end
    end

    # polyline between consecutive leaf centers (under the footprints is
    # impossible after the fact, so it goes on top at half alpha-style mix)
    if draw_centers && length(path) > 1
        pts = map(path) do n
            x, y = node_xy(n)                                   # node-level coords
            sz  = finest_grid(qt) >> (max_level(qt) - Int(n.depth))
            (round(Int, (x + 0.5) * sz), round(Int, (y + 0.5) * sz))  # 1..L
        end
        for (a, b) in zip(pts[1:end-1], pts[2:end])
            for (x, y) in _line_cells(a, b, L)
                ci = clamp(x, 1, L); ri = clamp(y, 1, L)
                li = (ci - 1) * d + (ri - 1) * f + 1 + (f ÷ 2) * (d - 1) + 1
                # simpler: paint the f×f block at this finest cell
                bx = (ci - 1) * f + 1
                by = (ri - 1) * f + 1
                for jj in 0:(f-1), ii in 0:(f-1)
                    img[(bx + ii - 1) * d + (by + jj)] = RGB{N0f8}(0.2, 0.4, 0.9)
                end
            end
        end
    end
    return img
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

