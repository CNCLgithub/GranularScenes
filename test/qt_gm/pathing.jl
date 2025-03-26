using Gen
using JSON
using Rooms
using Colors
using GranularScenes
# using Profile
# using BenchmarkTools
# using StatProfilerHTML

using GranularScenes:
    quad_tree_path,
    draw_mat,
    traverse_qt,
    project_qt,
    ex_path,
    delta_pi,
    node,
    tree_idx,
    max_leaves,
    leaf_from_idx,
    node_to_idx,
    qt_from_state

using Random
# Random.seed!(1101)

dataset = "path_block_2024-03-14"

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
end
function draw_gradients(tr::Gen.Trace, grads::Dict{Int64, Float64})
    qt = get_retval(tr)
    n = max_leaves(qt)
    leaves = qt.leaves
    m = zeros((n, n))
    for (i, v) = grads
        node = leaf_from_idx(qt, i).node
        for idx = node_to_idx(node, n)
            m[idx] = v
        end
    end
    m .*= 1.0 / maximum(m)
    display(draw_mat(m, true, colorant"black", colorant"red"))
end
function mytest()

    scene = 2
    door = 2
    base_path = "/spaths/datasets/$(dataset)/scenes"
    base_p = joinpath(base_path, "$(scene)_$(door).json")
    room = load_base_scene(base_p)
    display(room)

    state = zeros(Rooms.steps(room))
    state[data(room) .== obstacle_tile] .= 1.0

    model_params = QuadTreeModel(room)

    cm, nnodes = qt_from_state(0.12, state, model_params)
    println("Leaves: $(nnodes)")
    tr, ls = generate(qt_model, (0, model_params,), cm)

    @time plan = cost, path, grads = quad_tree_path(tr)
    @show path

    qt = get_retval(tr)
    geo = draw_mat(project_qt(qt), true, colorant"black", colorant"blue")
    path = Matrix{Float64}(ex_path(tr))
    pth = draw_mat(path, true, colorant"black", colorant"green")
    display(reduce(hcat, [geo, pth]))
    draw_gradients(tr, grads)

    # tprime, _ = rw_move(tr, 13)
    # plan = cost, path, grads = quad_tree_path(tprime)
    # draw_gradients(tprime, grads)

    # for lf = qt.leaves
    #     idx = tree_idx(node(lf))
    #     tprime, _ = rw_move(tr, idx)
    #     @show idx
    #     @show delta_pi(plan, tprime)
    # end
    return nothing
end

mytest();
