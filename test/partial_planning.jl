using Gen
using JSON
using Rooms
using Colors
using GranularScenes
using GranularScenes: integrate_update, project_qt, draw_mat, max_leaves,
    leaf_from_idx, node_to_idx, qt_a_star

dataset = "path_block_2024-03-14"
function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

function load_scene(idx::Int)
    path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(path, "$(idx)_1.json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    # base = expand(from_json(GridRoom, base_s), 2)
    base = from_json(GridRoom, base_s)
    clear_wall(base)
end

function ex_path(qt, path)
    n = max_leaves(qt)
    leaves = qt.leaves
    m = Matrix{Bool}(undef, n, n)
    fill!(m, false)
    for i = path
        node = leaf_from_idx(qt, i).node
        for idx = node_to_idx(node, n)
            m[idx] = true
        end
    end
    m
end

function mytest()

    room = load_scene(2)
    params = QuadTreeModel(room; obs_cost = 1.0)
    cm = choicemap()
    cm[:trackers => (1, Val(:production)) => :produce] = true
    cm[:trackers => (2, Val(:production)) => :produce] = true
    cm[:trackers => (3, Val(:production)) => :produce] = true
    cm[:trackers => (4, Val(:production)) => :produce] = false
    cm[:trackers => (5, Val(:production)) => :produce] = false
    qt_tr, _ = generate(qt_model, (params,), cm)
    qt = get_retval(qt_tr)
    # display(get_choices(qt_tr)[:trackers => (2, Val(:aggregation)) => :mu])

    @time (score, path) = quad_tree_path(qt_tr)
    @show score
    display(plan)

    # tr = plan(qt, params.obs_cost, first(params.entrance),
    #           first(params.exit), 5)
    # # display(get_choices(tr))
    # @show get_retval(tr)
    # path = get_plan(tr)
    # @show path

    geo = draw_mat(project_qt(qt), true, colorant"black", colorant"blue")
    pm = Matrix{Float64}(ex_path(qt, path))
    pm = draw_mat(pm, true, colorant"black", colorant"green")
    display(reduce(hcat, [geo, pm]))

    # qt_tr_prime, _ = rw_move(qt_tr, 2)
    # display(get_choices(qt_tr_prime)[:trackers => (2, Val(:aggregation)) => :mu])
    # qt_prime = get_retval(qt_tr_prime)
    # ratio = integrate_update(tr, qt_prime)
    # @show ratio
    # geo = draw_mat(project_qt(qt_prime), true, colorant"black", colorant"blue")
    # pm = Matrix{Float64}(ex_path(qt_prime, path))
    # pm = draw_mat(pm, true, colorant"black", colorant"green")
    # display(reduce(hcat, [geo, pm]))

    return nothing
end

mytest();
