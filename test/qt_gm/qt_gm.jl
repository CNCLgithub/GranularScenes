using Gen
using PyCall
using JSON
using Rooms: from_json, GridRoom, obstacle_tile,
    floor_tile, data
using GranularScenes
using ImageCore: colorview
using Colors: RGB
using FileIO: save
# using ProfileView
# using Profile
# using BenchmarkTools
# using StatProfilerHTML

dataset = "ccn_2023_exp"

function load_room(idx::Int)

    base_path = "test/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx)_1.json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
    d = data(base)
    d[:, 1:2] .= floor_tile
    # d[d .!= obstacle_tile] .= floor_tile
    GridRoom(base, d)
end

function mytest()
    r = load_room(1)

    params = QuadTreeModel(r;
                           render_kwargs =
                               Dict(:resolution => (512,512)))

    # cm = choicemap()
    # cm[:trackers => (1, Val(:production)) => :produce] = true
    # for i = 1 : 4
    #     cm[:trackers => (i + 1, Val(:production)) => :produce] = i == 2
    # end

    generate(qt_model, (params,))
    for _ = 1:10
        generate(qt_model, (params,))
    end
    (trace, ll) = generate(qt_model, (params,))
    # display(@benchmark generate($qt_model, ($params,), $cm) seconds=10 )
    # Profile.clear()
    # @profilehtml (trace, ll) = generate(qt_model, (params,), cm)
    # display(get_submap(get_choices(trace), :trackers))
    qt = get_retval(trace)
    _mu = render(params.renderer, qt)
    mu = @pycall _mu.to_numpy()::Array
    clamp!(mu, 0., 1.0)
    reverse!(mu, dims = 1)
    reverse!(mu, dims = 2)
    img = colorview(RGB, permutedims(mu, (3, 2, 1)))
    # img = channelview(st.img_mu)
    # save("/spaths/tests/qt_gm.png", img)
    @show ll
    display(img)
    return nothing
end

mytest();
