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

dataset = "window-0.1/2025-02-05_vifdDO"

function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx).json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

function mytest()
    r = load_room(1)

    params = QuadTreeModel(r;
                           render_kwargs =
                               Dict(:resolution => (128,128)))
    
    # call works at t=0
    (trace, ll) = generate(qt_model, (0, params,))
    # display(get_submap(get_choices(trace), :trackers))
    # display(get_submap(get_choices(trace), :changes))
    @show ll
    # call works at t=1
    # (trace, ll) = generate(qt_model, (1, params,))
    # display(get_submap(get_choices(trace), :trackers))
    # display(get_choices(trace)[:changes => 1 => :change])
    # display(get_submap(get_choices(trace), :changes => 1 => :location))
    return nothing
end

mytest();
