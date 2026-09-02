using Gen
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
                               Dict(:image_res => (512,512),
                                    :use_cuda => false),
                           pixel_var = 0.01)

    # cm = choicemap()
    # cm[:trackers => (1, Val(:production)) => :produce] = true
    # for i = 1 : 4
    #     cm[:trackers => (i + 1, Val(:production)) => :produce] = i == 2
    # end

    # generate(qt_model, (params,))
    # for _ = 1:10
    #     generate(qt_model, (params,))
    # end
    @time (trace, ll) = generate(qt_model, (params,))
    # display(@benchmark generate($qt_model, ($params,), $cm) seconds=10 )
    # Profile.clear()
    # @profilehtml (trace, ll) = generate(qt_model, (params,), cm)
    # display(get_submap(get_choices(trace), :trackers))
    display(trace[:depth][:, :, 1])
    return nothing
end

mytest();
