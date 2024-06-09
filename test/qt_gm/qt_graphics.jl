using Gen
using JSON
using PyCall
using Rooms: from_json, GridRoom, obstacle_tile,
    floor_tile, data
using GranularScenes

using ImageCore: colorview
using Colors: RGB

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

    res = (720, 480)

    room1 = load_room(1)
    room2 = load_room(2)

    display(room1)

    scene = TaichiScene(room1;
                        resolution = res,
                        window = false)
                        # window = true)

    img = observe_pixels(scene, room1, 0.1f0)
    img = render(scene, room1)
    ls = Gen.logpdf(observe_pixels, img, scene, room1, 0.1f0)

    println("logscore: $ls")


    # GranularScenes.window(scene, room1)

# end


    # for _ = 1:10
    #     GranularScenes.render(scene, room1)
    # end

    # _mu = GranularScenes.render(scene, room1)
    # py"print(type($_mu))"
    # mu = @pycall _mu.to_numpy()::Array
    # clamp!(mu, 0., 1.0)
    # reverse!(mu, dims = 1)

    # @show size(mu)
    # img = colorview(RGB, permutedims(mu, (3, 2, 1)))
    # reverse!(img, dims = 1)
    # display(img)


    # _mu = GranularScenes.render(scene, room2)
    # py"print(type($_mu))"
    # mu = @pycall _mu.to_numpy()::Array
    # clamp!(mu, 0., 1.0)
    # reverse!(mu, dims = 1)

    # @show size(mu)
    # img = colorview(RGB, permutedims(mu, (3, 2, 1)))
    # reverse!(img, dims = 1)
    # display(img)

    return nothing
end

mytest();
