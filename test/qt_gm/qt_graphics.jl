using Gen
using JSON
using PyCall
using Rooms: from_json, GridRoom, obstacle_tile,
    floor_tile, data
using GranularScenes

using ImageCore: colorview
using Colors: RGB

dataset = "path_block_2024-03-14"

function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
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

function test_rv()
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

end;

function test_window()
    res = (720, 480)

    room1 = load_room(1)

    display(room1)

    scene = TaichiScene(room1;
                        resolution = res,
                        window = true)

    GranularScenes.window(scene, room1)
end

function test_render()

    res = (256, 256)

    room1 = load_room(1)
    room2 = load_room(2)

    display(room1)

    scene = TaichiScene(room1;
                        resolution = res,
                        window = false)


    println("Scene 1")
    # GranularScenes.render(scene, room1)
    img = GranularScenes.render(scene, room1)
    GranularScenes.save_img_array(img, "/spaths/tests/scene1.png")

    println("Scene 2")
    img = GranularScenes.render(scene, room2)
    GranularScenes.save_img_array(img, "/spaths/tests/scene2.png")

    return nothing
end

test_render();
