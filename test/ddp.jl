using JSON
using Rooms
using PyCall
using FileIO
using GranularScenes
using GranularScenes: render, display_mat, display_img, process_ddp_input, process_taichi_array


cfg_path = "/project/scripts/nn/configs/og_decoder.yaml"
dataset = "window-0.1/2025-02-05_vifdDO"

function load_scene(path::String)
    local s
    open(path, "r") do f
        s = JSON.parse(f)
    end
    r = from_json(GridRoom, s)
    r = clear_wall(r)
    d = data(r)
    d[d .== obstacle_tile] .= floor_tile
    d[6:8, 8:10] .= obstacle_tile
    GridRoom(r, d)
end

function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

function main()

    template = clear_wall(GridRoom((16, 16), (16., 16.), [8], [252]))
    scene = 4
    println("Testing scene $(scene)")
    base_path = "/spaths/datasets/$(dataset)/scenes"
    room_path = joinpath(base_path, "$(scene).json")
    room = load_scene(room_path)
    display(room)
    renderer = TaichiScene(template; resolution = (128, 128))
    x = render(renderer, room)
    array = @pycall x.to_numpy()::Array
    save_img_array(array, "/spaths/tests/ddp-img-in.png")
    ddp_params = DataDrivenState(;config_path = cfg_path,
                                 var = 0.01)

    img = process_ddp_input(x, ddp_params.device)
    x = @pycall ddp_params.nn.forward(img)::PyObject
    state = @pycall x.detach().squeeze(0).cpu().numpy()::Matrix{Float64}
    println("Data-driven state")
    display_mat(state)
end;


main();

#TODO: patch `process_ddp_input` to initialize with bordered room
# The rooms make in the window dataset do not have their borders
# explicitly saved, thus simply loading them does not work
# without using a template to initialize `TaichiScene`
