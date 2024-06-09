using PyCall
using JSON
using FileIO
using Rooms
using GranularScenes
# using Colors: RGB
# using ImageCore: permutedims, colorview
using FunctionalCollections: PersistentVector

IMG_RES = (128, 128)

function occupancy_position(r::GridRoom)::Matrix{Float64}
    grid = zeros(steps(r))
    grid[data(r) .== obstacle_tile] .= 1.0
    grid
end

function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

function build(r::GridRoom;
               max_f::Int64 = 11,
               max_size::Int64 = 5,
               pct_open::Float64 = 0.3,
               side_buffer::Int64 = 0,
               factor = 2)

    dims = steps(r)
    # prevent furniture generated in either:
    # -1 out of sight
    # -2 blocking entrance exit
    # -3 hard to detect spaces next to walls
    weights = Matrix{Bool}(zeros(dims))
    # ensures that there is no furniture near the observer
    start_x = Int64(ceil(last(dims) * pct_open))
    stop_x = last(dims) - 2 # nor blocking the exit
    # buffer along sides
    start_y = side_buffer + 1
    stop_y = first(dims) - side_buffer
    weights[start_y:stop_y, start_x:stop_x] .= 1.0
    vmap = PersistentVector(vec(weights))

    # sample obstacles
    # TODO: rename "furniture*" -> "obstacles*"
    with_furn = furniture_gm(r, vmap, max_f, max_size)
    result = expand(with_furn, factor)
    clear_wall(result)
end

function save_trial(dpath::String, i::Int64, r::GridRoom,
                    img, og)
    out = "$(dpath)/$(i)"
    isdir(out) || mkdir(out)

    open("$(out)/room.json", "w") do f
        rj = r |> json
        write(f, rj)
    end
    open("$(out)/scene.json", "w") do f
        r2 = translate(r, Int64[]; cubes = false)
        r2j = r2 |> json
        write(f, r2j)
    end
    save_img_array(img, "$(out)/render.png")
    # occupancy grid saved as grayscale image
    save("$(out)/og.png", og)
    return nothing
end

function main()
    # Parameters
    # name = "ccn_2023_ddp_train_11f_32x32"
    # n = 5000
    name = "ccn_2023_ddp_test_11f_32x32"
    n = 16

    hn = Int(n // 2)
    room_dims = (16., 16.)
    room_bins = (16, 16)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_bins)
    doors = inds[door_rows, room_bins[2]]

    # empty rooms with doors
    templates = Vector{GridRoom}(undef, length(doors))
    for i = 1:length(templates)
        r = GridRoom(room_bins, room_dims, entrance, [doors[i]])
        templates[i] = clear_wall(r)
    end

    # will store summary of generated rooms here
    m = Dict(
        :n => n,
        :templates => templates,
        :og_shape => (32, 32),
        :img_res => IMG_RES
    )
    out = "/spaths/datasets/$(name)"
    isdir(out) || mkdir(out)

    template = templates[1]
    ti_scene = TaichiScene(expand(template, 2);
                               resolution = IMG_RES)
    for i = 1:hn
        r = build(template)
        occ = occupancy_position(r)
        # select mitsuba scene
        @time _mu = GranularScenes.render(ti_scene, r)
        img = @pycall _mu.to_numpy()::Array
        save_trial(out, i, r, img, occ)
    end

    template = templates[2]
    ti_scene = TaichiScene(expand(template, 2);
                               resolution = IMG_RES)
    for i = hn:n
        r = build(template)
        occ = occupancy_position(r)
        # select mitsuba scene
        @time _mu = GranularScenes.render(ti_scene, r)
        img = @pycall _mu.to_numpy()::Array
        save_trial(out, i, r, img, occ)
    end
    
    m[:img_mu] = zeros(3)
    m[:img_sd] = ones(3)

    open("$(out)_manifest.json", "w") do f
        write(f, m |> json)
    end
    return nothing
end

main();
