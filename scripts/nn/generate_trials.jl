using JSON
using Rooms
using PyCall
using FileIO
using ProgressMeter
using GranularScenes


include("/project/scripts/stimuli/room_process.jl")

IMG_RES = (128, 128)

function occupancy_position(r::GridRoom)::Matrix{Float64}
    grid = zeros(Rooms.steps(r))
    grid[data(r) .== obstacle_tile] .= 1.0
    grid
end

function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

#################################################################################
# Trial generation
#################################################################################

function sample_room!(x, d1, d2)
    reset_chasis!(x)
    # clear front of room
    foreach(c -> clear_col!(x, c), 1:4)
    # row-14 col-6
    right_corner = 5 * 16 + 14
    clear_region!(x, right_corner, 2)
    # row-2 col-6
    left_corner = 5 * 16 + 2
    clear_region!(x, left_corner, 2)
    # clear near doors
    clear_region!(x, d1, 2)
    clear_region!(x, d2, 2)
    clear_region!(x, d1-2, 1)
    clear_region!(x, d2+2, 1)
    sample_room!(x)
    room_from_process(x)
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
    # name = "ddp_train_11f_32x32"
    # n = 10000
    name = "ddp_test_11f_32x32"
    n = 16

    hn = Int(n // 2)
    room_dims = (16., 16.)
    room_bins = (16, 16)
    start = 8
    entrance = [start]
    doors = [252, 244]

    # empty rooms with doors
    templates = Vector{GridRoom}(undef, length(doors))
    for i = 1:length(templates)
        r = GridRoom(room_bins, room_dims, entrance, [doors[i]])
        templates[i] = clear_wall(r)
    end
    x = RoomProcess(room_bins, start, doors[1])

    # will store summary of generated rooms here
    m = Dict(
        :n => n,
        :templates => templates,
        :og_shape => room_bins,
        :img_res => IMG_RES
    )
    out = "/spaths/datasets/$(name)"
    isdir(out) || mkdir(out)

    template = templates[1]
    ti_scene = TaichiScene(template;
                           resolution = IMG_RES)
    @showprogress desc="Sampling door 1" for i = 1:hn
        r = sample_room!(x, doors[1], doors[2])
        occ = occupancy_position(r)
        # select mitsuba scene
        _mu = GranularScenes.render(ti_scene, r)
        img = @pycall _mu.to_numpy()::Array
        save_trial(out, i, r, img, occ)
    end

    x = RoomProcess(room_bins, start, doors[2])
    template = templates[2]
    ti_scene = TaichiScene(template;
                           resolution = IMG_RES)
    @showprogress desc="Sampling door 2" for i = hn:n
        r = sample_room!(x, doors[1], doors[2])
        occ = occupancy_position(r)
        # select mitsuba scene
        _mu = GranularScenes.render(ti_scene, r)
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
