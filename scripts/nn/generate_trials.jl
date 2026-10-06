using Rooms
using StaticArrays
using ProgressMeter
using GranularScenes


include("/project/scripts/stimuli/room_process.jl")

IMG_RES = (256, 256)
IMG_VAR = 0.001f0

# CAMERA SETTINGS
cam_height = 50.0f0
cam_pitch  = -12.0f0
cam_fov    = 0.56f0
d = 16
mid = d ÷ 2
room_half = d ÷ 2 - 4
max_room_height = d ÷ 2
obs_h = 5
floor_y = 2
floor_world_y = Float32(floor_y - mid)
# --- camera scaled to the grid --------------------------------------
# cam_height slider (5..60) is world units; must stay < d÷2 (bbox top).
cam_h       = min(cam_height, Float32(max_room_height - 2))
cam_world_y = floor_world_y + cam_h
# cam_world_z = Float32(d-2)
cam_world_z = Float32(-(d-2)) 
cam_world_x = 0.0f0

pitch_rad   = deg2rad(Float32(cam_pitch))
# target_dist = Float32(-room_half * 2)
target_dist = Float32(room_half * 2)
target_x    = 0.0f0
target_y    = cam_world_y + target_dist * tan(pitch_rad)
target_z    = cam_world_z + target_dist
cam_pos = SVector{3,Float32}(cam_world_x, cam_world_y, cam_world_z)
look_at = SVector{3,Float32}(target_x, target_y, target_z)

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
    name = "ddp_train_11f_32x32"
    n = 10000
    # name = "ddp_test_11f_32x32"
    # n = 16

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

    template = templates[1]
    renderer = QuadTreeRenderer(
        ;
        image_res = IMG_RES,
        use_cuda = true,
        wall_mode = true,
        grid_res = d,
        obstacle_height = 5,
        camera_pos = cam_pos,
        look_at = look_at,
        fov = cam_fov,
    )

    out = "/spaths/datasets/$(name).hdf5"
    writer = DDPSDatasetWriter(out, n)
    
    @showprogress desc="Sampling door 1" for i = 1:hn
        r = sample_room!(x, doors[1], doors[2])
        occ = occupancy_position(r)
        depth = qt_observe(renderer, r, IMG_VAR)
        write_trial!(writer, depth, occ)
    end

    x = RoomProcess(room_bins, start, doors[2])
    template = templates[2]
    @showprogress desc="Sampling door 2" for i = (hn+1):n
        r = sample_room!(x, doors[1], doors[2])
        occ = occupancy_position(r)
        # select mitsuba scene
        depth = qt_observe(renderer, r, IMG_VAR)
        write_trial!(writer, depth, occ)
    end

    close(writer)
    
    return nothing
end

main();
