using CSV
                     chisel_temp::Float64 = -1.0,
                     fix_steps::Int64 = 10,
                     extra_pieces::Int64 = 8,
                     piece_size::Int64 = 3)

    # sample some obstacles in the middle section of the roo
    oweights = gen_obstacle_weights(right_room, Int64[])
    right_room = furniture_gm(right_room, oweights,
                              extra_pieces, piece_size)
    left_room  = add(left_room, right_room)

    right_path = dpath(right_room)
    left_path = dpath(left_room)

    (left_room, left_path, right_room, right_path)
end

function eval_pair(left_door::GridRoom,
                   left_path::Vector{Int64},
                   right_door::GridRoom,
                   right_path::Vector{Int64};
                   max_lr_intersect = 8,
                   max_lr_length_diff = 3,
                   min_block_diff = 3
                   )

    tile = 0
    # Not a valid sample
    ((isempty(left_path) || isempty(right_path)) ||
        length(intersect(left_path, right_path)) > max_lr_intersect ||
        abs(length(right_path) - length(left_path)) > max_lr_length_diff) &&
        return tile

    # Where to place obstacle that blocks the right path
    nl = length(right_path)
    # avoid blocking entrance or exit
    trng = 3:(nl-5)
    g = pathgraph(right_door)
    gt = deepcopy(g)
    ens = entrance(right_door)
    ext = exits(right_door)
    # look for first tile that blocks the path

    @inbounds for i = trng
        tid = right_path[i]
        # block tile
        right_temp = add(right_door, Set{Int64}(tid))
        # does it block path ?
        new_path = dpath(right_temp)
        if isempty(new_path) || length(new_path) > nl + min_block_diff
            left_temp = add(left_door, Set{Int64}(tid))
            new_left_path = dpath(left_temp)
            if new_left_path == left_path
                tile = tid
                break
            end
        end
    end

    return tile
end



function main()
    name = "path_block"
    dataset_base = "/spaths/datasets/$(name)"
    isdir(dataset_base) || mkdir(dataset_base)
    dataset_out = mktempdir(dataset_base;
                            prefix = "$(Dates.today())_",
                            cleanup = false)
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)


    # Parameters
    room_steps = (16, 16)
    room_bounds = (32., 32.)
    entrance = [8, 9]
    door_rows = [5, 12]
    inds = LinearIndices(room_steps)
    doors = inds[door_rows, room_steps[2]]

    # number of trials
    n = 6

    # empty room with doors
    left_cond = empty_room(room_steps, room_bounds, entrance, [doors[1]])
    right_cond = empty_room(room_steps, room_bounds, entrance, [doors[2]])

    # will store summary of generated rooms here
    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   tidx = Int64[])

    max_steps = 1000
    i = 1 # scene id
    c = 0 # number of attempts;
    while i <= n && c < max_steps * n
        # generate a room pair
        (left, lpath, right, rpath) = sample_pair(left_cond, right_cond)
        tile = eval_pair(left, lpath, right, rpath)
        # viz_room(right, rpath)
        # viz_room(left, lpath)
        # no valid pair generated, try again or finish
        c += 1
        tile == 0 && continue

        println("accepted pair!")
        viz_room(right, rpath)
        blocked_room = add(right, Set{Int64}(tile))
        viz_room(blocked_room, dpath(blocked_room))
        viz_room(left, lpath)

        # save
        toflip = (i-1) % 2
        push!(df, [i, toflip, tile])
        open("$(scenes_out)/$(i)_1.json", "w") do f
            write(f, left |> json)
        end
        open("$(scenes_out)/$(i)_2.json", "w") do f
            write(f, right |> json)
        end

        print("scene $(i)/$(n)\r")
        i += 1
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    return nothing
end

main();
