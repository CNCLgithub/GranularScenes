using CSV
using JSON
using Dates
using DataFrames
using StaticArrays
using SparseArrays
using LinearAlgebra
using Gen
using Graphs
using Rooms
using GranularScenes
using FunctionalCollections
using WaveFunctionCollapse
import WaveFunctionCollapse as WFC

function viz_room(room)
    GranularScenes.display_mat(Float64.(data(room) .== floor_tile))
    return nothing
end

function room_from_wave(wave, bounds, start, dest)
    steps = size(wave)
    d = Matrix{Tile}(undef, steps)
    fill!(d, floor_tile)
    # set entrances and exits
    # these are technically floors but are along the border
    d[start] = floor_tile
    d[dest] = floor_tile
    @inbounds for i = eachindex(wave)
        wave[i] && continue
        d[i] = obstacle_tile
    end
    g = Rooms.init_pathgraph(GridRoom, d)
    GridRoom(steps, (32.0, 32.0), [start], [dest], g, d)
end

const HT4D = SMatrix{4, 4, Bool}

ht4d_elems = HT4D[
    HT4D([0 0 0 0;
          0 0 0 0;
          0 0 0 0;
          0 0 0 0]), # 1
    HT4D([0 1 1 0;
          0 1 1 0;
          0 1 1 0;
          0 1 1 0]), # 2
    HT4D([0 0 0 0;
          1 1 1 1;
          1 1 1 1;
          0 0 0 0]), # 3
    HT4D([0 1 1 0;
          1 1 1 1;
          1 1 1 1;
          0 0 0 0]), # 4
    HT4D([0 0 0 0;
          1 1 1 1;
          1 1 1 1;
          0 1 1 0]), # 5
    HT4D([0 1 1 0;
          1 1 1 0;
          1 1 1 0;
          0 1 1 0]), # 6
    HT4D([0 1 1 0;
          0 1 1 1;
          0 1 1 1;
          0 1 1 0]), # 7
    HT4D([0 1 1 0;
          1 1 1 0;
          1 1 1 0;
          0 0 0 0]), # 8
    HT4D([0 1 1 0;
          0 1 1 1;
          0 1 1 1;
          0 0 0 0]), # 9
    HT4D([0 0 0 0;
          0 1 1 1;
          0 1 1 1;
          0 1 1 0]), # 10
    HT4D([0 0 0 0;
          1 1 1 0;
          1 1 1 0;
          0 1 1 0]), # 11
    HT4D([0 1 1 0;
          1 1 1 1;
          1 1 1 1;
          0 1 1 0]), # 12
    HT4D([0 0 0 0;
          0 1 1 1;
          0 1 1 1;
          0 0 0 0]), # 13
    HT4D([0 0 0 0;
          1 1 1 0;
          1 1 1 0;
          0 0 0 0]), # 14
    HT4D([0 0 0 0;
          0 1 1 0;
          0 1 1 0;
          0 1 1 0]), # 15
    HT4D([0 1 1 0;
          0 1 1 0;
          0 1 1 0;
          0 0 0 0]), # 16
    HT4D([1 1 1 1;
          1 1 1 1;
          1 1 1 1;
          1 1 1 1]), # 17
    HT4D([0 0 0 0;
          1 1 0 0;
          1 1 0 0;
          0 0 0 0]), # 18
]

function dist(x::CartesianIndex{2}, y::CartesianIndex{2})
    sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
end

function add_segment!(m::Matrix{Int64}, pidx::Int, cdir, pdir)
    m[pidx] == 0 || return nothing
    if m[pidx] != 0
        fix_template!(m, start, )
    end
    if cdir == WFC.Above # going up
        if pdir == WFC.Above
            m[pidx] = 2
        elseif pdir == WFC.Left
            m[pidx] = 9
        elseif pdir == WFC.Right
            m[pidx] = 8
        end
    elseif cdir == WFC.Below
        if pdir == WFC.Below
            m[pidx] = 2
        elseif pdir == WFC.Left
            m[pidx] = 10
        elseif pdir == WFC.Right
            m[pidx] = 11
        end
    elseif cdir == WFC.Left
        if pdir == WFC.Below
            m[pidx] = 8
        elseif pdir == WFC.Above
            m[pidx] = 9
        elseif pdir == WFC.Left
            m[pidx] = 3
        end
    else
        if pdir == WFC.Below
            m[pidx] = 9
        elseif pdir == WFC.Above
            m[pidx] = 10
        elseif pdir == WFC.Right
            m[pidx] = 3
        end
    end
    return nothing
end

function sample_path(n::Int, start::Int, dest::Int, temp::Float64 = 1)
    m = zeros(Int64, (n, n))
    # m[:, 1] .= 1 # don't set near entrance
    m[start] = 3
    path = Int64[start]
    sample_path!(m, path, start + n, dest, temp)
    return (m, path)
end

function sample_path!(m::Matrix{Int64}, path::Vector{Int64},
                      start::Int, dest::Int, temp::Float64,
                      dir_prev = WFC.Right)
    cis = CartesianIndices(m)
    dest_ci = cis[dest]
    n = size(m, 1)
    gs = GridSpace(n * n, (n, n))
    push!(path, start)
    current = prev = start
    dir = dir_prev
    temp = temp + 0.0
    while current != dest
        ns = WFC.neighbors(gs, current)
        nns = length(ns)
        cur_dis = dist(cis[current], dest_ci)
        ws = fill(-Inf, nns)
        for i = 1:nns
            n, _ = ns[i]
            m[n] == 0 || continue # skip if set
            d = dist(cis[n], dest_ci)
            d < cur_dis || continue
            ws[i] = -d
        end
        _ws = WFC.softmax(ws, temp)
        if all(isinf, ws)
            display(m)
            error()
        end
        selected = isapprox(temp, 0.0) ? argmax(ws) : categorical(softmax(ws, temp))
        prev = current
        dir_prev = dir
        current, dir = ns[selected]
        add_segment!(m, prev, dir, dir_prev)
        push!(path, current)
        temp *= 0.95
    end
    add_segment!(m, current, WFC.Right, dir)
    return nothing
end

function compile_path(n::Int, start::Int, steps::Vector{WFC.GridRelation})
    m = zeros(Int64, (n, n))
    path = Int64[]
    compile_path!(m, path, start, steps)
    return (m, path)
end

function compile_path!(m::Matrix{Int64}, path::Vector{Int64},
                       start::Int, steps::Vector{WFC.GridRelation},
                       dir_prev=WFC.Right)
    cis = CartesianIndices(m)
    n = size(m, 1)
    gs = GridSpace(n * n, (n, n))
    push!(path, start)
    current = prev = start
    dir = dir_prev
    for step in steps
        dir_prev = dir
        dir = step
        prev = current
        current = WFC.move(Val(step), gs, current)
        add_segment!(m, prev, step, dir_prev)
        push!(path, current)
    end
    add_segment!(m, current, WFC.Right, dir)
    return nothing
end

function sample_pair(n::Int,
                     start::Int,
                     left_door::Int,
                     right_door::Int,
                     sp::GridSpace,
                     ts::TileSet,
                     path_temp::Float64 = 2.0,
                     fix_steps::Int64 = 10,
                     extra_pieces::Int64 = 13,
                     piece_size::Int64 = 5)

    # # hard-coded path to left doorA
    # left_steps = [
    #     WFC.Right,
    #     WFC.Right,
    #     WFC.Above,
    #     WFC.Above,
    #     WFC.Right,
    #     WFC.Right,
    #     WFC.Right,
    #     WFC.Right,
    # ]
    # template, left_path = compile_path(n, start, left_steps)
    # display(template)

    # right_path = deepcopy(left_path[1:3])
    # start_from = right_path[end]
    # right_steps = [
    #     WFC.Below,
    #     WFC.Right,
    #     WFC.Above,
    #     WFC.Right,
    #     WFC.Right,
    #     WFC.Below,
    #     WFC.Below,
    #     WFC.Right,
    # ]

    # compile_path!(template, right_path, start_from, right_steps)
    # display(template)

    # if left_steps[3] == WFC.Above && right_steps[1] == WFC.Below
    #     template[left_path[3]] = 6
    # end

    # to_block = right_path[8]
    # hard-coded path to left doorA
    left_steps = [
        WFC.Right,
        WFC.Right,
        WFC.Above,
        WFC.Above,
        WFC.Right,
        WFC.Right,
        WFC.Right,
        WFC.Right,
    ]
    template, left_path = compile_path(n, start, left_steps)
    display(template)

    right_path = deepcopy(left_path[1:3])
    start_from = right_path[end]
    right_steps = [
        WFC.Right,
        WFC.Right,
        WFC.Right,
        WFC.Below,
        WFC.Below,
        WFC.Right,
    ]

    compile_path!(template, right_path, start_from, right_steps)
    display(template)

    template[left_path[3]] = 4

    to_block = right_path[6]

    # Apply WFC to populate the rest of the room
    wave = WaveState(template, sp, ts)
    collapse!(wave, sp, ts)
    (wave, left_path, right_path, to_block)
end

function eval_pair(wave::WaveState,
                   left_path::Vector{Int64},
                   right_path::Vector{Int64},
                   cut::Int)
    nl = length(left_path)
    nr = length(right_path)
    left_tiles = count(==(3), wave.wave[right_path[1:cut+2]])
    # Not a valid sample if:
    #     (1) Missing path for either door
    #     (2) Left and Right paths overlap too much
    #     (3) One path is much longer than the other
    (nl == 0 ||
        nr == 0 ||
        nr - cut < 4 ||
        left_tiles > 2 ||
        abs(nl - nr) > 3) &&
        return 0

    # Where to place obstacle that blocks the right path
    htile_idx = rand(right_path[(cut + 2):(end-2)])
end



function main()
    name = "path_block_maze"
    dataset_base = "/spaths/datasets/$(name)"
    isdir(dataset_base) || mkdir(dataset_base)
    dataset_out = mktempdir(dataset_base;
                            prefix = "$(Dates.today())_",
                            cleanup = false)
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)

    # Parameters
    n = 7
    room_steps = (n, n)
    room_bounds = (32., 32.)
    start = 4
    entrance = [start]
    door_rows = [2, 6]
    inds = LinearIndices(room_steps)
    doors = (left_door, right_door) =
        inds[door_rows, room_steps[2]]


    sp = GridSpace(prod(room_steps), room_steps)
    ts = TileSet(ht4d_elems, sp)

    # number of trials
    ntrials = 2

    # will store summary of generated rooms here
    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   tidx = Int64[])

    i = 1 # scene id
    c = 0 # number of attempts;
    while i <= ntrials && c < 1000 * ntrials
        # generate a room pair
        (wave, lpath, rpath, tile) = sample_pair(room_steps[1], entrance[1],
                                                 doors[1], doors[2],
                                                 sp, ts)
        # tile = eval_pair(wave, lpath, rpath, cut)
        c += 1
        # no valid pair generated, try again or finish
        tile == 0 && continue
        println("accepted pair!")
        # create rooms
        expanded = WFC.expand(wave, sp, ts)
        right = room_from_wave(expanded, (n, n), start * 4 - 1,
                               # HACK: lol no idea have this works
                               right_door * 4 - 1 + (n * 4 * (n * 4 - n)))
        left = room_from_wave(expanded, (n, n), start * 4 - 1,
                              right_door * 4 - 1  + (n * 4 * (n * 4 - n)))
        blocked_wave = deepcopy(wave)
        blocked_wave.wave[tile] = 14 # set to obstacles
        blocked_exp = WFC.expand(blocked_wave, sp, ts)
        blocked_left = room_from_wave(blocked_exp, (n, n), start * 4 - 1,
                                      # HACK: lol no idea have this works
                                      left_door * 4 - 1 + (n * 4 * (n * 4 - n)))
        blocked_right = room_from_wave(blocked_exp, (n, n), start * 4 - 1,
                                       # HACK: lol no idea have this works
                                       right_door * 4 - 1 + (n * 4 * (n * 4 - n)))

        # viz_room(right)
        viz_room(blocked_right)
        # save
        toflip = (i-1) % 2
        push!(df, [i, toflip, tile])
        open("$(scenes_out)/$(i)_1.json", "w") do f
            write(f, left |> json)
        end
        open("$(scenes_out)/$(i)_2.json", "w") do f
            write(f, right |> json)
        end
        open("$(scenes_out)/$(i)_1_blocked.json", "w") do f
            write(f, blocked_left |> json)
        end
        open("$(scenes_out)/$(i)_2_blocked.json", "w") do f
            write(f, blocked_right |> json)
        end
        print("scene $(i)/$(n)\r")
        i += 1
    end
    @show df
    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)
    println("Created $(i-1) scenes in $(c) attempts")
    println("Finished writing dataset to $(dataset_out)")
    return nothing
end

main();
