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

function empty_room(steps, bounds, entrance, exits)
    r = GridRoom(steps, bounds, entrance, exits)
    m = data(r)
    m[minimum(entrance) - 1 : maximum(entrance) + 1] .= floor_tile
    GridRoom(r, m)
end

const HT3D = SMatrix{3, 3, Bool}

ht3d_elems = HT3D[
    HT3D([0 0 0;
          0 0 0;
          0 0 0]), # 1
    HT3D([0 1 0;
          0 1 0;
          0 1 0]), # 2
    HT3D([0 0 0;
          1 1 1;
          0 0 0]), # 3
    HT3D([0 1 0;
          1 1 1;
          0 0 0]), # 4
    HT3D([0 0 0;
          1 1 1;
          0 1 0]), # 5
    HT3D([0 1 0;
          1 1 0;
          0 1 0]), # 6
    HT3D([0 1 0;
          0 1 1;
          0 1 0]), # 7
    HT3D([0 1 0;
          1 1 0;
          0 0 0]), # 8
    HT3D([0 1 0;
          0 1 1;
          0 0 0]), # 9
    HT3D([0 0 0;
          0 1 1;
          0 1 0]), # 10
    HT3D([0 0 0;
          1 1 0;
          0 1 0]), # 11
    HT3D([0 1 0;
          1 1 1;
          0 1 0]), # 12
    HT3D([0 0 0;
          0 1 1;
          0 0 0]), # 13
    HT3D([0 0 0;
          1 1 0;
          0 0 0]), # 14
    HT3D([0 0 0;
          0 1 0;
          0 1 0]), # 15
    HT3D([0 1 0;
          0 1 0;
          0 0 0]), # 16
]

function dist(x::CartesianIndex{2}, y::CartesianIndex{2})
    sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)
end

function add_segment!(m::Matrix{Int64}, pidx::Int, cdir, pdir)
    m[pidx] == 0 || return nothing
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
    m[:, 1] .= 1 # don't set near entrance
    m[start] = 0
    sample_path!(m, start, dest, temp)
end

function sample_path!(m::Matrix{Int64}, start::Int, dest::Int, temp::Float64,
                      dir_prev = WFC.Right)
    cis = CartesianIndices(m)
    dest_ci = cis[dest]
    n = size(m, 1)
    gs = GridSpace(n * n, (n, n))
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
        # @show current
        # @show ns
        # @show ws
        _ws = WFC.softmax(ws, temp)
        # @show _ws
        # @show argmax(ws)
        # @show argmax(_ws)
        if all(isinf, ws)
            display(m)
            error()
        end
        selected = isapprox(temp, 0.0) ? argmax(ws) : categorical(softmax(ws, temp))
        prev = current
        dir_prev = dir
        current, dir = ns[selected]
        add_segment!(m, prev, dir, dir_prev)
        temp *= 0.95
    end
    add_segment!(m, current, WFC.Right, dir)
    return m
end

function sample_pair(n::Int,
                     start::Int,
                     left_door::Int,
                     right_door::Int,
                     path_temp::Float64 = 2.0,
                     fix_steps::Int64 = 10,
                     extra_pieces::Int64 = 13,
                     piece_size::Int64 = 5)


    # sample some obstacles in the middle section of the roo
    # oweights = gen_obstacle_weights(right_room, Int64[])
    # right_room = furniture_gm(right_room, oweights,
    #                           extra_pieces, piece_size)
    # left_room  = add(left_room, right_room)

    # sample a random path for right door
    template = sample_path(n, start, right_door, path_temp)
    display(template)
    # sample path for left door
    #     - branch from right path, at random point
    #     - but not too close to entrance
    start_from = rand(findall(==(3), vec(template))[2:end-1])
    template[start_from] = 4 # update to connector
    display(template)
    println("######### sample_path! ##########")
    @show start_from
    sample_path!(template, start_from, left_door, path_temp, WFC.Above)

    display(template)


    sp = GridSpace(length(template), size(template))
    ts = TileSet(ht3d_elems, sp)
    # template = [1 1; 0 2]
    ws = WaveState(template, sp, ts)
    # display(template)
    # display(ws.weights)
    @time collapse!(ws, sp, ts)
    display(ws.wave)
    GranularScenes.display_mat(Float64.(WFC.expand(ws, sp, ts)))
    error()

    # # path_tiles = chisel_path(right_room, chisel_temp)
    # # right_path = findall(vec(path_tiles))
    # # # add some obstacles to coerce this path
    # # fixed = fix_shortest_path(right_room, right_path, fix_steps)
    # # right_room = add(right_room, fixed)
    # right_path = dpath(right_room)

    # # # add obstacles to left room
    # # left_room = add(left_room, fixed)
    # # # left path accounting for right path
    # # visited = Matrix{Bool}(data(left_room) .!= floor_tile)
    # # visited[right_path] .= true
    # # path_tiles = chisel_path(left_room, chisel_temp, visited)
    # # left_path = findall(vec(path_tiles))
    # # fixed = fix_shortest_path(left_room, left_path, fix_steps)
    # # left_room = add(left_room, fixed)
    # left_path = dpath(left_room)

    # # # add left chisels to right door
    # # right_room = add(right_room, fixed)
    # # right_path = dpath(right_room)

    # (left_room, left_path, right_room, right_path)
end

function eval_pair(left_door::GridRoom,
                   left_path::Vector{Int64},
                   right_door::GridRoom,
                   right_path::Vector{Int64}
                   )

    tile = 0
    # Not a valid sample
    ((isempty(left_path) || isempty(right_path)) ||
        length(intersect(left_path, right_path)) > 7 ||
        abs(length(right_path) - length(left_path)) > 3) &&
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
        # ns = collect(neighbors(g, tid))
        # for n = ns
        #     rem_edge!(gt, tid, n)
        # end
        # does it block path ?
        new_path = dpath(right_temp)
        if isempty(new_path) || length(new_path) > nl + 7
            left_temp = add(left_door, Set{Int64}(tid))
            new_left_path = dpath(left_temp)
            if new_left_path == left_path
                tile = tid
                break
            end
        # else
            # # reset block
            # for n = ns
            #     add_edge!(gt, tid, n)
            # end
        end
    end

    return tile
end



function main()
    name = "path_block_maze_$(Dates.today())"
    dataset_out = "/spaths/datasets/$(name)"
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)


    # Parameters
    room_steps = (16, 16)
    room_bounds = (32., 32.)
    entrance = [8, 9]
    door_rows = [4, 13]
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

    i = 1 # scene id
    c = 0 # number of attempts;
    while i <= n && c < 1000 * n
        # generate a room pair
        (left, lpath, right, rpath) = sample_pair(16, 8, doors[1], doors[2],)
        tile = eval_pair(left, lpath, right, rpath)
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
