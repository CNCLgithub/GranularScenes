#################################################################################
# Imports
#################################################################################
using CSV
using JSON
using Dates
using Rooms
using DataFrames
using LinearAlgebra
using ProgressMeter
using BenchmarkTools
using StatProfilerHTML
using Gen: uniform_discrete
using DataStructures: PriorityQueue, dequeue!

#################################################################################
# Math
#################################################################################

"""
Samples from a categorical distribution with no memory allocation

Note! Does not check if `sum(ws) == 1`
"""
function fast_categorical(ws::Array{Float64})
    n = length(ws)
    x = 1
    w = 0.0
    a = rand()
    @inbounds for i = 1:(n-1)
        w += ws[i]
        a < w && break
        x += 1
    end
    return x
end

function city_block_dist(icol::Int, irow::Int, tcol::Int, trow::Int)
    abs(icol - tcol) + abs(irow - trow)
end

#################################################################################
# Room process
#################################################################################

@enum Cell f o b

import Base.show

function Base.show(io::IO, m::MIME"text/plain", x::Cell)
    show(io, m, Integer(x))
end

mutable struct RoomProcess
    # obstacles
    m::Matrix{Cell}
    c::Vector{Int}
    w::Vector{Float64}
    d::Vector{Float64}
    # pathing
    ent::Int
    ext::Int
    open_set::PriorityQueue
    closed_set::Matrix{Bool}
    g_score::Matrix{Int}
    came_from::Matrix{Int}
    heuristic::Function
end

import Base.copy!

function Base.copy!(dst::RoomProcess, src::RoomProcess)
    copy!(dst.m, src.m)
    copy!(dst.c, src.c)
    copy!(dst.w, src.w)
    copy!(dst.d, src.d)
    dst.ent = src.ent
    dst.ext = src.ext
    copy!(dst.open_set, src.open_set)
    copy!(dst.closed_set, src.closed_set)
    copy!(dst.g_score, src.g_score)
    copy!(dst.came_from, src.came_from)
    dst.heuristic = src.heuristic
end

function RoomProcess(dims::Tuple{Int, Int}, ent::Int, ext::Int)
    m = fill(f, dims)
    c = fill(dims[1], dims[2])
    w = Vector{Float64}(undef, dims[2])
    d = Vector{Float64}(undef, 4)

    # astar
    open_set = PriorityQueue{Int,Float64}()
    closed_set = similar(m, Bool)
    g_score = similar(m, Int)
    came_from = similar(m, Int)
    nr = size(m, 1)
    tcol = ceil(Int, ext / nr)
    trow = (ext - 1) % nr + 1
    heuristic = (c, r) -> city_block_dist(c, r, tcol, trow)

    RoomProcess(m,c,w,d, ent, ext,
                open_set, closed_set,
                g_score, came_from, heuristic)
end

countfree(x::RoomProcess) = sum(x.c)


function sample_start(x::RoomProcess)
    copyto!(x.w, x.c)
    lmul!(1.0 / sum(x.c), x.w)
    c = fast_categorical(x.w)
    ri = uniform_discrete(1, x.c[c])
    r = 1
    while ri > 0
        if x.m[r, c] === f
            ri -= 1
        end
        r += 1
    end
    (c, r)
end

function block!(x::RoomProcess, i::Int)
    rowsize = size(x.m, 1)
    c = div(i-1, rowsize) + 1
    r = (i - 1) % rowsize + 1
    block!(x, c, r)
end

function block!(x::RoomProcess, c::Int, r::Int)
    checkloc(x, c, r) || return false
    x.c[c] -= 1
    x.m[r, c] = b
    return true
end

function sample_region!(x::RoomProcess, c::Int, r::Int)
    rdim = uniform_discrete(0, 3)
    cdim = uniform_discrete(0, 3)
    for i = 0:cdim, j = 0:rdim
        !(block!(x, c + i, r + j)) && break
    end
    return nothing
end

function sample_snake!(x::RoomProcess, c::Int, r::Int)
    n = uniform_discrete(0, 4)
    success = block!(x, c, r)
    while !(success) || n > 0
        d = sample_direction(x, c, r)
        iszero(d) && break # ran out of places
        c, r = shiftby(d, c, r)
        success = block!(x, c, r)
        n -= 1
    end
    return nothing
end

@inline checkloc(x::RoomProcess, c::Int, r::Int) =
    checkbounds(Bool, x.m, c, r) && x.m[r, c] === f

function sample_direction(x::RoomProcess, c::Int, r::Int)
    x.d[1] = checkloc(x, c - 1, r) ? 1.0 : 0.0 # left
    x.d[2] = checkloc(x, c + 1, r) ? 1.0 : 0.0 # right
    x.d[3] = checkloc(x, c, r - 1) ? 1.0 : 0.0 # up
    x.d[4] = checkloc(x, c, r + 1) ? 1.0 : 0.0 # down
    s = sum(x.d)
    iszero(s) && return 0
    lmul!(1.0 / sum(x.d), x.d)
    fast_categorical(x.d)
end

function shiftby(d::Int, c::Int, r::Int)
    d == 1 && return (c - 1, r) # left
    d == 2 && return (c + 1, r) # right
    d == 3 && return (c, r - 1) # up
    (c, r + 1)                  # down
end


function clear_col!(x::RoomProcess, c::Int)
    x.m[:, c] .= o
    x.c[c] = 0
    return nothing
end

function clear_region!(x::RoomProcess, i::Int, radius::Int = 1)
    rowsize = size(x.m, 1)

    c = div(i-1, rowsize) + 1
    r = (i - 1) % rowsize + 1

    cstart = max(1, c - radius)
    cstop = min(size(x.m, 2), c + radius)

    rstart = max(1, r - radius)
    rstop = min(rowsize, r + radius)
    rcount = rstop - rstart

    for cidx = cstart:cstop
        fcount = count(==(f), x.m[rstart:rstop])
        x.m[rstart:rstop, cidx] .= o
        x.c[cidx] -= fcount
    end

    return nothing
end

function clear_row!(x::RoomProcess, r::Int)
    x.m[r, :] .= o
    for c = 1:size(x.m, 2)
        x.c[c] = max(x.c[c] - 1, 0)
    end
    return nothing
end

function sample_room!(x::RoomProcess)
    reset_chasis!(x)
    n = 10
    for _ = 1:n
        countfree(x) > 0 || break # no more space
        (c, r) = sample_start(x)
        if rand() > 0.5
            sample_region!(x, c, r)
        else
            sample_snake!(x, c, r)
        end
    end

    # display(x.m)

    return nothing
end

#################################################################################
# A*
#################################################################################

function reset_astar!(x::RoomProcess)

    empty!(x.open_set)
    x.open_set[x.ent] = 0.0

    fill!(x.closed_set, false)

    fill!(x.g_score, typemax(Int))
    x.g_score[x.ent] = 0

    fill!(x.came_from, 0)
    x.came_from[x.ent] = x.ent

    return nothing
end

function astar!(x::RoomProcess)
    reset_astar!(x)
    total_path = Int[]
    score = 0
    current = 0
    rowsize = size(x.m, 1)
    @inbounds while !isempty(x.open_set)
        current = dequeue!(x.open_set)
        c = div(current-1, rowsize) + 1
        r = (current - 1) % rowsize + 1

        if current === x.ext
            reconstruct_path!(total_path, x.came_from, current)
            score = x.g_score[current]
            return score, total_path
        end

        x.closed_set[current] = true

        for dir in 1:4 # left, right, up, down
            (nc, nr) = neighbor = shiftby(dir, c, r)
            # go next if neighor is in closed set (or out of bounds)
            get(x.closed_set, (nr, nc), true) && continue
            @inbounds begin
                cost = x.m[nr, nc] === b ?
                    typemax(Int) : x.g_score[current] + 1
            end

            if cost < get(x.g_score, (nr, nc), typemax(Int))
                x.g_score[nr, nc] = cost
                priority = cost + x.heuristic(nc, nr)
                nidx = nr + (nc - 1)*size(x.m, 1)
                x.open_set[nidx] = priority
                x.came_from[nr, nc] = current
            end
        end
    end
    return score, total_path
end

function reconstruct_path!(
    total_path::Vector{Int}, # a vector to be filled with the shortest path
    came_from, # a vector holding the parent of each node in the A* exploration
    end_idx, # the end vertex
)
    curr_idx = end_idx
    while came_from[curr_idx] != curr_idx
        pushfirst!(total_path, curr_idx)
        curr_idx = came_from[curr_idx]
    end
    pushfirst!(total_path, curr_idx)
    return nothing
end

#################################################################################
# Trial generation
#################################################################################

function reset_chasis!(x::RoomProcess)
    fill!(x.m, f)
    fill!(x.c, size(x.m, 1))
    foreach(c -> clear_col!(x, c), 1:4)

    # clear_col!(x, 1)
    # clear_row!(x, 1)
    # clear_row!(x, size(x.m, 1))
    return nothing
end

function room_from_process(x::RoomProcess)
    steps = size(x.m)
    d = similar(x.m, Tile)
    fill!(d, floor_tile)
    # set entrances and exits
    # these are technically floors but are along the border
    d[x.ent] = floor_tile
    d[x.ext] = floor_tile
    @inbounds for i = eachindex(x.m)
        if x.m[i] === b
            d[i] = obstacle_tile
        end
    end
    g = Rooms.init_pathgraph(GridRoom, d)
    GridRoom(steps, (32.0, 32.0), [x.ent], [x.ext], g, d)
end

function evaluate_sample!(temp::RoomProcess, x::RoomProcess)
    # path plan
    score, path = astar!(x)
    idx = 0

    # no valid path
    iszero(score) && return (score, path, idx)

    # display(x.m)
    # reflect path
    rowsize = size(x.m, 1)
    center = div(rowsize, 2)
    for idx = path
        cidx = div(idx-1, rowsize) + 1
        ridx = (idx - 1) % rowsize + 1
        offset = center - (ridx - center) + 1
        if x.m[offset, cidx] === f
            x.c[cidx] -= 1
        end
        x.m[offset, cidx] = o
    end
    # display(x.m)
    # error()

    # path plan
    score, path = astar!(x)
    idx = 0

    # no valid path
    iszero(score) && return (score, path, idx)


    # Criterion 1: no similar paths
    np = length(path)
    sigcount = 0 # times blocking changes path
    largest  = 0 # largest path block
    toblock  = 0
    # path too short to modify
    np < 15 && return (0, Int[], 0)
    @inbounds for i = 1:(np - 7)
        path_idx = path[i]
        path_row = (path_idx - 1) % rowsize + 1
        path_col = div(path_idx - 1, rowsize) + 1
        # skip regions that are
        #   1) too eccentric (horizontally)
        abs(path_row - center) > 6 && continue
        #   2) too close to the door (depth)
        rowsize - path_col < 2 && continue
        copy!(temp, x)
        block!(temp, path_idx)
        tscore, tpath = astar!(temp)
        delta = abs(tscore - score)
        if delta > 10
            sigcount += 1
            if delta > largest
                largest = delta
                toblock = path_idx
            end
        end
    end

    # Criterion 3: obstacle density on both sides
    left_density = count(==(b), x.m[1:center, :])
    right_density = count(==(b), x.m[(center+1):end, :])
    perc_diff = (abs(left_density - right_density)) / left_density
    if perc_diff > 0.10
        score = 0
        toblock = 0
        path = Int[]
    end

    return (score, path, toblock)
end


function sample_trial!(x, t, d1, d2)
    score = 0
    iter = 0
    toblock = 0
    path = Int[]
    while toblock === 0 && iter < 100000
        (score, path, toblock) = run_sample(x, t, d1, d2)
        iter += 1
    end

    return (score, path, toblock)
end

#################################################################################
# profiling
#################################################################################

function run_sample(x, t, d1, d2)
    sample_room!(x)
    # clear near doors
    clear_region!(x, d1, 2)
    clear_region!(x, d2, 2)
    clear_region!(x, d1-1, 1)
    clear_region!(x, d2+1, 1)
    # clear_region!(x, ceil(Int, (d1 + d2) / 2))
    evaluate_sample!(t, x)
end

function proftest(x, n::Int)
    foreach(_ -> run_sample(x), 1:n)
    return nothing
end

#################################################################################
# main
#################################################################################


function main()
    dims = (16, 16)
    start = 8
    dest1 = 252
    dest2 = 244
    x = RoomProcess(dims, start, dest1)
    t = deepcopy(x)
    attempts = 10

    score = 0
    path = Int[]
    toblock = 0


    df = DataFrame(scene = Int64[],
                   flipx = Bool[],
                   blocked = Int[])
    name = "window-0.1"
    dataset_base = "/spaths/datasets/$(name)"
    isdir(dataset_base) || mkdir(dataset_base)
    dataset_out = mktempdir(dataset_base;
                            prefix = "$(Dates.today())_",
                            cleanup = false)
    isdir(dataset_out) || mkdir(dataset_out)

    scenes_out = "$(dataset_out)/scenes"
    isdir(scenes_out) || mkdir(scenes_out)

    count = 0
    @showprogress dt=1 desc="Generating rooms..." for _ = 1:attempts
        score, path, toblock = sample_trial!(x, t, dest1, dest2)
        if toblock !== 0
            count += 1
            r = room_from_process(x)
            display(r)
            y = similar(x.m, Bool)
            fill!(y, false)
            y[path] .= true
            display(y)
            # save
            toflip = (count-1) % 2
            push!(df, [count, toflip, toblock])
            open("$(scenes_out)/$(count).json", "w") do f
                write(f, r |> json)
            end
        end
    end

    # saving summary / manifest
    CSV.write("$(scenes_out).csv", df)

    # copy this script for reproducability
    cp(@__FILE__, "$(scenes_out)_script.jl"; force=true)


    # display(path)
    # @profilehtml proftest(x, 10)
    # @profilehtml proftest(x, 100000)

    # b = @benchmark proftest($x, 1000)
    # display(b)

    println("Finished with: $(dataset_out)")

    return 0
end

main();
