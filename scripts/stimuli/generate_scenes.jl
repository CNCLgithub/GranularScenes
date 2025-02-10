#################################################################################
# Imports
#################################################################################
using CSV
using JSON
using Dates
using Rooms
using DataFrames
using ProgressMeter
using BenchmarkTools
using StatProfilerHTML

include("/project/scripts/stimuli/room_process.jl")

#################################################################################
# Globals
#################################################################################

MIN_PATH_IMPACT = 15
MIN_DOOR_COST_DIFF = 4
MAX_ECC = 5


#################################################################################
# Trial generation
#################################################################################

function eval_path_block!(temp::RoomProcess, x::RoomProcess,
                          score::Int,
                          path::Vector{Int})
    rowsize = size(x.m, 1)
    center = div(rowsize, 2)
    np = length(path)
    sigcount = 0 # times blocking changes path
    largest  = 0 # largest path block
    toblock  = 0 # location to add obstacle
    @inbounds for i = 1:np
        path_idx = path[i]
        path_row = (path_idx - 1) % rowsize + 1
        path_col = div(path_idx - 1, rowsize) + 1
        # skip regions that are
        #   1) too eccentric (horizontally)
        #   2) too close to the door (depth)
        # if (abs(path_row - center) > MAX_ECC) ||
        #     (rowsize - path_col < MAX_ECC)
        #     continue
        # end
        copy!(temp, x)
        block!(temp, path_idx)
        tscore, tpath = astar!(temp)
        delta = abs(tscore - score)
        if delta > MIN_PATH_IMPACT
            sigcount += 1
            if delta > largest
                largest = delta
                toblock = path_idx
            end
        end
    end
    return toblock
end

function compare_path_complexities!(temp, x, score, other_door)
    copy!(temp, x)
    temp.ext = other_door
    (oscore, opath) = astar!(temp)
    score - oscore < MIN_DOOR_COST_DIFF
end

function compare_obstacle_densities(x)
    rowsize = size(x.m, 1)
    center = div(rowsize, 2)
    left_count  = count(==(b), x.m[1:center, :])
    right_count = count(==(b), x.m[(center+1):end, :])
    left_count != right_count
end

function evaluate_valid_path!(x)
    # path plan
    score, path = astar!(x)

    # no valid path
    iszero(score) && return (score, path)

    # Scan path and
    # 1) check if it is too eccentric
    # 2) reflect path for other door
    rowsize = size(x.m, 1)
    center = div(rowsize, 2)
    ecc = false
    for idx = path
        cidx = div(idx-1, rowsize) + 1
        ridx = (idx - 1) % rowsize + 1
        if (abs(ridx - center) > MAX_ECC)
            return (0, path)
        end
        offset = center - (ridx - center) + 1
        if x.m[offset, cidx] === f
            x.c[cidx] -= 1
        end
        x.m[offset, cidx] = o
    end

    # replan in case shorter path now exists
    score, path = astar!(x)
end

function evaluate_sample!(temp::RoomProcess, x::RoomProcess,
                          other_door::Int)
    # Criterion: path to door exists
    (score, path) = evaluate_valid_path!(x)
    score === 0 && return (0, path, 0)

    # Criterion: obstacle density On == Off
    compare_obstacle_densities(x) && return (0, path, 0)

    # Criterion: D1 path > D2 path
    compare_path_complexities!(temp, x, score, other_door) &&
        return (0, path, 0)

    # Criterion: no similar short paths
    toblock = eval_path_block!(temp, x, score, path)

    return (score, path, toblock)
end


function run_sample!(x, t, d1, d2)
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
    evaluate_sample!(t, x, d2)
end


# Called in `main`
function sample_trial!(x, t, d1, d2)
    score = 0
    iter = 0
    toblock = 0
    path = Int[]
    while toblock === 0
        (score, path, toblock) = run_sample!(x, t, d1, d2)
        iter += 1
    end

    return (score, path, toblock)
end


#################################################################################
# profiling
#################################################################################

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
    attempts = 10


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

    x = RoomProcess(dims, start, dest1)
    t = RoomProcess(dims, start, dest1)

    count = 0
    p = Progress(attempts; dt=1, desc = "Generating rooms...")
    while count < attempts
        (score, path, toblock) = run_sample!(x, t, dest1, dest2)
        if toblock !== 0
            previous = df[:, :blocked]
            if !in(toblock, previous)
                count += 1
                toflip = (count-1) % 2
                r = room_from_process(x)
                println()
                display(r)
                y = similar(x.m, Bool)
                fill!(y, false)
                y[path] .= true
                display(y)
                # save
                push!(df, [count, toflip, toblock])
                open("$(scenes_out)/$(count).json", "w") do f
                    write(f, r |> json)
                end
            end
        end
        update!(p, count)
    end
    finish!(p)

    # saving summary / manifest
    sort!(df, :scene)
    CSV.write("$(scenes_out).csv", df)

    # copy this script for reproducability
    cp(@__FILE__, "$(scenes_out)_script.jl"; force=true)
    cp("$(@__DIR__)/room_process.jl", "$(scenes_out)_room_process.jl"; force=true)


    # display(path)
    # @profilehtml proftest(x, 10)
    # @profilehtml proftest(x, 100000)

    # b = @benchmark proftest($x, 1000)
    # display(b)

    println("Finished with: $(dataset_out)")

    return 0
end

main();
