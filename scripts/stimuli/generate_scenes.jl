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
# Trial generation
#################################################################################

function evaluate_sample!(temp::RoomProcess, x::RoomProcess,
                          other_door::Int)
    # path plan
    score, path = astar!(x)
    idx = 0

    # no valid path
    iszero(score) && return (score, path, idx)

    # reflect path for other door
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

    # replan in case shorter path now exists
    score, path = astar!(x)
    idx = 0

    # no valid path
    iszero(score) && return (score, path, idx)


    # Criterion: obstacle density On == Off
    left_count = count(==(b), x.m[1:center, :])
    right_count = count(==(b), x.m[(center+1):end, :])
    left_count != right_count && return (0, path, idx)

    # Criterion: D1 path > D2 path
    copy!(temp, x)
    temp.ext = other_door
    (oscore, opath) = astar!(temp)
    score - oscore < 3 && return (0, path, 0)

    # Criterion: no similar short paths
    np = length(path)
    sigcount = 0 # times blocking changes path
    largest  = 0 # largest path block
    toblock  = 0
    # path too short to modify
    np < 15 && return (0, Int[], 0)
    @inbounds for i = 1:np
        path_idx = path[i]
        path_row = (path_idx - 1) % rowsize + 1
        path_col = div(path_idx - 1, rowsize) + 1
        # skip regions that are
        #   1) too eccentric (horizontally)
        #   2) too close to the door (depth)
        if (abs(path_row - center) > 5) ||
            (rowsize - path_col < 6)
            continue
        end
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
    toblock === 0 && return (score, path, toblock)


    return (score, path, toblock)
end


function run_sample(x, t, d1, d2)
    sample_room!(x)
    # clear near doors
    clear_region!(x, d1, 2)
    clear_region!(x, d2, 2)
    clear_region!(x, d1-1, 1)
    clear_region!(x, d2+1, 1)
    # clear_region!(x, ceil(Int, (d1 + d2) / 2))
    evaluate_sample!(t, x, d2)
end


# Called in `main`
function sample_trial!(x, t, d1, d2)
    score = 0
    iter = 0
    toblock = 0
    path = Int[]
    while toblock === 0 && iter < 500000
        (score, path, toblock) = run_sample(x, t, d1, d2)
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
            println()
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
    cp("$(@__DIR__)/room_process.jl", "$(scenes_out)_room_process.jl"; force=true)


    # display(path)
    # @profilehtml proftest(x, 10)
    # @profilehtml proftest(x, 100000)

    # b = @benchmark proftest($x, 1000)
    # display(b)

    println("Finished with: $(dataset_out)")
    println("Found $(count) / $(attempts) rooms")

    return 0
end

main();
