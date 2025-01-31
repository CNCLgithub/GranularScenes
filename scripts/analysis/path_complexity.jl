using CSV
using JSON
using Dates
using Rooms
using DataFrames

# for A*
include("/project/scripts/stimuli/room_process.jl")

function load_scene(name::String, room::Int)
    base_p = "/spaths/datasets/$(name)/scenes/$(room).json"
    local base_s
    open(base_p, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

function path_length(x::RoomProcess)
    score, total_path = astar!(x)

end

function analyze!(x::RoomProcess)

end

function main()
    dset = "window-0.1/2025-01-22_BJFn5j"
    df = DataFrame(
        :scene => Int64[],
        :door => Int64[],
        :cost => Float64[],
    )

    start = 8
    onpath = 252
    offpath = 244

    for scene = 1:10
        room = load_scene(dset, scene)
        for door_idx = 1:2
            door = door_idx === 1 ? offpath : onpath
            x = room_to_process(room, door)
            cost, path = astar!(x)
            l = length(path)
            push!(df, (;scene = scene, door = door_idx,
                       cost = cost))
        end
    end
    display(df)
    out = "/spaths/datasets/$(dset)/path_complexity.csv"
    CSV.write(out, df)
end

main();
