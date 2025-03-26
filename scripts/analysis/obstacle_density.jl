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

function main()
    dset = "window-0.1/2025-01-22_BJFn5j"
    df = DataFrame(
        :scene => Int64[],
        :front => Bool[],
        :left => Bool[],
        :count => Int64[],
    )

    start = 8
    onpath = 252
    offpath = 244

    for scene = 1:10
        room = load_scene(dset, scene)
        x = room_to_process(room, onpath)

        push!(df, (scene, true, true,   count(==(b), x.m[1:8, 1:8])))
        push!(df, (scene, true, false,  count(==(b), x.m[9:16, 1:8])))
        push!(df, (scene, false, true,  count(==(b), x.m[1:8, 9:16])))
        push!(df, (scene, false, false, count(==(b), x.m[9:16, 9:16])))
    end
    display(df)
    out = "/spaths/datasets/$(dset)/obstacle_density.csv"
    CSV.write(out, df)
end

main();
