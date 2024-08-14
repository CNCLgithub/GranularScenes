using CSV
using JSON
using Rooms
using FileIO
using ArgParse
using DataFrames
using GranularScenes: add, display_mat

blender_args = Dict(
    :template => "$(@__DIR__)/stair_template.blend",
    :script => "$(@__DIR__)/render_stairs.py",
    :blender => "/spaths/bin/blender-4.2.0-linux-x64/blender"
)

function load_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

function render_stims(df::DataFrame, name::String)
    out = "/spaths/datasets/$(name)/render_stairs"
    isdir(out) || mkdir(out)
    for r in eachrow(df), door = 1:2
        base = load_scene("/spaths/datasets/$(name)/scenes/$(r.scene)_$(door).json")
        p = "$(out)/$(r.scene)_$(door)"
        display_mat(Float64.(data(base) .== floor_tile))
        renderer = Blender(;blender_args...,
                           mode = door == 1 ? "noflip" : "flip")
        Rooms.render(renderer, base, p)

        blocked = load_scene("/spaths/datasets/$(name)/scenes/$(r.scene)_$(door)_blocked.json")
        p = "$(out)/$(r.scene)_$(door)_blocked"
        Rooms.render(renderer, blocked, p)
    end
end

function main()
    cmd = ["path_block_maze/2024-08-13_leRe54", "0"]
    args = parse_commandline(;x=cmd)

    name = args["dataset"]
    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    seeds = unique(df.scene)
    if args["scene"] == 0
        seeds = unique(df.scene)
    else
        seeds = [args["scene"]]
        df = df[df.scene .== args["scene"], :]
    end

    render_stims(df, name)
    return nothing
end



function parse_commandline(;x=ARGS)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "dataset"
        help = "Which scene to run"
        arg_type = String
        required = true

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        required = true


        "--threads"
        help = "Number of threads for cycles"
        arg_type = Int64
        default = 4
    end
    return parse_args(x, s)
end



main();
