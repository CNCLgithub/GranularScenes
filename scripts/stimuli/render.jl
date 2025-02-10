using CSV
using JSON
using Rooms
using FileIO
using ArgParse
using DataFrames
using GranularScenes: add

function blender_modes(mode::String)
    in(mode, ["door", "bookshelf"])
end

function gen_blender_args(mode::String)
    Dict(
        :template => "$(@__DIR__)/$(mode)_template.blend",
        :script => "$(@__DIR__)/render_$(mode).py",
        :blender => "/spaths/bin/blender-4.2.0-linux-x64/blender",
    )
end

function render_stims(df::DataFrame, name::String, mode::String)
    out = "/spaths/datasets/$(name)/render_$(mode)"
    isdir(out) || mkdir(out)
    blender_args = gen_blender_args(mode)
    for r in eachrow(df)
        base_p = "/spaths/datasets/$(name)/scenes/$(r.scene).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        blocked = add(base, Set{Int64}(r.blocked))

        p = "$(out)/$(r.scene)_1"
        renderer =  Blender(;blender_args..., mode = "noflip")
        Rooms.render(renderer, base, p)

        p = "$(out)/$(r.scene)_1_blocked"
        Rooms.render(renderer, blocked, p)

        p = "$(out)/$(r.scene)_2"
        renderer =  Blender(;blender_args..., mode = "flip")
        Rooms.render(renderer, base, p)

        p = "$(out)/$(r.scene)_2_blocked"
        Rooms.render(renderer, blocked, p)
    end
    return nothing
end

function main()
    cmd = ["window-0.1/2025-02-05_vifdDO", "door", "0"]
    args = parse_commandline(;x=cmd)
    name = args["dataset"]
    src = "/spaths/datasets/$(name)"
    df = DataFrame(CSV.File("$(src)/scenes.csv"))
    if args["scene"] != 0
        df = df[df.scene .== args["scene"], :]
    end
    render_stims(df, name, args["mode"])
    return nothing
end

function parse_commandline(;x=ARGS)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "dataset"
        help = "Which scene to run"
        arg_type = String
        required = true

        "mode"
        help = "Which render variant"
        arg_type = String
        required = true
        range_tester = blender_modes

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
