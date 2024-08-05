using CSV
using JSON
using Rooms
using FileIO
using ArgParse
using DataFrames

blender_args = Dict(
    :mode => "full",
    # :mode => "none",
    :template => "/spaths/datasets/vss_template.blend",
    :script => "$(@__DIR__)/render.py"
)

function render_stims(df::DataFrame, name::String)
    out = "/spaths/datasets/$(name)/render_cycles"
    isdir(out) || mkdir(out)
    renderer = Blender(;blender_args...)
    for r in eachrow(df), door = 1:2
        base_p = "/spaths/datasets/$(name)/scenes/$(r.scene)_$(door).json"
        local base_s
        open(base_p, "r") do f
            base_s = JSON.parse(f)
        end
        base = from_json(GridRoom, base_s)
        p = "$(out)/$(r.scene)_$(door)"
        Rooms.render(renderer, base, p)
        blocked = add(base, Set{Int64}(r.tidx))
        p = "$(out)/$(r.scene)_$(door)_blocked"
        Rooms.render(renderer, blocked, p)
    end
end

function main()
    cmd = ["path_block_2024-03-14", "1"]
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
