using CSV
using Gen: get_retval
using JSON
using JLD2
using FileIO
using ArgParse
using DataFrames
using Gen_Compose
using Rooms
using GranularScenes
# using FunctionalScenes: shift_furniture

using Random

dataset = "path_block_2024-03-14"

function parse_commandline(c)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--gm"
        help = "Generative Model params"
        arg_type = String
        default = "$(@__DIR__)/gm.json"

        "--proc"
        help = "Inference procedure params"
        arg_type = String
        default = "$(@__DIR__)/proc.json"

        "--ddp"
        help = "DDP config"
        arg_type = String
        default = "/project/scripts/nn/configs/og_decoder.yaml"

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "--viz", "-v"
        help = "Whether to render masks"
        action = :store_true

        "--move", "-m"
        help = "Which scene to run"
        arg_type = Symbol
        required = false

        "--furniture", "-f"
        help = "Which scene to run"
        arg_type = Int64
        required = false

        "--attention", "-a"
        help = "Attention module"
        action = :store_true

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 2

        "door"
        help = "door"
        arg_type = Int64
        default = 2

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        default = 5

    end

    return parse_args(c, s)
end

function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

function load_base_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
    clear_wall(base)
end



function main(c=ARGS)
    args = parse_commandline(c)
    base_path = "/spaths/datasets/$(dataset)/scenes"
    scene = args["scene"]
    door = args["door"]
    base_p = joinpath(base_path, "$(scene)_$(door).json")

    println("Running inference on scene $(scene)")


    args["restart"] = true
    args["attention"] = false
    # args["attention"] = true
    out_path = "/spaths/experiments/$(dataset)/$(scene)_$(door)"
    if args["attention"]
        out_path *= "_wattention"
    end

    println("Saving results to: $(out_path)")

    room = load_base_scene(base_p)

    # Load query (identifies the estimand)
    query = query_from_params(room, args["gm"];
                              render_kwargs =
                                  Dict(:resolution => (256,256)))

    # Load estimator - Adaptive MCMC
    model_params = first(query.args)
    # ddp_params = DataDrivenState(;config_path = args["ddp"],
    #                              var = 0.001)
    gt_img = GranularScenes.render(model_params.renderer, room)

    protocol = args["attention"] ? AdaptiveComputation() : UniformProtocol()
    proc = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                      # ddp = generate_cm_from_ddp,
                      # ddp_args = (ddp_params, gt_img, model_params, 3),
                      protocol = protocol
                      )

    println("Loaded configuration...")

    try
        isdir("/spaths/experiments/$(dataset)") || mkpath("/spaths/experiments/$(dataset)")
        isdir(out_path) || mkpath(out_path)
    catch e
        println("could not make dir $(out_path)")
    end

    # save the gt image for reference
    save_img_array(gt_img, "$(out_path)/gt.png")

    # how many chains to run
    for c = 1:args["chain"]
        # Random.seed!(c)
        out = joinpath(out_path, "$(c).jld2")
        complete = false
        if isfile(out)
            println("Record found for chain: $(c)")
            if args["restart"]
                println("restarting")
                rm(out)
            else
                println("skipping")
                complete = true
            end
        end
        println("Starting chain $c")
        nsteps = proc.samples
        dlog = JLD2Logger(50, out)
        chain = run_chain(proc, query, nsteps, dlog)
        qt = get_retval(chain.state)
        img = GranularScenes.render(model_params.renderer, qt)
        save_img_array(img, "$(out_path)/$(c)_img_mu.png")
        println("Chain $(c) complete")
    end

    return nothing
end

main();
