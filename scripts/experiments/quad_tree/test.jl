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

using Random

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

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 2

        "door"
        help = "door"
        arg_type = Int64
        default = 1

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        default = 1

    end

    return parse_args(c, s)
end

function load_scene()
    r = GridRoom((16, 16), (16., 16.))
    d = data(r)
    d[:, 1:2] .= floor_tile
    d[2:4, 2:4] .= obstacle_tile
    expand(GridRoom(r, d), 2)
end



function main(c=ARGS)
    args = parse_commandline(c)
    println("Running inference on scene TEST")

    out_path = "/spaths/tests/inference"

    println("Saving results to: $(out_path)")

    room = load_scene()
    display(room)

    # Load query (identifies the estimand)
    query = query_from_params(room, args["gm"];
                              render_kwargs =
                                  Dict(:resolution => (256,256)))

    # Load estimator - Adaptive MCMC
    model_params = first(query.args)
    # ddp_params = DataDrivenState(;config_path = args["ddp"],
    #                              var = 0.15)
    gt_img = GranularScenes.render(model_params.renderer, room)

    proc = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                      samples = 300,
                      # ddp = generate_cm_from_ddp,
                      # ddp_args = (ddp_params, gt_img, model_params, 3),
                      # start with no ddp
                      # ddp_args = (ddp_params, gt_img, model_params),
                      #
                      # also start with no attention
                      protocol = UniformProtocol())

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

        if isfile(out) && args["restart"]
            println("chain $c restarting")
            rm(out)
        end
        complete = false
        if isfile(out)
            corrupted = true
            jldopen(out, "r") do file
                # if it doesnt have this key
                # or it didnt finish steps, restart
                if haskey(file, "current_idx")
                    n_steps = file["current_idx"]
                    if n_steps == proc.samples
                        println("Chain $(c) already completed")
                        corrupted = false
                        complete = true
                    else
                        println("Chain $(c) corrupted. Restarting...")
                    end
                end
            end
            corrupted && rm(out)
        end
        if !complete
            println("starting chain $c")
            nsteps = proc.samples
            dlog = JLD2Logger(10, out)
            chain = run_chain(proc, query, nsteps, dlog)
            qt = get_retval(chain.state)
            img = GranularScenes.render(model_params.renderer, qt)
            save_img_array(img, "$(out_path)/$(c)_img_mu.png")
            println("Chain $(c) complete")

            # scene = TaichiScene(room; resolution = (512,512),
            #                     window = true)
            # GranularScenes.window(scene, qt)
        end
    end

    return nothing
end



# function outer()
#     args = Dict("scene" => 8)
#     # args = parse_outer()
#     i = args["scene"]
#     # scene | door | chain | attention
#     cmd = ["$(i)","1", "1", "A"]
#     # cmd = ["--restart", "$(i)", "1", "1", "A"]
#     main(cmd);
# end

main();
