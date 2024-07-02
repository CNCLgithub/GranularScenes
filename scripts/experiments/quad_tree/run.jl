using CSV
using Gen: get_retval
using JSON
using JLD2
using FileIO
using ArgParse
# using DataFrames
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

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 1

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        default = 10

    end

    return parse_args(c, s)
end

function clear_wall(r::GridRoom)
    # remove wall near camera
    d = data(r)
    d[:, 1:2] .= floor_tile
    GridRoom(r, d)
end

function load_scene(path::String)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    # base = expand(from_json(GridRoom, base_s), 2)
    base = from_json(GridRoom, base_s)
    clear_wall(base)
end

function block_tile(r::GridRoom, tidx::Int)
    d = deepcopy(data(r))
    d[tidx] = obstacle_tile
    GridRoom(r, d)
end

function inference(proc, query, out)
    dlog = JLD2Logger(50, out)
    chain = run_chain(proc, query, proc.samples + 1, dlog)
    qt = get_retval(chain.state)
end


function main(c=ARGS)
    args = parse_commandline(c)
    base_path = "/spaths/datasets/$(dataset)/scenes"
    scene = args["scene"]

    println("Running inference on scene $(scene)")
    args["restart"] = true

    manifest = CSV.File(base_path * ".csv")
    del_tile = manifest[:tidx][scene]

    for door = [1, 2]
        out_path = "/spaths/experiments/$(dataset)/$(scene)_$(door)"

        base_p = joinpath(base_path, "$(scene)_$(door).json")
        train = load_scene(base_p)
        test = block_tile(train, del_tile)


        println("Saving results to: $(out_path)")

        # Load query (identifies the estimand)
        query = query_from_params(train, test, args["gm"];
                                  render_kwargs = Dict(:resolution => (256,256)))


        # model_params = first(query.args)
        # gt_img = GranularScenes.render(model_params.renderer, train)
        # save the gt image for reference
        # save_img_array(gt_img, "$(out_path)/gt.png")

        # ddp_params = DataDrivenState(;config_path = args["ddp"],
        #                              var = 0.001)

        ac_proc = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                        # ddp = generate_cm_from_ddp,
                        # ddp_args = (ddp_params, gt_img, model_params, 3),
                        protocol = AdaptiveComputation()
                        )
        un_proc = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                        # ddp = generate_cm_from_ddp,
                        # ddp_args = (ddp_params, gt_img, model_params, 3),
                        protocol = UniformProtocol()
                        )
        try
            isdir("/spaths/experiments/$(dataset)") || mkpath("/spaths/experiments/$(dataset)")
            isdir(out_path) || mkpath(out_path)
        catch e
            println("could not make dir $(out_path)")
        end


        # how many chains to run
        for c = 1:args["chain"]
            # Random.seed!(c)
            ac_out = joinpath(out_path, "ac_$(c).jld2")
            un_out = joinpath(out_path, "un_$(c).jld2")
            complete = false
            if isfile(ac_out) || isfile(un_out)
                println("Record(s) found for chain: $(c)")
                if args["restart"]
                    println("restarting")
                    isfile(ac_out) && rm(ac_out)
                    isfile(un_out) && rm(un_out)
                else
                    complete = isfile(ac_out) && isfile(un_out)
                end
            end
            if !complete
                println("Starting chain $c")
                ac_qt = inference(ac_proc, query, ac_out)
                # un_qt = inference(un_proc, query, un_out)
            end
            println("Chain $(c) complete")
        end
    end
    return nothing
end

main();
