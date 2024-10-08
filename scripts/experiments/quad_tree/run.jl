using CSV
using JSON
using JLD2
using Rooms
using FileIO
using ArgParse
using Accessors
using Gen_Compose
using GranularScenes


function parse_commandline(c)
    s = ArgParseSettings()

    @add_arg_table! s begin

        "--dataset"
        help = "Trial dataset"
        arg_type = String
        default = "path_block_2024-03-14"

        "--gm"
        help = "Generative Model params"
        arg_type = String
        default = "$(@__DIR__)/gm.json"

        "--proc"
        help = "Inference procedure params"
        arg_type = String
        default = "$(@__DIR__)/procedure.json"

        "--decision"
        help = "Decision procedure params"
        arg_type = String
        default = "$(@__DIR__)/decision.json"

        "--ddp"
        help = "DDP config"
        arg_type = String
        default = "/project/scripts/nn/configs/og_decoder.yaml"

        "--reverse", "-f"
        help = "Infer image 2 then image 1"
        action = :store_true

        "--restart", "-r"
        help = "Whether to resume inference"
        action = :store_true

        "attention"
        help = "What attention protocol to use"
        arg_type = Symbol
        range_tester = x -> x == :ac || x == :un
        default = :ac

        "granularity"
        help = "What granularity schema to use"
        arg_type = Symbol
        range_tester = x -> x == :fixed || x == :multi
        default = :multi

        "scene"
        help = "Which scene to run"
        arg_type = Int64
        default = 2

        "chain"
        help = "The number of chains to run"
        arg_type = Int
        default = 1

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
    cidx = CartesianIndices(Rooms.steps(r))[tidx]
    println("Blocked tile at location $(cidx)")
    GridRoom(r, d)
end

function main(c=ARGS)
    args = parse_commandline(c)

    dataset = args["dataset"]
    base_path = "/spaths/datasets/$(dataset)/scenes"

    scene = args["scene"]

    println("Running inference on scene $(scene)")
    args["restart"] = true

    manifest = CSV.File(base_path * ".csv")
    tile = manifest[:tidx][scene]


    attention = args["attention"]
    granularity = args["granularity"]

    model = "$(attention)_$(granularity)"

    doors = [2]

    for door = doors
        out_path = "/spaths/experiments/flicker_$(dataset)_$(model)/$(scene)_$(door)"
        if args["reverse"]
            out_path *= "_reversed"
        end

        base_p = joinpath(base_path, "$(scene)_$(door).json")
        room1 = load_scene(base_p)
        room2 = block_tile(room1, tile)

        gm_params = QuadTreeModel(room1;
                                  read_json(args["gm"])...,
                                  render_kwargs =
                                      Dict(:resolution => (128,128)))

        println("Saving results to: $(out_path)")

        dargs = read_json(args["decision"])

        q1 = query_from_params(room1, gm_params)
        q2 = query_from_params(room2, gm_params)


        model_params = first(q1.args)
        img = GranularScenes.render(model_params.renderer, room1)
        ddp_params = DataDrivenState(;config_path = args["ddp"],
                                     var = 0.25)


        proc_1_kwargs = read_json(args["proc"])
        proc_2_kwargs = deepcopy(proc_1_kwargs)

        proc_1_kwargs[:ddp]  = generate_cm_from_ddp
        proc_1_kwargs[:ddp_args] = (ddp_params, img, model_params, 3, 3)
        proc_2_kwargs[:ddp]  = generate_cm_from_ddp
        proc_2_kwargs[:ddp_args] = (ddp_params, img, model_params, 3, 3)


        protocol = attention == :ac ?
            AdaptiveComputation() : UniformProtocol()
        proc_1_kwargs[:protocol] =
            proc_2_kwargs[:protocol] = protocol


        if granularity == :fixed
            # TODO: Update this branch
            proc_1_kwargs[:ddp] = fixed_granularity_cm
            proc_1_kwargs[:ddp_args] = (gm_params, 5) # HACK: hard coded
            proc_2_kwargs[:ddp] = generate_cm_from_ppd
            proc_2_kwargs[:ddp_args] = (gm_params,
                                           0.0) # 0 var forces max split
            # No split-merge moves
            proc_1_kwargs[:sm_budget] =
                proc_2_kwargs[:sm_budget] = 0
        # REVIEW: no longer needed?
        # else
        #     proc_2_kwargs[:ddp] = generate_cm_from_ppd
        #     proc_2_kwargs[:ddp_args] = (gm_params,
        #                                    0.0175)
        end
        proc_1 = AdaptiveMH(; proc_1_kwargs...)
        proc_2 = AdaptiveMH(; proc_2_kwargs...)

        try
            isdir("/spaths/experiments/$(dataset)_$(model)") ||
                mkpath("/spaths/experiments/$(dataset)_$(model)")
            isdir(out_path) || mkpath(out_path)
        catch e
            println("could not make dir $(out_path)")
        end

        # how many chains to run
        for c = 1:args["chain"]
            # Random.seed!(c)
            c1_out = joinpath(out_path, "c1_$(c).jld2")
            c2_out = joinpath(out_path, "c2_$(c).jld2")
            outs = [c1_out, c2_out]
            complete = false
            if any(isfile, outs)
                println("Record(s) found for chain: $(c)")
                if args["restart"]
                    println("restarting")
                    foreach(o -> isfile(o) && rm(o), outs)
                else
                    complete = all(isfile, outs)
                end
            end
            if !complete
                println("Starting chain $c")
                chain_steps = dargs[:epoch_steps] * dargs[:epochs]
                log1 = MemLogger(dargs[:margin_size])
                log2 = MemLogger(dargs[:margin_size])
                c1 = Gen_Compose.initialize_chain(proc_1, q1, chain_steps)
                c2 = Gen_Compose.initialize_chain(proc_2, q2, chain_steps)
                e = 1; klm = zeros(16, 16); max_kl = 0.0; cidx = CartesianIndex(0, 0);
                while e < dargs[:epochs] && max_kl < dargs[:kl_threshold]
                    (klm, max_kl, cidx, eloc) =
                        search_step!(c1, c2, log1, log2,
                                     dargs[:epoch_steps],
                                     dargs[:search_weight])
                    e += 1
                end
                # println("Inference END")
                # GranularScenes.viz_chain(c1)
                # GranularScenes.viz_chain(c2)
                # println("Expected $(eloc); Max KL: $(max_kl), @ index $(cidx)")
                # GranularScenes.display_mat(klm)
            end
            println("Chain $(c) complete")
        end
    end
    return nothing
end

main();
