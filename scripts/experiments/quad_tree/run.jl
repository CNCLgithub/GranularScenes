using CSV
using JSON
using JLD2
using Rooms
using FileIO
# using Random
using ArgParse
using Accessors
using Gen_Compose
using GranularScenes
using Gen: get_retval
using LinearAlgebra: lmul!


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

        "--ddp"
        help = "DDP config"
        arg_type = String
        default = "/project/scripts/nn/configs/og_decoder.yaml"

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
        default = 1

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
    GridRoom(r, d)
end

function train(proc, query, out)
    dlog = JLD2Logger(50, out)
    c = run_chain(proc, query, proc.samples, dlog)
    println("Final step")
    GranularScenes.viz_chain(c)
    marginal = marginalize(buffer(dlog), :obstacles)
    # HACK: save last chunk of train chain
    Gen_Compose.report_step!(dlog, c)
    return marginal
end

function marginalize(bfr, key)
    n = length(bfr)
    @show n
    @assert n > 0
    marginal = similar(bfr[1][key])
    fill!(marginal, 0.0)
    for i = 1:n
        datum = bfr[i][key]
        for j = eachindex(marginal)
            marginal[j] += datum[j]
        end
    end
    lmul!(1.0 / n, marginal)
end

function test(marginal, proc, query, out)
    _..., mp, var = proc.ddp_args
    p = setproperties(proc; ddp_args = (marginal, mp, var))
    dlog = JLD2Logger(p.samples, out)
    run_chain(p, query, p.samples + 1, dlog)
    return nothing
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

    for door = [1, 2]
        out_path = "/spaths/experiments/$(dataset)_$(model)/$(scene)_$(door)"

        base_p = joinpath(base_path, "$(scene)_$(door).json")
        train_room = load_scene(base_p)
        test_room = block_tile(train_room, tile)

        gm_params = QuadTreeModel(train_room;
                                  read_json(args["gm"])...,
                                  render_kwargs =
                                      Dict(:resolution => (256,256)))

        println("Saving results to: $(out_path)")

        train_query = query_from_params(train_room, gm_params)
        test_query = query_from_params(test_room, gm_params)

        train_proc_kwargs = read_json(args["proc"])
        test_proc_kwargs = deepcopy(train_proc_kwargs)
        test_proc_kwargs[:samples] = 10

        protocol = attention == :ac ?
            AdaptiveComputation() : UniformProtocol()
        train_proc_kwargs[:protocol] =
            test_proc_kwargs[:protocol] = protocol


        if granularity == :fixed
            train_proc_kwargs[:ddp] = fixed_granularity_cm
            train_proc_kwargs[:ddp_args] = (gm_params, 5) # HACK: hard coded
            test_proc_kwargs[:ddp] = generate_cm_from_ppd
            test_proc_kwargs[:ddp_args] = (gm_params,
                                           0.0) # 0 var forces max split
            # No split-merge moves
            train_proc_kwargs[:sm_budget] =
                test_proc_kwargs[:sm_budget] = 0
        else
            test_proc_kwargs[:ddp] = generate_cm_from_ppd
            test_proc_kwargs[:ddp_args] = (gm_params,
                                           0.0175)
        end
        train_proc = AdaptiveMH(; train_proc_kwargs...)
        test_proc = AdaptiveMH(; test_proc_kwargs...)

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
            c3_out = joinpath(out_path, "c3_$(c).jld2")
            outs = [c1_out, c2_out, c3_out]
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
                trained = train(train_proc, train_query, c1_out)
                test(trained, test_proc, test_query, c2_out)
                test(trained, test_proc, train_query, c3_out)
            end
            println("Chain $(c) complete")
        end
    end
    return nothing
end

main();
