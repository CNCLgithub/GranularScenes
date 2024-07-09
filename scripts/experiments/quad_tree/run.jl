using CSV
using JSON
using JLD2
using Rooms
using FileIO
# using Random
using ArgParse
using Gen_Compose
using GranularScenes
using Gen: get_retval
using LinearAlgebra: lmul!

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

function test(marginal, query, out)
    model_params = first(query.args)
    p = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                      protocol = AdaptiveComputation(),
                      ddp = generate_cm_from_ppd,
                      ddp_args = (marginal,
                                  model_params,
                                  0.0175),
                      samples = 10)
    dlog = JLD2Logger(p.samples, out)
    run_chain(p, query, p.samples + 1, dlog)
    return nothing
end


function main(c=ARGS)
    args = parse_commandline(c)
    base_path = "/spaths/datasets/$(dataset)/scenes"
    scene = args["scene"]

    println("Running inference on scene $(scene)")
    args["restart"] = true

    manifest = CSV.File(base_path * ".csv")
    tile = manifest[:tidx][scene]

    model = "ac"

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

        proc = AdaptiveMH(;read_json("$(@__DIR__)/attention.json")...,
                          protocol = AdaptiveComputation())
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
                trained = train(proc, train_query, c1_out)
                test(trained, test_query, c2_out)
                test(trained, train_query, c3_out)
            end
            println("Chain $(c) complete")
        end
    end
    return nothing
end

main();
