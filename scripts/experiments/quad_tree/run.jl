using CSV
using JSON
using JLD2
using Rooms
using FileIO
using ArgParse
using Accessors
using DataFrames
using Gen_Compose
using UnicodePlots
using GranularScenes
using StaticArrays: SVector

import GranularScenes as GS

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
    (GridRoom(r, d), cidx)
end

function loc_error(gt::CartesianIndex{2}, qt::QuadTree, ws::Array{Float64})
    ml = GS.max_leaves(qt)
    li = LinearIndices((ml, ml))[gt]
    idx = GS.idx_to_node_space(li, ml)
    n = GS.node(GS.traverse_qt(qt, idx))
    weight_at_gt = ws[qt.mapping[GS.tree_idx(n)]]
    max_weight = maximum(ws)
    le = max_weight - weight_at_gt
    return le
    # @show le
    # @show GS.area(n)
    # le * GS.area(n)
end
function loc_error(a::CartesianIndex{2}, b::SVector{2, Float64}, max_kl::Float64)
    dx = a.I[1] - b[1]
    dy = a.I[2] - b[2]
    sqrt(dx^2 + dy^2) # / max(max_kl, 0.02)
end
function loc_error(a::CartesianIndex{2}, b::CartesianIndex{2}, max_kl::Float64)
    dx = a.I[1] - b.I[1]
    dy = a.I[2] - b.I[2]
    sqrt(dx^2 + dy^2) # / max(max_kl, 0.02)
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

    doors = [1,2]

    results = DataFrame(
        :door => Int64[],
        :step => Int64[],
        :loc_error => Float64[],
    )

    for door = doors
        out_path = "/spaths/experiments/flicker_$(dataset)_$(model)/$(scene)_$(door)"
        if args["reverse"]
            out_path *= "_reversed"
        end

        base_p = joinpath(base_path, "$(scene)_$(door).json")
        room1 = load_scene(base_p)
        room2, blocked_idx = block_tile(room1, tile)

        gm_params = QuadTreeModel(room1;
                                  read_json(args["gm"])...,
                                  render_kwargs =
                                      Dict(:resolution => (128,128)))

        println("Saving results to: $(out_path)")

        dargs = read_json(args["decision"])

        q = query_from_params(room1, room2, gm_params)


        model_params = first(q.args)
        img = GranularScenes.render(model_params.renderer, room1)
        ddp_params = DataDrivenState(;config_path = args["ddp"],
                                     var = 0.225)
        ddp_cm = generate_cm_from_ddp(ddp_params, img, model_params, 3, 4)


        proc_kwargs = read_json(args["proc"])

        # HACK: Instead, rework `AdaptiveMH` or right-hand of query
        proc_kwargs[:ddp]  = (_...) -> deepcopy(ddp_cm)

        protocol = attention == :ac ?
            AdaptiveComputation() : UniformProtocol()
        proc_kwargs[:protocol] = protocol


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
        proc = AdaptiveMH(; proc_kwargs...)

        try
            isdir("/spaths/experiments/$(dataset)_$(model)") ||
                mkpath("/spaths/experiments/$(dataset)_$(model)")
            isdir(out_path) || mkpath(out_path)
        catch e
            println("could not make dir $(out_path)")
        end

        # how many chains to run
        for chain_idx = 1:args["chain"]
            # Random.seed!(c)
            # c1_out = joinpath(out_path, "c1_$(c).jld2")
            # c2_out = joinpath(out_path, "c2_$(c).jld2")
            # outs = [c1_out, c2_out]
            complete = false
            # if any(isfile, outs)
            #     println("Record(s) found for chain: $(c)")
            #     if args["restart"]
            #         println("restarting")
            #         foreach(o -> isfile(o) && rm(o), outs)
            #     else
            #         complete = all(isfile, outs)
            #     end
            # end
            if !complete
                println("Starting chain $(chain_idx)")
                chain_steps = dargs[:epoch_steps] * dargs[:epochs]
                log = MemLogger(dargs[:margin_size])
                c = Gen_Compose.initialize_chain(proc, q, chain_steps)
                e = 1; le = 1.0;
                while e < dargs[:epochs] # && max_kl < dargs[:kl_threshold]
                    (qt, ws, _) =
                        search_step!(c, log,
                                     dargs[:epoch_steps],
                                     dargs[:search_weight])
                    le = loc_error(blocked_idx, qt, ws)
                    # println("Step $(e), LE: $(le)")
                    push!(results, (door, e * dargs[:epoch_steps], le))

                    e += 1
                end
                GranularScenes.viz_chain(log)
                # GranularScenes.display_mat(klm)
            end
            println("Chain $(chain_idx) complete")
        end
    end
    # filter!(:step => >(20), results) # remove burnin
    @show extrema(results.loc_error)
    left_df, right_df = groupby(results, :door)
    plt = lineplot(left_df.step, left_df.loc_error; name = "Left door",
                   ylim = extrema(results.loc_error),
                   )
    lineplot!(plt, right_df.step, right_df.loc_error; name = "Right door")
    display(plt)
    display(last(left_df, 3))
    display(last(right_df, 3))
    return nothing
end

main();
