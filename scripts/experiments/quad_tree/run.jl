using CSV
using JSON
using JLD2
using Rooms
using FileIO
using ArgParse
using DataFrames
using Gen_Compose
using GranularScenes
using Gen: get_score
import GranularScenes as GS

function parse_commandline(c)
    s = ArgParseSettings()

    @add_arg_table! s begin

        "--dataset"
        help = "Trial dataset"
        arg_type = String
        default = "window-0.1/2025-02-05_vifdDO"

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

# NOTE: This is needed due to the fact that the window-0.1/* datasets
# do not store wall information; the leads to weird depth map values
# and can throw off the DDP VAE.
function fix_room(r::GridRoom, ent::Int, door::Int)
    template = GridRoom((16, 16), (16., 16.), [ent], [door])
    # remove wall near camera
    d = data(template)
    d[:, 1:2] .= floor_tile
    for i = eachindex(data(r))
        if data(r)[i] == obstacle_tile
            d[i] = obstacle_tile
        end
    end
    GridRoom(template, d)
end

function load_scene(path::String, ent::Int, door::Int)
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    base = from_json(GridRoom, base_s)
    fix_room(base, ent, door)
end

function block_tile(r::GridRoom, tidx::Int)
    tidx == 0 && return r
    d = deepcopy(data(r))
    d[tidx] = obstacle_tile
    cidx = CartesianIndices(Rooms.steps(r))[tidx]
    println("Blocked tile $(tidx) at location $(cidx)")
    GridRoom(r, d)
end

function init_results()
    DataFrame(
        :gt => Symbol[],
        :scene => Int64[],
        :door => Symbol[],
        :chain => Int64[],
        :pr_same => Float64[],
        :pr_change => Float64[],
        :ratio => Float64[],
    )
end

function run_scene(args)

    dataset = args["dataset"]
    base_path = "/spaths/datasets/$(dataset)/scenes"

    scene = args["scene"]

    println("Running inference on scene $(scene)")
    args["restart"] = true

    manifest = CSV.File(base_path * ".csv")
    tile = manifest[:blocked][scene]

    attention = args["attention"]
    granularity = args["granularity"]

    model = "$(attention)_$(granularity)"

    # doors = [2]
    # door_tiles = [252]
    doors = [2,1]
    door_tiles = [252, 244]
    ent_tiles = [9, 8]

    results = DataFrame(
        :gt => Symbol[],
        :scene => Int64[],
        :door => Symbol[],
        :chain => Int64[],
        :pr_same => Float64[],
        :pr_change => Float64[],
        :ratio => Float64[],
    )

    for (door, door_tile, ent_tile) = zip(doors, door_tiles, ent_tiles)
        out_path = "/spaths/experiments/$(dataset)_$(model)/$(scene)_$(door)"

        base_p = joinpath(base_path, "$(scene).json")
        room1 = load_scene(base_p, door_tile, ent_tile)
        room2 = block_tile(room1, tile)

        gm_params = QuadTreeModel(room1;
                                  read_json(args["gm"])...,
                                  render_kwargs =
                                      Dict(:resolution => (128,128)))

        println("Saving results to: $(out_path)")

        dargs = read_json(args["decision"])

        q0 = query_from_params(room1, gm_params)
        q_change = extend_query(q0, room2)
        q_same = extend_query(q0, room1)

        img = GranularScenes.render(gm_params.renderer, room1)
        ddp_params = DataDrivenState(;config_path = args["ddp"],
                                     var = 0.13)
        ddp_cm = generate_cm_from_ddp(ddp_params, img, gm_params, 3, 4)


        proc_kwargs = read_json(args["proc"])

        # HACK: Instead, rework `AdaptiveMH` or right-hand of query
        proc_kwargs[:ddp]  = (_...) -> deepcopy(ddp_cm)

        protocol = attention == :ac ?
            AdaptiveComputation() : UniformProtocol()
        proc_kwargs[:protocol] = protocol


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

            chain_path = "$(out_path)/chain_summary_$(chain_idx).csv"

            if isfile(chain_path)
                if args["restart"]
                    println("Redoing chain $(chain_idx)...")
                    rm(chain_path)
                else
                    println("Chain $(chain_idx) already completed.")
                    continue
                end
            end

            println("Starting train chain $(chain_idx)")
            l0 = MemLogger(dargs[:margin_size])
            total_steps = dargs[:train_steps] + 2 * dargs[:test_steps]
            # Infer Pr(S | X0)
            c0 = Gen_Compose.initialize_chain(proc, q0, total_steps)
            GS.train!(c0, l0, dargs[:train_steps])
            # TODO: record something about training
            println("Chain after X_0")
            GranularScenes.viz_chain(l0)

            results = init_results()
            for (gt, q1) = zip([:change, :same], [q_change, q_same])
                c1, l1 = fork(c0, l0)
                # Evaluate Pr(X1 = X0 | S, Change = F)
                # NOTE: This is cached into the c0 chain
                # without impacting the acceptance ratio
                prob_same = test_same(c1, l1, q1)

                # Evaluate Pr(X1 | S, Change = T, Loc = *)
                extend_chain!(c1, q1)
                prob_change = test_change!(c1, l1, dargs[:test_steps])

                # Decision
                dweight = prob_change - prob_same

                push!(results,
                      (gt, scene, door == 2 ? :on : :off, chain_idx,
                       prob_same, prob_change, dweight))

            end
            println("Chain $(chain_idx) complete")
            display(results)
            CSV.write(chain_path, results)
        end
    end
    return nothing
end


function main(c=ARGS)
    args = parse_commandline(c)
    for scene = 2:2
        args["scene"] = scene
        run_scene(args)
    end
end

main();
