using CSV
using JLD2
using JSON
using FileIO
using PyCall
using DataFrames
using Statistics
using LinearAlgebra
using Gen: logsumexp


np = pyimport("numpy")

# assuming scenes are 32x32
dataset = "path_block_2024-03-14"
exp_path = "/spaths/experiments/$(dataset)"
burnin = 1
chains = 20
max_step = 200
steps = max_step - burnin

# Data to extract
qt_dims = (chains, steps, 16, 16)
att = Array{Float32, 4}(undef, qt_dims)
geo = Array{Float32, 4}(undef, qt_dims)
pmat = Array{Bool, 4}(undef, qt_dims)
gran = Array{UInt8, 4}(undef, qt_dims)
img = Array{Float32, 4}(undef, (chains, 256, 256, 3))
_img = Array{Float32, 3}(undef, (256, 256, 3))

function aggregate_chains!(df::DataFrame, scene::Int, door::Int, model::Symbol)
    path = "$(exp_path)/$(scene)_$(door)"
    # @show path
    score::Float32 = -Inf
    likelihood::Float32 = -Inf
    train::Float32 = -Inf
    test::Float32 = -Inf
    for c = 1:chains
        c_path = "$(path)/$(model)_$(c).jld2"
        fill!(_img, 0.0)
        jldopen(c_path, "r") do file
            for s = 1:steps
                # @show s
                data = file["$(s + burnin)"]
                att[c, s, :, :] = data[:attention]
                geo[c, s, :, :] = data[:obstacles]
                pmat[c, s, :, :] = data[:path]
                gran[c, s, :, :] = data[:granularity]
                _img[:, :, :] += data[:img]

                score = logsumexp(score, data[:score])
                likelihood = logsumexp(likelihood, data[:likelihood])
                train = logsumexp(train, data[:train])
                test = logsumexp(test, data[:test])
            end
            rmul!(_img, 1.0 / steps)
            img[c, :, :, :] = _img
        end
    end
    np.savez("$(path)_$(model)_aggregated.npz",
             att = att,
             geo=geo,
             pmat=pmat,
             gran=gran,
             img=img,
             )

    logn = log(chains * steps)
    score -= logn
    likelihood -= logn
    train -= logn
    test -= logn
    push!(df, [model, scene, door, score, likelihood, train, test])
    return nothing
end

function main()
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    result = DataFrame(
        model = Symbol[],
        scene = UInt8[],
        door = UInt8[],
        score = Float32[],
        likelihood = Float32[],
        train = Float32[],
        test = Float32[]
    )
    # for r in eachrow(df)
    for scene = 1:6
        for door = [1, 2]
            aggregate_chains!(result, scene, door, :ac)
            # aggregate_chains!(result, scene, door, :un)
        end
    end
    display(result)
    CSV.write("/spaths/experiments/$(dataset)_chains.csv", result)
    return nothing
end

main();
