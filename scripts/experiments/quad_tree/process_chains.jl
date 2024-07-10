using CSV
using JLD2
using JSON
using FileIO
using PyCall
using DataFrames
using DataFrames: select
using Statistics
using LinearAlgebra
using Gen: logsumexp


np = pyimport("numpy")

# assuming scenes are 32x32
dataset = "path_block_2024-03-14"
model = "ac"
scenes = 1:6
exp_path = "/spaths/experiments/$(dataset)_$(model)"
burnin = 50
chains = 30
max_step = 100
steps = max_step - burnin

# Data to extract
qt_dims = (chains, steps, 16, 16)
att = Array{Float32, 4}(undef, qt_dims)
geo = Array{Float32, 4}(undef, qt_dims)
pmat = Array{Bool, 4}(undef, qt_dims)
gran = Array{UInt8, 4}(undef, qt_dims)
img = Array{Float32, 4}(undef, (chains, 256, 256, 3))
_img = Array{Float32, 3}(undef, (256, 256, 3))

function aggregate_chains!(df::DataFrame, scene::Int, door::Int, mode::Symbol)
    path = "$(exp_path)/$(scene)_$(door)"
    @show path
    start_s = mode == :c1 ? burnin : 1
    max_s = steps
    for c = 1:chains
        c_path = "$(path)/$(mode)_$(c).jld2"
        fill!(_img, 0.0)
        score::Float32 = -Inf
        likelihood::Float32 = -Inf
        prior::Float32 = -Inf
        jldopen(c_path, "r") do file
            @show file["current_idx"]
            max_s = min(file["current_idx"], max_step)
            # integrate across samples
            for s = start_s:max_s
                # @show s
                data = file["$(s)"]
                # att[c, s, :, :] = data[:attention]
                # geo[c, s, :, :] = data[:obstacles]
                # pmat[c, s, :, :] = data[:path]
                # gran[c, s, :, :] = data[:granularity]
                # _img[:, :, :] += data[:img]

                score = logsumexp(score, data[:score])
                likelihood = logsumexp(likelihood, data[:likelihood])
            end
            rmul!(_img, 1.0 / (max_s - start_s))
            img[c, :, :, :] = _img
        end
        nsteps = max_s - start_s
        @show nsteps
        logn = log(nsteps)
        score -= logn
        likelihood -= logn
        prior = score - likelihood
        push!(df, [scene, door, c, mode, score, likelihood, prior])
    end
    # np.savez("$(path)_$(mode)_aggregated.npz",
    #          att = att,
    #          geo=geo,
    #          pmat=pmat,
    #          gran=gran,
    #          img=img,
    #          )
    # @show max_s
    return nothing
end

function main()
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    result = DataFrame(
        scene = UInt8[],
        door = UInt8[],
        chain = UInt8[],
        mode = Symbol[],
        score = Float32[],
        likelihood = Float32[],
        prior = Float32[]
    )
    # for r in eachrow(df)
    for scene = scenes
        for door = [1, 2]
            aggregate_chains!(result, scene, door, :c1)
            aggregate_chains!(result, scene, door, :c2)
            aggregate_chains!(result, scene, door, :c3)
            # aggregate_chains!(result, scene, door, :un)
        end
    end

    sort!(result, Cols(:scene, :door, :chain, :mode))
    display(result)

    # diffs = combine(groupby(filter(:mode => !=(:c3), result), Cols(:scene, :door, :chain)),
    #                 :score => (x -> abs(x[2] - x[1])) =>
    #                     :diff_score)
    # diffs = combine(groupby(diffs, Cols(:scene, :door)),
    #                 :diff_score => mean =>
    #                     :avg_diff_score)

    wide = unstack(select(result, Cols(:scene, :door, :mode, :chain, :score)),
                          :mode, :score)

    # wide = unstack(select(result, Cols(:mode, :scene, :door, :chain, :score)),
    #                       :mode, :score)
    transform!(wide,
               [:c2, :c1] => ByRow(-) => :delta_change,
               [:c3, :c1] => ByRow(-) => :delta_same,
               # [:c1, :c2] => ByRow(abs ∘ -) => :delta_change,
               # [:c1, :c3] => ByRow(abs ∘ -) => :delta_same,
               )
    transform!(wide,
               [:delta_change, :delta_same] => ByRow(-) => :w,
               )
    transform!(wide,
               :w => ByRow(w -> exp(min(0., w))) => :p,
               )

    display(wide)

    diffs = combine(groupby(wide, Cols(:scene, :door)),
                    :delta_change => mean,
                    :delta_same => mean,
                    :w => mean,
                    :p => mean,
                    # :delta_change => std,
                    # :delta_same => std,
                    )

    display(diffs)
    CSV.write("/spaths/experiments/$(dataset)_$(model)_chains.csv", result)
    return result
end

result = main();
