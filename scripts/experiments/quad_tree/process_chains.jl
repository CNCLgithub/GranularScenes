using CSV
using JLD2
using JSON
using FileIO
using PyCall
using DataFrames
using Statistics
using LinearAlgebra


np = pyimport("numpy")

# assuming scenes are 32x32
dataset = "path_block_2024-03-14"
burnin = 25
chains = 3
max_step = 150
steps = max_step - burnin

# Data to extract
qt_dims = (chains, steps, 16, 16)
att = Array{Float32, 4}(undef, qt_dims)
geo = Array{Float32, 4}(undef, qt_dims)
pmat = Array{Bool, 4}(undef, qt_dims)
gran = Array{UInt8, 4}(undef, qt_dims)
img = Array{Float32, 4}(undef, (chains, 256, 256, 3))
_img = Array{Float32, 3}(undef, (256, 256, 3))

function aggregate_chains(path::String, model::String)
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

    return nothing
end

function main()
    exp_path = "/spaths/experiments/$(dataset)"
    df = DataFrame(CSV.File("/spaths/datasets/$(dataset)/scenes.csv"))
    # for r in eachrow(df)
    for scene = 1:3
        for door = [1, 2]
            scene_path = "$(exp_path)/$(scene)_$(door)"
            @show scene_path
            aggregate_chains(scene_path, "ac")
            aggregate_chains(scene_path, "un")

            # shift_path = "$(exp_path)/$(r.id)_$(r.door)_furniture_$(r.move)"
            # @show shift_path
            # aggregate_chains(shift_path)
        end
    end
    return nothing
end

main();
