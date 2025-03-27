using CSV
using JSON
using FileIO
using DataFrames


dataset = "window-0.1/2025-02-05_vifdDO"
model = "ac_multi"
scenes = 1:10
exp_path = "/spaths/experiments/$(dataset)_$(model)"

function main()

    results = DataFrame(
        :gt => String[],
        :scene => Int64[],
        :door => String[],
        :chain => Int64[],
        :pr_same => Float64[],
        :pr_change => Float64[],
        :ratio => Float64[],
    )

    for scene = scenes, door = [1, 2]

        subpath = "$(exp_path)/$(scene)_$(door)"
        chains = readdir(subpath, join=true)
        filter!(x -> endswith(x, ".csv"), chains)
        for chain_path = chains
            part = DataFrame(CSV.File(chain_path))
            append!(results, part)
        end
    end

    display(results)
    CSV.write("$(exp_path)_chain_summaries.csv", results)

end

main();
