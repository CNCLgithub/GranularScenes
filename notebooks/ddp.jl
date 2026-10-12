### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000001
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ d1a2b3c4-0002-4000-8000-000000000002
begin
	using HDF5, Flux, Optimisers, Random, Statistics, Printf, cuDNN, FileIO, ImageCore, JLD2
	using Flux: state
	import CUDA

	using PlutoUI
	using Gen: get_retval
	using Rooms
	using GranularScenes
	import GranularScenes as GS
end

# ╔═╡ d1a2b3c4-0003-4000-8000-000000000003
html"""
<style>
    @media screen {
        main {
            margin: 0 auto;
            max-width: 2500px;
            padding-left: max(100px, 10%);
            padding-right: max(100px, 10%);
        }
    }
	pluto-output {
    font-size: 1.2em; /* Adjust base text size */
    font-family: "Inter";
	}

pluto-output h1 {
    font-size: 2.5rem; /* Adjust header sizes */
	font-family: "Inter";
}

pluto-output h2 {
    font-size: 3.0rem;
}

cm-editor .cm-scroller,
.cm-editor .cm-content {
    font-family: "Fira Code", monospace !important;
    font-size: 18px !important; /* Adjust size here */
}
</style>
"""

# ╔═╡ d1a2b3c4-0004-4000-8000-000000000004
md"""
# DDP + AdaptiveMH agent

- **Part 1** — the Flux depth-VAE + occupancy decoder (ported from `scripts/nn/test_ddp.jl`).
- **Part 2** — an agent initialized with the DDP via the `AdaptiveMH` constructor (`ddp` kwarg).
- **Part 3** — chain dynamics visualization.
"""

# ╔═╡ d1a2b3c4-0005-4000-8000-000000000005
# ----------------------------------------------------------------- config ---
begin
	DATA     = get(ENV, "DDP_DATA",     "/spaths/datasets/ddp_train_11f_32x32.hdf5")
	VAE_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxvae_e30.weights.jld2")
	OCC_CKPT = get(ENV, "DDP_OCC_CKPT", "/spaths/checkpoints/fluxocc_e30.weights.jld2")  # NB: test_ddp.jl had a DDP_VAE_CKPT typo here

	const DATA_DEPTH_MIN = 3.971077f0
	const DATA_DEPTH_MAX = 10.181246f0

	# DDP -> quadtree parameters
	QT_VAR       = 0.01      # split threshold on node footprint std
	QT_MAX_LEVEL = 3         # 8x8 max leaves
	QT_MAX_DEPTH = 4

	# inference chain
	N_CHAIN_STEPS = 60
	VIZ_EVERY     = 5
	MARGIN_SIZE   = 50
end

# ╔═╡ d1a2b3c4-0007-4000-8000-000000000007
begin
	@printf "Loading %s ...\n" DATA
	X, O = load_data(DATA)
	N = size(X, 4)
	@printf "dataset: %s, N=%d\n" size(X) N
	@printf "occupied fraction: %.4f (predict-prior MSE baseline: %.6f)\n" mean(O .> 0.5f0) mean((mean(O) .- O).^2)
end

# ╔═╡ d1a2b3c4-0008-4000-8000-000000000008
begin
	use_gpu = CUDA.functional()
	dev = use_gpu ? (x -> fmap(CUDA.cu, x)) : Flux.cpu
	use_gpu && @info "GPU: " * CUDA.name(CUDA.device())

	# NOTE: qualified — after the inference include (cell below), unqualified
	# DataDrivenState/occ_grid resolve to the src/inference variants.
	dds = GranularScenes.DataDrivenState(
		;
		device = dev,
		vae_path = VAE_CKPT,
		occ_path = OCC_CKPT,
		var = 0.01
	)

	@info "Loaded frozen VAE from " VAE_CKPT
	@info "Loaded frozen OCC from " OCC_CKPT
end


# ╔═╡ d1a2b3c4-0011-4000-8000-000000000011
md"""
# Part 2 — agent initialized with the DDP
"""

# ╔═╡ d1a2b3c4-0013-4000-8000-000000000013
# --------------------------------------------------- scene & model ----------
begin
    dataset = "window-0.1/2025-02-05_vifdDO"

    function load_room(idx::Int)

        base_path = "/spaths/datasets/$(dataset)/scenes"
        path = joinpath(base_path, "$(idx).json")
        local base_s
        open(path, "r") do f
            base_s = JSON.parse(f)
        end
        from_json(GridRoom, base_s)
    end
end

# ╔═╡ d1a2b3c4-0014-4000-8000-000000000014
# --------------------------------------------------- DDP proposal -----------
# Top-level (src/inference/procedures/ddp/ddp.jl) DataDrivenState: same Flux
# checkpoints as Part 1, without the device kwarg.
begin
	ddp = DataDrivenState(;
						  vae_path = VAE_CKPT,
						  occ_path = OCC_CKPT,
						  var = QT_VAR)

	ddp_cm = generate_cm_from_ddp(ddp, img, QT_MAX_LEVEL;
								  max_depth = QT_MAX_DEPTH)
	display_mat(project_qt(ddp_cm))
end

# ╔═╡ d1a2b3c4-0015-4000-8000-000000000015
# --------------------------------------------------- query & procedure ------
begin
	q0 = query_from_params(room, gm_params)

	proc = AdaptiveMH(;
					  rw_budget = 20,
					  sm_budget = 20,
					  ddp = (_...) -> deepcopy(ddp_cm),
					  protocol = AdaptiveComputation())

	# NOTE: the top-level train! (src/planning.jl) is used below; GS.train! does
	# not exist — run.jl's GS.train!/GranularScenes.viz_chain calls are stale.
	n_total = N_CHAIN_STEPS + 2 * MARGIN_SIZE

	c0 = initialize_chain(proc, q0, n_total)
	l0 = MemLogger(MARGIN_SIZE)
end

# ╔═╡ d1a2b3c4-0016-4000-8000-000000000016
begin
	train!(c0, l0, N_CHAIN_STEPS)
	viz_chain(l0)
end

# ╔═╡ d1a2b3c4-0017-4000-8000-000000000017
md"""
# Part 3 — chain dynamics
"""

# ╔═╡ 2a3ac72c-a2cb-4ef4-8870-093996cb3b2a
# Runs the chain one step at a time, recording the log score, the number of
# quadtree leaves, and a rendered frame of the current quadtree geometry.
function chain_dynamics(c::AMHChain, l, steps::Int; viz_every::Int = 5)
	frames = Any[]
	scores = Float64[]
	nleaf  = Int[]
	for i = 1:steps
		train!(c, l, 1)
		push!(scores, Gen.get_score(c.state))
		qt = get_retval(c.state)
		push!(nleaf, length(qt.leaves))
		if i % viz_every == 0
			push!(frames, draw_mat(project_qt(qt), true, colorant"black", colorant"blue"))
		end
	end
	frames, scores, nleaf
end

# ╔═╡ de289c12-7f9a-4018-800d-5e6486b12416
frames, scores, nleaf = chain_dynamics(c0, l0, N_CHAIN_STEPS; viz_every = VIZ_EVERY)

# ╔═╡ d1a2b3c4-0019-4000-8000-000000000019
@bind frame_i Slider(1:length(frames); default = length(frames), show_value = true)

# ╔═╡ d1a2b3c4-0020-4000-8000-000000000020
# browse the recorded quadtree states with the slider above
Table(frames[max(1, frame_i):end])

# ╔═╡ d1a2b3c4-0021-4000-8000-000000000021
# score trajectory: higher (closer to 0) = better posterior fit
let s = scores
	mn, mx = extrema(s)
	rng01 = max(mx - mn, eps())
	blocks = "▁▂▃▄▅▆▇█"
	spark = join(blocks[clamp(round(Int, (v - mn) / rng01 * 7) + 1, 1, 8)] for v in s)
	md"score over $(length(s)) steps: min $(Printf.@sprintf "%.3f" mn), max $(Printf.@sprintf "%.3f" mx): `$spark`"
end

# ╔═╡ Cell order:
# ╠═d1a2b3c4-0001-4000-8000-000000000001
# ╠═d1a2b3c4-0002-4000-8000-000000000002
# ╟─d1a2b3c4-0003-4000-8000-000000000003
# ╟─d1a2b3c4-0004-4000-8000-000000000004
# ╠═d1a2b3c4-0005-4000-8000-000000000005
# ╠═d1a2b3c4-0006-4000-8000-000000000006
# ╠═d1a2b3c4-0007-4000-8000-000000000007
# ╠═d1a2b3c4-0008-4000-8000-000000000008
# ╠═d1a2b3c4-0009-4000-8000-000000000009
# ╟─d1a2b3c4-0011-4000-8000-000000000011
# ╠═d1a2b3c4-0013-4000-8000-000000000013
# ╠═d1a2b3c4-0014-4000-8000-000000000014
# ╠═d1a2b3c4-0015-4000-8000-000000000015
# ╠═d1a2b3c4-0016-4000-8000-000000000016
# ╠═d1a2b3c4-0017-4000-8000-000000000017
# ╠═2a3ac72c-a2cb-4ef4-8870-093996cb3b2a
# ╠═de289c12-7f9a-4018-800d-5e6486b12416
# ╠═d1a2b3c4-0019-4000-8000-000000000019
# ╠═d1a2b3c4-0020-4000-8000-000000000020
# ╠═d1a2b3c4-0021-4000-8000-000000000021
