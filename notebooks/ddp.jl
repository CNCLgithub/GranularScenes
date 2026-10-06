### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000001
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000002
begin

	using Lux, Reactant, Enzyme, Optimisers, MLUtils, Random, Printf, Statistics
	using PlutoUI
	using MLUtils
	# using Plots
	using Revise
	using GranularScenes
	import GranularScenes: DepthOccVAE, vae_loss_function, occ_loss_function,
		load_depth_dataset, DepthOccDataset, plot_vae_panels
end

# ╔═╡ 665b3437-0ce9-4818-ad14-32c4a433ee4e
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


# ╔═╡ d1a2b3c4-0001-4000-8000-000000000003
md"""
# Depth-Map VAE + Occupancy Decoder (two-stage)

Stage 1 trains the VAE (encoder + depth decoder) on 256×256 depth maps.
Stage 2 freezes the VAE and trains the occupancy decoder (32×32 latent → 16×16
occupancy grid). Each epoch displays the test-grid visualization for the first
3 scenes.
"""

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000004
md"## Hyperparameters"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000005
hyper = (
	batchsize = 32,
	seed = 0,
	vae_epochs = 5,
	occ_epochs = 20,
	weight_decay = 1.0f-5,
	learning_rate = 1.0f-3,
	occ_learning_rate = 1.0f-3,
	β = 1.0f0,
	max_num_filters = 64,
	image_shape = (256, 256, 1),
);


# ╔═╡ d1a2b3c4-0001-4000-8000-000000000006
rng = Xoshiro(); Random.seed!(rng, hyper.seed);

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000029
# ╠═╡ disabled = true
#=╠═╡
begin
	# TensorBoard logs land in ../tblogs/<run-name>; view with
	#   tensorboard --logdir ../tblogs
	logger = TBLogger(joinpath(@__DIR__, "..", "tblogs", Printf.format(
		Printf.Format("run_%Y%m%d_%H%M%S"), now())); min_level=Logging.Info)
end
  ╠═╡ =#

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000007
md"## Model"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000008
model = DepthOccVAE(rng; 
					image_shape=hyper.image_shape, max_num_filters=hyper.max_num_filters)

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000009
begin
	xdev = reactant_device(; force=true)
	cdev = cpu_device()
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000010
ps, st = Lux.setup(rng, model) |> xdev

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000011
Printf.@printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e6)

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000012
md"## Data"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000013
# train_dataloader = load_depth_dataset(; batchsize=hyper.batchsize) |> xdev
dataset = DDPSDataset("/spaths/datasets/ddp_train_11f_32x32.hdf5")

# ╔═╡ 24af709c-a124-4671-86d0-41cbb4c82a67
train_dataloader = DataLoader(dataset, batchsize=hyper.batchsize; shuffle=true, partial=false);

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000014
md"## Test set (first 3 scenes)"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000015
begin
	test_X = rand(Float32, hyper.image_shape..., 3)
	test_O = Float32.(rand(Float32, 16, 16, 1, 3) .> 0.5)
end;

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000030
begin
	# Encode the test scenes with the current encoder and return the latent
	# grids (32, 32, 1, 3) for μ and logσ², on the CPU for logging/plotting.
	function test_latents(parameters, states)
		st_enc = Lux.testmode(states.encoder)
		(μ, logσ², _), _ = model.encoder(test_X, parameters.encoder, st_enc)
		return cdev(μ), cdev(logσ²)
	end

	"helper: test_latents"
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000016
md"## Visualization"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000018
md"## Stage 1: VAE training"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000019
begin
	opt = AdamW(; eta=hyper.learning_rate, lambda=hyper.weight_decay)
	train_state = Training.TrainState(model, ps, st, opt)
end;

# ╔═╡ 9d2ce5dc-e423-41c6-9353-e1a3c176622a
viz_forward = @compile donated_args=:none model(xdev(test_X), train_state.parameters, Lux.testmode(train_state.states));

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000017
begin
	@printf "=== Initial state (before training) ===\n"
    (x_rec, occ, μ, logσ²), _ = viz_forward(xdev(test_X), train_state.parameters, Lux.testmode(train_state.states))
    panels = plot_vae_panels(test_X, test_O, Array(x_rec), Array(occ))
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000020
begin
	for epoch in 1:hyper.vae_epochs
		loss_total = 0.0f0
		total_samples = 0
		start_time = time()

		for (i, (X, _)) in enumerate(train_dataloader)
			X_dev = xdev(X)
			(_, loss, _, train_state) = Training.single_train_step!(
				AutoEnzyme(),
				(m, p, s, x) -> vae_loss_function(m, p, s, x; β=hyper.β),
				X_dev,
				train_state;
				return_gradients=Val(false),
				compile_options=Reactant.CompileOptions(donated_args=:none),
			)
			loss_total += loss
			total_samples += size(X, ndims(X))
		end

		@printf "[stage 1] Epoch %d, Train Loss: %.7f, Time: %.4fs\n" epoch (loss_total / length(train_dataloader)) (time() - start_time)
		# with_logger(logger) do
		# 	log_value(0, "train/vae_loss", loss_total / length(train_dataloader))
		# 	# Latent state per test scene: 32x32 heatmap of μ and logσ².
		# 	(μ, logσ²) = test_latents(train_state.parameters, train_state.states)
		# 	for k in axes(μ, 4)
		# 		log_image(0, "z/mean/scene$(k)", μ[:, :, 1, k])   # 32x32 heatmap
		# 		log_image(0, "z/logvar/scene$(k)", logσ²[:, :, 1, k])
		# 	end
		# end
	end
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000021
md"### Visualization after stage 1"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000022
begin
    (_x_rec, _occ, _μ, _logσ²), _ = viz_forward(xdev(test_X), train_state.parameters, Lux.testmode(train_state.states))
    panels_s1 = plot_vae_panels(test_X, test_O, Array(_x_rec), Array(_occ))
	panels_s1
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000023
md"## Stage 2: Occupancy decoder training (frozen VAE)"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000024
# ╠═╡ disabled = true
#=╠═╡
begin
	occ_model = model.occ_decoder
	occ_ps = train_state.parameters.occ_decoder
	occ_st = train_state.states.occ_decoder
	occ_opt = AdamW(; eta=hyper.occ_learning_rate, lambda=hyper.weight_decay)
	occ_train_state = Training.TrainState(occ_model, occ_ps, occ_st, occ_opt)
end
  ╠═╡ =#

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000025
begin
	# Cache z for all training samples with the frozen encoder (test mode).
	zs = nothing
	st_enc = Lux.testmode(train_state.states.encoder)
	for (i, (X, _)) in enumerate(train_dataloader)
		(z, _, _), _ = model.encoder(X, train_state.parameters.encoder, st_enc)
		zs = zs === nothing ? cdev(z) : cat(zs, cdev(z); dims=4)
	end
	z_cache = xdev(zs)
	# Note: with shuffle=true the batch indices are not aligned with the cache.
	# For exact index alignment use shuffle=false in a stage-2 loader.
end

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000026
#=╠═╡
begin
	for epoch in 1:hyper.occ_epochs
		loss_total = 0.0f0
		total_samples = 0
		start_time = time()

		for (i, (X, O)) in enumerate(train_dataloader)
			idxs = ((i-1)*hyper.batchsize+1):min(i*hyper.batchsize, size(z_cache, 4))
			z = z_cache[:, :, :, idxs]
			(_, loss, _, occ_train_state) = Training.single_train_step!(
				AutoEnzyme(),
				occ_loss_function,
				(z, O),
				occ_train_state;
				return_gradients=Val(false),
			)
			loss_total += loss
			total_samples += size(O, ndims(O))
		end

		@printf "[stage 2] Epoch %d, Train Loss: %.7f, Time: %.4fs\n" epoch (loss_total / length(train_dataloader)) (time() - start_time)

		# Show the test-grid visualization every epoch (once per epoch, not per
		# batch). Rows are scenes; columns are | GT depth | Recon Depth | GT occ
		# | Recon Occ |.
		(panels, (x_rec, occ, μ, logσ²)) = viz_test_grid(model, train_state.parameters,
			Lux.testmode(train_state.states), test_X, test_O;
			filepath=joinpath(@__DIR__, "..", "viz", "ddp_epoch$(lpad(epoch, 3, '0')).png"))
		display(panels)

		with_logger(logger) do
			log_value(0, "train/occ_loss", loss_total / length(train_dataloader))
			# Log per-scene reconstructions for cross-epoch comparison.
			x_rec_cpu, occ_cpu = cdev(x_rec), cdev(occ)
			for k in axes(occ, 4)
				log_image(0, "recon/depth/scene$(k)", x_rec_cpu[:, :, 1, k])
				log_image(0, "recon/occ/scene$(k)", occ_cpu[:, :, 1, k])
			end
		end
	end
end
  ╠═╡ =#

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000027
md"### Final visualization"

# ╔═╡ d1a2b3c4-0001-4000-8000-000000000028
begin
	(panels_final, _) = viz_test_grid(model, train_state.parameters,
		Lux.testmode(train_state.states), test_X, test_O)
	panels_final
end

# ╔═╡ Cell order:
# ╟─665b3437-0ce9-4818-ad14-32c4a433ee4e
# ╠═d1a2b3c4-0001-4000-8000-000000000001
# ╠═d1a2b3c4-0001-4000-8000-000000000002
# ╟─d1a2b3c4-0001-4000-8000-000000000003
# ╟─d1a2b3c4-0001-4000-8000-000000000004
# ╠═d1a2b3c4-0001-4000-8000-000000000005
# ╠═d1a2b3c4-0001-4000-8000-000000000006
# ╠═d1a2b3c4-0001-4000-8000-000000000029
# ╟─d1a2b3c4-0001-4000-8000-000000000007
# ╠═d1a2b3c4-0001-4000-8000-000000000008
# ╠═d1a2b3c4-0001-4000-8000-000000000009
# ╠═d1a2b3c4-0001-4000-8000-000000000010
# ╠═d1a2b3c4-0001-4000-8000-000000000011
# ╟─d1a2b3c4-0001-4000-8000-000000000012
# ╠═d1a2b3c4-0001-4000-8000-000000000013
# ╠═24af709c-a124-4671-86d0-41cbb4c82a67
# ╟─d1a2b3c4-0001-4000-8000-000000000014
# ╠═d1a2b3c4-0001-4000-8000-000000000015
# ╠═d1a2b3c4-0001-4000-8000-000000000030
# ╟─d1a2b3c4-0001-4000-8000-000000000016
# ╠═9d2ce5dc-e423-41c6-9353-e1a3c176622a
# ╠═d1a2b3c4-0001-4000-8000-000000000017
# ╟─d1a2b3c4-0001-4000-8000-000000000018
# ╠═d1a2b3c4-0001-4000-8000-000000000019
# ╠═d1a2b3c4-0001-4000-8000-000000000020
# ╟─d1a2b3c4-0001-4000-8000-000000000021
# ╠═d1a2b3c4-0001-4000-8000-000000000022
# ╟─d1a2b3c4-0001-4000-8000-000000000023
# ╠═d1a2b3c4-0001-4000-8000-000000000024
# ╠═d1a2b3c4-0001-4000-8000-000000000025
# ╠═d1a2b3c4-0001-4000-8000-000000000026
# ╠═d1a2b3c4-0001-4000-8000-000000000027
# ╠═d1a2b3c4-0001-4000-8000-000000000028
