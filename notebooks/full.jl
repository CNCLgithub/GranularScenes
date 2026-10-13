### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 1d39e7ee-a6e3-11f1-3258-19edf57342c6
begin
	using Pkg
	Pkg.activate("..")
	
	using Gen
	using PlutoUI
	using Random
	using Printf
	using JSON
	using CUDA
	using Colors
	using StaticArrays

	import Flux

	using Rooms
	
	using Revise
	using GranularScenes
	import GranularScenes as GS
end


# ╔═╡ d697c7c5-664d-4273-a24a-78823aab6bae
html"""
<style>
    @media screen {
        main {
            margin: 0 auto;
            max-width: 3000px;
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

# ╔═╡ 14a33876-0998-47a2-a7ce-96cace0cd335
dataset = "window-0.1/2025-02-05_vifdDO"

# ╔═╡ 8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx).json")
    local base_s
    open(path, "r") do f
        base_s = Dict(JSON.parse(f))
    end
    from_json(GridRoom, base_s)
end

# ╔═╡ 437e2b88-ae73-48f9-8d26-511b193f4bc3
begin
	# CAMERA SETTINGS
	IMG_RES = (256, 256)
	cam_height = 50.0f0
	cam_pitch  = -12.0f0
	cam_fov    = 0.56f0
	d = 16
	mid = d ÷ 2
	room_half = d ÷ 2 - 4
	max_room_height = d ÷ 2
	obs_h = 5
	floor_y = 2
	floor_world_y = Float32(floor_y - mid)
	# --- camera scaled to the grid --------------------------------------
	# cam_height slider (5..60) is world units; must stay < d÷2 (bbox top).
	cam_h       = min(cam_height, Float32(max_room_height - 2))
	cam_world_y = floor_world_y + cam_h
	# cam_world_z = Float32(d-2)
	cam_world_z = Float32(-(d-2)) 
	cam_world_x = 0.0f0
	
	pitch_rad   = deg2rad(Float32(cam_pitch))
	# target_dist = Float32(-room_half * 2)
	target_dist = Float32(room_half * 2)
	target_x    = 0.0f0
	target_y    = cam_world_y + target_dist * tan(pitch_rad)
	target_z    = cam_world_z + target_dist
	cam_pos = SVector{3,Float32}(cam_world_x, cam_world_y, cam_world_z)
	look_at = SVector{3,Float32}(target_x, target_y, target_z)
	
end

# ╔═╡ 7982dc9c-7e3a-481c-9973-f29b2bac96a1
begin
	DATA     = get(ENV, "DDP_DATA",     "/spaths/datasets/ddp_train_11f_32x32.hdf5")
	VAE_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxvae_e30.weights.jld2")
	OCC_CKPT = get(ENV, "DDP_OCC_CKPT", "/spaths/checkpoints/fluxocc_e30.weights.jld2")

	const DATA_DEPTH_MIN = 3.971077f0
	const DATA_DEPTH_MAX = 10.181246f0

	# DDP -> quadtree parameters
	QT_VAR       = 0.36      # split threshold on node footprint std
	QT_MAX_LEVEL = 4         # 8x8 max leaves
	QT_MAX_DEPTH = 6

	# inference chain
	N_CHAIN_STEPS = 60
	VIZ_EVERY     = 5
	MARGIN_SIZE   = 50
end


# ╔═╡ b0f77a86-186e-462e-928f-9d857aecc307
begin
	dds = GS.DataDrivenState(
		;
		device = Flux.cpu,
		vae_path = VAE_CKPT,
		occ_path = OCC_CKPT,
		var = QT_VAR
	)

	@info "Loaded frozen VAE from " VAE_CKPT
	@info "Loaded frozen OCC from " OCC_CKPT
end


# ╔═╡ 67bb0b77-f540-480a-aa42-0188d0df1ca4
function run_model()
    r = load_room(1)

    
    pixel_var = Float32(.05)
     renderer = QuadTreeRenderer(
        ;
        image_res = IMG_RES,
        use_cuda = true,
        wall_mode = true,
        grid_res = d,
        obstacle_height = 5,
        camera_pos = cam_pos,
        look_at = look_at,
        fov = cam_fov,
    )


    depth = qt_observe(renderer, r, pixel_var)
    gt_depth = depth_map_array(depth)

    cm = choicemap(:depth => depth)

    depth_for_ddp = reshape(depth, 256, 256, 1, 1)
    depth_for_ddp .-= DATA_DEPTH_MIN
    depth_for_ddp .*= 1f0 / DATA_DEPTH_MAX
    qt_vae = GS.qt_ddp(dds, depth_for_ddp)

    @show GS.nleaves(qt_vae)
    GS.cm_from_qt!(cm, qt_vae)

    
    vision_prior = QTVisionPrior()
    vision_obsm  = QTVisionLikelihood(renderer, pixel_var)
    
    
    vision_prot = AdaptiveMH(
        ;
                            model = qt_vision,
                            model_args = (qt_vae.schema, vision_prior, vision_obsm),
                            obs = cm,
                             chain_length = 1000,
                            )

    vision_module = PerceptionModule(vision_prot, cm)

    canonical_qt = QuadTree(5, 5)
    start_pos = GS.room_index_to_qt(r, first(entrance(r)), canonical_qt)
    stop_pos = GS.room_index_to_qt(r, first(exits(r)), canonical_qt)

    @show start_pos
    @show stop_pos

    planning_prot = AStarPlanner(;
        ent = start_pos,
        ext = stop_pos,
        nsamples = vision_prot.chain_length,
    )
    planning_module = PlanningModule(planning_prot)

    ac = AdaptiveComputation{QTVisionTrace}(
        ;
        load = 10,
        itemp = 2,
    )
    attention_module = AttentionModule(ac)

    mo = MO(;window=20)
    granularity_module = GranularityModule(mo, qt_vae.schema)

    println("Pre-attentive Perception")
    @time for _ = 1:10
        step_module!(vision_module)
    end

    println("Planning")
    @time step_module!(planning_module, vision_module)

    println("Attention Loop")

    for _ = 1:200
        @time step_module!(attention_module, vision_module, planning_module)
        @time step_module!(granularity_module, attention_module, vision_module, planning_module)
    end
    
    best_path_idx, best_path_vec = GranularScenes.best_path(planning_module)
    best_path_qt = get_retval(vision_module.state.samples[best_path_idx])
    path_viz = qt_topdown(best_path_qt, best_path_vec)

    att_viz = att_topdown(best_path_qt, attention_module)

    map_trace = maximum_aposteriori(vision_module)
    map_trace, _... = Gen.regenerate(map_trace, select(:depth))
    map_depth = depth_map_array(map_trace[:depth])

    gt_room = r
    (; gt_room, gt_depth, qt_vae, map_trace, map_depth, path_viz, att_viz, renderer)
end


# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bebb
run = run_model();

# ╔═╡ 290f6e10-8b16-4646-86bf-9d79d9831919
run.gt_depth

# ╔═╡ 7eb74e0e-10e8-4d69-9cfe-d88ca20d081b
run.map_depth

# ╔═╡ af4439b6-6bb5-41dd-be5a-f9820a2aa5f3
extrema(run.map_trace[:qt => :weights])

# ╔═╡ 2991cb98-187b-4b21-a47e-e32e04346353
run.att_viz

# ╔═╡ 327e0967-9bd0-461b-804f-cbf542fa6496
qt_topdown(run.qt_vae)

# ╔═╡ 07b90484-8f48-4c17-a4ce-271d618c5d25
qt_topdown(run.map_trace[])

# ╔═╡ 4e263aee-7db2-45ce-b8be-375d4806318f
qt_topdown(run.gt_room)

# ╔═╡ Cell order:
# ╟─d697c7c5-664d-4273-a24a-78823aab6bae
# ╠═1d39e7ee-a6e3-11f1-3258-19edf57342c6
# ╠═290f6e10-8b16-4646-86bf-9d79d9831919
# ╠═7eb74e0e-10e8-4d69-9cfe-d88ca20d081b
# ╠═af4439b6-6bb5-41dd-be5a-f9820a2aa5f3
# ╠═2991cb98-187b-4b21-a47e-e32e04346353
# ╠═327e0967-9bd0-461b-804f-cbf542fa6496
# ╠═07b90484-8f48-4c17-a4ce-271d618c5d25
# ╠═4e263aee-7db2-45ce-b8be-375d4806318f
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bebb
# ╠═14a33876-0998-47a2-a7ce-96cace0cd335
# ╠═8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
# ╠═437e2b88-ae73-48f9-8d26-511b193f4bc3
# ╠═7982dc9c-7e3a-481c-9973-f29b2bac96a1
# ╠═b0f77a86-186e-462e-928f-9d857aecc307
# ╠═67bb0b77-f540-480a-aa42-0188d0df1ca4
