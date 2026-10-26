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
	using FileIO

	import Flux

	using Rooms
	
	using Revise
	using GranularScenes
	import GranularScenes as GS
end


# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bec1
begin
	using Base64
	# Pluto sanitises markdown, which strips `data:` image sources. So the table is
	# emitted as a raw HTML cell output instead: a cell's own display output is not
	# sanitised, and an `img` with a data URI renders normally there.
	struct RawHTML
		html::String
	end

	Base.show(io::IO, ::MIME"text/html", h::RawHTML) = print(io, h.html)

	function png_uri(img)
		io = IOBuffer()
		FileIO.save(FileIO.Stream{FileIO.format"PNG"}(io), img)
		return "data:image/png;base64," * base64encode(take!(io))
	end

	img_html(img; w = 256) = "<img width=\"$w\" style=\"image-rendering:pixelated\" src=\"$(png_uri(img))\">"
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
	OCC_CKPT = get(ENV, "DDP_OCC_CKPT", "/spaths/checkpoints/fluxocc_e30.weights.jld2")  # NB: test_ddp.jl had a DDP_VAE_CKPT typo here

	const DATA_DEPTH_MIN = 3.971077f0
	const DATA_DEPTH_MAX = 10.181246f0

	# DDP -> quadtree parameters
	QT_VAR       = 0.01      # split threshold on node footprint std
	QT_MAX_LEVEL = 3         # 8x8 max leaves
	QT_MAX_DEPTH = 4

	# inference chain
	N_CHAIN_STEPS = 500
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
    @show size(depth_for_ddp)
    qt = GS.qt_ddp(dds, depth_for_ddp)
    @show typeof(cm)
    @show typeof(qt)
    GS.cm_from_qt!(cm, qt)

    
    vision_prior = QTVisionPrior()
    vision_obsm  = QTVisionLikelihood(renderer, pixel_var)
    
    
    vision_prot = AdaptiveMH(
        ;
                            model = qt_vision,
                            model_args = (qt.schema, vision_prior, vision_obsm),
                            obs = cm,
                             chain_length = 20,
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
        load = 5,
        itemp = 1,
    )
    attention_module = AttentionModule(ac)

    mo = MO(;window=5)
    granularity_module = GranularityModule(mo, qt.schema)

    println("Pre-attentive Perception")
    @time for _ = 1:10
        step_module!(vision_module)
    end

    println("Planning")
    @time step_module!(planning_module, vision_module)

    println("Attention Loop")

    # Rollout inspection. Panels stay at their native resolution (a 4x4 or
    # 16x16 quadtree image, a 256x256 depth map) and are upscaled once, for the
    # frame the slider selects. A 16x16 RGB{Float64} panel is 6 kB, so 100
    # frames cost ~110 MB; retained 256x256 montages cost ~600 MB.
    gt_occ   = room_topdown(r)                                        # 16x16, static
    gt_panel = GS.viz_depth(depth; lo = DATA_DEPTH_MIN, hi = DATA_DEPTH_MAX)

    path_panels = Vector{Matrix{RGB{Float64}}}(undef, N_CHAIN_STEPS)
    att_panels  = Vector{Matrix{RGB{Float64}}}(undef, N_CHAIN_STEPS)
    pred_panels = Vector{Matrix{RGB{Float64}}}(undef, N_CHAIN_STEPS)
    err_panels  = Vector{Matrix{RGB{Float64}}}(undef, N_CHAIN_STEPS)

    n_leaves = Vector{Int}(undef, N_CHAIN_STEPS)
    ms_att   = Vector{Float64}(undef, N_CHAIN_STEPS)
    ms_gran  = Vector{Float64}(undef, N_CHAIN_STEPS)

    for t = 1:N_CHAIN_STEPS
        ta = @timed step_module!(attention_module, vision_module, planning_module)
        tg = @timed step_module!(granularity_module, attention_module, vision_module, planning_module)

        # The frame is the MAP trace of the current vision module, i.e. the
        # state the next step will condition on.
        pred_trace = maximum_aposteriori(vision_module)
        pred_trace, = Gen.regenerate(pred_trace, select(:depth))
        pred_panel = GS.viz_depth(pred_trace[:depth]; lo = DATA_DEPTH_MIN, hi = DATA_DEPTH_MAX)

        bp_idx, bp_vec = GranularScenes.best_path(planning_module)
        bp_qt = get_retval(vision_module.state.samples[bp_idx])

        ms_att[t] = 1e3 * ta.time
        ms_gran[t] = 1e3 * tg.time
        n_leaves[t] = GS.nleaves(granularity_module.state.schema)
        path_panels[t] = qt_topdown(bp_qt, bp_vec)
        att_panels[t] =  att_topdown(bp_qt, attention_module)
        pred_panels[t] =  pred_panel
        err_panels[t] =  GS.viz_err(gt_panel, pred_panel)

        #pred_trace = nothing
        #t % 10 == 0 && GC.gc(false)   # prompt-snapshot garbage from the renderer
    end

    frames = (; gt_occ, gt_panel,
                path = path_panels, att = att_panels,
                pred = pred_panels, err = err_panels,
                t = collect(1:N_CHAIN_STEPS), n_leaves, ms_att, ms_gran)

    best_path_idx, best_path_vec = GranularScenes.best_path(planning_module)
    best_path_qt = get_retval(vision_module.state.samples[best_path_idx])
    path_viz = qt_topdown(best_path_qt, best_path_vec)

    att_viz = att_topdown(best_path_qt, attention_module)

    map_trace = maximum_aposteriori(vision_module)
    map_trace, _... = Gen.regenerate(map_trace, select(:depth))
    map_depth = depth_map_array(map_trace[:depth])

    gt_topdown = gt_occ
    (; gt_topdown, gt_depth, map_depth, best_path = path_viz, att_vis = att_viz, frames)
end


# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bebb
(; gt_topdown, gt_depth, map_depth, best_path, att_vis, frames) = run_model();

# ╔═╡ 290f6e10-8b16-4646-86bf-9d79d9831919
gt_depth

# ╔═╡ 7eb74e0e-10e8-4d69-9cfe-d88ca20d081b
map_depth

# ╔═╡ 2991cb98-187b-4b21-a47e-e32e04346353
att_vis

# ╔═╡ db88ff55-b7f0-4ea4-b63f-d36178efd42f
best_path

# ╔═╡ 4e263aee-7db2-45ce-b8be-375d4806318f
gt_topdown

# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bec2
@bind frame_i Slider(1:length(frames.t); default = 1, show_value = true)

# ╔═╡ 0d1f4a02-6b2e-4b7c-9f1a-2c5e8d3a7b04
let i = frame_i
	cell(img) = "<td style=\"text-align:center;padding:4px\">$(img_html(img))</td>"
	head(s)   = "<th style=\"font-weight:600;padding:4px\">$s</th>"
	RawHTML("""
	<div style=\"font-family:sans-serif\">
	<b>step $(frames.t[i]) / $(last(frames.t))</b> · leaves
	<b>$(frames.n_leaves[i])</b> · attention
	<b>$(round(frames.ms_att[i], digits = 1)) ms</b> · granularity
	<b>$(round(frames.ms_gran[i], digits = 1)) ms</b>
	</div>
	<table style=\"border-collapse:separate;border-spacing:4px\">
	<tr>$(head("occupancy"))$(head("best path"))$(head("attention"))</tr>
	<tr>$(cell(frames.gt_occ))$(cell(frames.path[i]))$(cell(frames.att[i]))</tr>
	<tr>$(head("gt depth"))$(head("MAP depth"))$(head("abs err"))</tr>
	<tr>$(cell(frames.gt_panel))$(cell(frames.pred[i]))$(cell(frames.err[i]))</tr>
	</table>
	""")
end

# ╔═╡ Cell order:
# ╟─d697c7c5-664d-4273-a24a-78823aab6bae
# ╠═1d39e7ee-a6e3-11f1-3258-19edf57342c6
# ╠═290f6e10-8b16-4646-86bf-9d79d9831919
# ╠═7eb74e0e-10e8-4d69-9cfe-d88ca20d081b
# ╠═2991cb98-187b-4b21-a47e-e32e04346353
# ╠═db88ff55-b7f0-4ea4-b63f-d36178efd42f
# ╠═4e263aee-7db2-45ce-b8be-375d4806318f
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bebb
# ╠═14a33876-0998-47a2-a7ce-96cace0cd335
# ╠═8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
# ╠═437e2b88-ae73-48f9-8d26-511b193f4bc3
# ╠═7982dc9c-7e3a-481c-9973-f29b2bac96a1
# ╠═b0f77a86-186e-462e-928f-9d857aecc307
# ╠═67bb0b77-f540-480a-aa42-0188d0df1ca4
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bec1
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bec2
# ╠═0d1f4a02-6b2e-4b7c-9f1a-2c5e8d3a7b04
