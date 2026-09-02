### A Pluto.jl notebook ###
# v1.0.3

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
	using Colors
	using StaticArrays
	# using StatProfilerHTML

	using Rooms
	
	using Revise
	using GranularScenes
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

# ╔═╡ 740595ee-0af4-4c8a-96f0-0343f7004a24
"""
    test_render_cross(cam_height, cam_pitch, cam_fov, obs_height_ratio)

Renders a chiral test scene. 
"""
function test_render_cross(cam_height, cam_pitch, cam_fov, obs_height_ratio)
    # 1. Setup Renderer
    grid_dim = 128
    r_test = QuadTreeRenderer(; grid_res = grid_dim,
                                image_res = (256, 256),
                                use_cuda = false)

    d = grid_dim
    mid = d ÷ 2
    room_half = d ÷ 2 - 4
    max_room_height = d ÷ 2
    obs_h = round(Int, max_room_height * obs_height_ratio)

    # 2. Build Room Geometry on Host
    mat = zeros(Float32, d, d, d)

    # (a) Floor plane
    floor_y = 2
    for gx in (mid - room_half):(mid + room_half)
        for gz in (mid - room_half):(mid + room_half)
            mat[gx, floor_y, gz] = 1.0f0
        end
    end

    # (b) Outer Room Walls (Left, Right, Far)
    for gy in floor_y:(floor_y + max_room_height)
        for i in (mid - room_half):(mid + room_half)
            mat[mid - room_half, gy, i] = 1.0f0  # Left wall (-X)
            mat[mid + room_half, gy, i] = 1.0f0  # Right wall (+X)
            mat[i, gy, mid + room_half] = 1.0f0  # Far wall (+Z)
        end
    end

    # (c) Asymmetric "F"-like test pattern for chirality check:
    # - Central spine along Z
    # - Long bar to the RIGHT (+X) near center
    # - Short bar to the RIGHT (+X) further forward (+Z)
    # - Distinctive tall pillar on the LEFT (-X)
    arm_w = max(1, d ÷ 32)
    spine_len = room_half - 10

    for gy in floor_y:(floor_y + obs_h)
        # Main spine (near to far along Z)
        for gz in (mid - spine_len):(mid + spine_len)
            for gx in (mid - arm_w):(mid + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end

        # Long arm sticking out to the RIGHT (+X)
        for gx in (mid):(mid + spine_len ÷ 2)
            for gz in (mid - arm_w):(mid + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end

        # Shorter secondary arm to the RIGHT (+X), further down (+Z)
        for gx in (mid):(mid + spine_len ÷ 4)
            for gz in (mid + spine_len ÷ 2 - arm_w):(mid + spine_len ÷ 2 + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end
    end

    # Tall marker pillar on the LEFT (-X, near Z)
    pillar_h = round(Int, max_room_height * 0.8)
    for gy in floor_y:(floor_y + pillar_h)
        for gx in (mid - spine_len ÷ 2 - arm_w):(mid - spine_len ÷ 2 + arm_w)
            for gz in (mid - spine_len ÷ 3 - arm_w):(mid - spine_len ÷ 3 + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end
    end

    copyto!(r_test.grid_material, mat)

    # 3. Explicit Bounding Box
    half = Float32(d) * r_test.voxel_dx / 2.0f0
    r_test.bbox[1] = -half; r_test.bbox[2] = -half; r_test.bbox[3] = -half
    r_test.bbox[4] =  half; r_test.bbox[5] =  half; r_test.bbox[6] =  half
    @show r_test.bbox

    # 4. Camera Geometry (Standing at near wall, looking forward +Z, pitched down)
    floor_world_y = Float32(floor_y - mid)
    cam_world_y = floor_world_y + Float32(cam_height)
    cam_world_z = Float32(-room_half + 2)
    cam_world_x = 0.0f0

    pitch_rad = deg2rad(Float32(cam_pitch))
    target_dist = Float32(room_half * 2)
    target_x = 0.0f0
    target_y = cam_world_y + target_dist * tan(pitch_rad)
    target_z = cam_world_z + target_dist

    r_test.camera_pos = SVector{3,Float32}(cam_world_x, cam_world_y, cam_world_z)
    r_test.look_at    = SVector{3,Float32}(target_x, target_y, target_z)
    r_test.up         = SVector{3,Float32}(0.0f0, 1.0f0, 0.0f0)
    r_test.fov        = Float32(cam_fov)

    # 5. Render
    render!(r_test)

    depth_map(r_test)
end

# ╔═╡ 20aaf2ed-350d-4585-aa12-6fc010d67bd2
@bind cam_height Slider(10.0:1.0:100.0, default=50.0, show_value=true)

# ╔═╡ cff3abed-e02a-4918-80e4-3ebeba0fc59c
@bind cam_pitch Slider(-40.0:1.0:10.0, default=-12.0, show_value=true) # degrees down from horizontal

# ╔═╡ 5e44a860-09b7-44fe-b805-bb50058a081c
@bind cam_fov Slider(0.1:0.02:1.2, default=0.55, show_value=true)

# ╔═╡ dcad0f17-d957-4a43-87c4-f1334be5b59b
@bind obs_height_ratio Slider(0.1:0.05:0.9, default=0.45, show_value=true) # fraction of room height

# ╔═╡ df1e063b-08f7-4c22-a980-a756cac6c563
md"""
### Room & Camera Controls
- **Camera Height (head height)**: $(cam_height) voxels
- **Camera Pitch (look down)**: $(cam_pitch)°
- **Field of View (FOV)**: $(cam_fov)
- **Obstacle Height**: $(obs_height_ratio * 100)% of room height
"""

# ╔═╡ d6108d2b-1d6d-4530-aafe-d2ee3b1c9c6e
test_render_cross(cam_height, cam_pitch, cam_fov, obs_height_ratio)

# ╔═╡ a4186c8c-f1ad-479a-a5fc-5274b4344528
function render_cross(cam_height, cam_pitch, cam_fov, obs_height_ratio)
    # 1. Setup Renderer
    grid_dim = 128
    r_test = QuadTreeRenderer(; grid_res = grid_dim,
                                image_res = (256, 256),
                                use_cuda = false)

    d = grid_dim
    mid = d ÷ 2
    room_half = d ÷ 2 - 4          # Room spans [mid - room_half, mid + room_half]
    max_room_height = d ÷ 2        # Height of outer walls
    obs_h = round(Int, max_room_height * obs_height_ratio)

    # 2. Build Room Geometry on Host
    mat = zeros(Float32, d, d, d)

    # (a) Floor plane (1 voxel thick at bottom of room)
    floor_y = 2
    for gx in (mid - room_half):(mid + room_half)
        for gz in (mid - room_half):(mid + room_half)
            mat[gx, floor_y, gz] = 1.0f0
        end
    end

    # (b) Outer Room Walls (perimeter)
    for gy in floor_y:(floor_y + max_room_height)
        for i in (mid - room_half):(mid + room_half)
            mat[mid - room_half, gy, i] = 1.0f0  # Left wall (X min)
            mat[mid + room_half, gy, i] = 1.0f0  # Right wall (X max)
            mat[i, gy, mid + room_half] = 1.0f0  # Far wall (Z max)
            # (Near wall omitted so camera at wall can see in)
        end
    end

    # (c) Test Obstacles: Cross pattern with specified obstacle height
    arm_w = max(1, d ÷ 32)
    arm_len = room_half ÷ 2
    for gy in floor_y:(floor_y + obs_h)
        # X-aligned arm
        for gx in (mid - arm_len):(mid + arm_len)
            for gz in (mid - arm_w):(mid + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end
        # Z-aligned arm
        for gz in (mid - arm_len):(mid + arm_len)
            for gx in (mid - arm_w):(mid + arm_w)
                mat[gx, gy, gz] = 1.0f0
            end
        end
    end

    copyto!(r_test.grid_material, mat)

    # 3. Explicit Bounding Box (centered grid: world coords = grid_idx - mid)
    half = Float32(d) * r_test.voxel_dx / 2.0f0
    r_test.bbox[1] = -half; r_test.bbox[2] = -half; r_test.bbox[3] = -half
    r_test.bbox[4] =  half; r_test.bbox[5] =  half; r_test.bbox[6] =  half

    # 4. Camera Geometry:
    # Placed at near wall (center X=0, near Z = -room_half)
    # Head height Y = (floor_y - mid) + cam_height in world coords
    floor_world_y = Float32(floor_y - mid)
    cam_world_y = floor_world_y + Float32(cam_height)
    cam_world_z = Float32(-room_half + 2)  # slightly inside the near wall
    cam_world_x = 0.0f0

    # Look target: Straight ahead in +Z, tilted down according to pitch
    pitch_rad = deg2rad(Float32(cam_pitch))
    target_dist = Float32(room_half * 2)
    target_x = 0.0f0
    target_y = cam_world_y + target_dist * tan(pitch_rad)
    target_z = cam_world_z + target_dist

    r_test.camera_pos = SVector{3,Float32}(cam_world_x, cam_world_y, cam_world_z)
    r_test.look_at    = SVector{3,Float32}(target_x, target_y, target_z)
    r_test.up         = SVector{3,Float32}(0.0f0, 1.0f0, 0.0f0)
    r_test.fov        = Float32(cam_fov)

    # 5. Render
    render!(r_test)

    # 6. Display Depth Map
    depth = Array(r_test.depth_buffer)[:, :, 1]
    hits = depth .> 0.0f0
    d_display = zeros(Float32, size(depth))
    if any(hits)
        d_min, d_max = extrema(depth[hits])
        # Invert so closer objects are brighter
        d_display[hits] .= one(Float32) .- (depth[hits] .- d_min) ./ max(d_max - d_min, Float32(1E-4))
    end

    Gray.(d_display)
end

# ╔═╡ 14a33876-0998-47a2-a7ce-96cace0cd335
dataset = "window-0.1/2025-02-05_vifdDO"

# ╔═╡ 8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx).json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

# ╔═╡ 67bb0b77-f540-480a-aa42-0188d0df1ca4
function mytest()
    r = load_room(1)

    d = grid_dim = 16
  
    mid = d ÷ 2
    room_half = d ÷ 2 - 4
    max_room_height = d ÷ 2
    obs_h = round(Int, max_room_height * obs_height_ratio)

    # (a) Floor plane
    floor_y = 2

    floor_world_y = Float32(floor_y - mid)
    cam_world_y = floor_world_y + Float32(cam_height)
    cam_world_z = Float32(-room_half + 2)
    cam_world_x = 0.0f0

    pitch_rad = deg2rad(Float32(cam_pitch))
    target_dist = Float32(room_half * 2)
    target_x = 0.0f0
    target_y = cam_world_y + target_dist * tan(pitch_rad)
    target_z = cam_world_z + target_dist

    params = QuadTreeModel(r;
                           render_kwargs = Dict(:image_res => (256,256), 
                                                :use_cuda => false,
                                                :grid_res => grid_dim,
                         :obstacle_height => obs_h,
                         :camera_pos => SVector{3,Float32}(cam_world_x, cam_world_y, cam_world_z),
                         :look_at   => SVector{3,Float32}(target_x, target_y, target_z),
                         :fov => Float32(cam_fov)))


    qt = QuadTree(quad_tree_prior(params.start_node, 1))
    @time depth = qt_observe(params.renderer, qt, params.pixel_var)

    
 

    @show params.renderer.bbox

    

    (params.renderer, depth_map_array(depth))
end


# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bebb
begin
    renderer, depth = mytest();
    depth
end

# ╔═╡ 96230e38-9edf-4fa5-ab2e-a80b699371cf
# Debug visualizers for the renderer's obstacle buffer (`grid_material`).
# Paste into notebooks/qt_render.jl after a `write_obstacles!(r, qt)` call.
# Shows the buffer directly (orthographic projections), independent of the
# ray marcher, so you can confirm the buffer is filled as expected.

"""
    debug_topdown(r::QuadTreeRenderer)

Top-down orthographic view of the obstacle buffer: maximum intensity over the
vertical axis (dim 2), giving the floor plan `(x, z)` plane.
"""
function debug_topdown(r::QuadTreeRenderer)
    mat = r.grid_material isa Array ? r.grid_material : Array(r.grid_material)
    top_down = dropdims(maximum(mat, dims = 2), dims = 2)   # (x, z)
    @show size(top_down)
    return Gray.(clamp.(top_down', 0.0f0, 1.0f0))            # (z, x) display
end

# ╔═╡ 5f309061-321a-412d-a1dd-f1da047568f4
debug_topdown(renderer)           # floor plan

# ╔═╡ f5927a6a-566a-4d8e-af45-f6f530b8b9fa
"""
    debug_sideview(r::QuadTreeRenderer; from_x = true)

Side orthographic view of the obstacle buffer.  `from_x = true` projects along
the lateral axis (dim 1) → the (z, y) elevation from the side.  `from_x =
false` projects along z (dim 3) → the (x, y) elevation from the front.
"""
function debug_sideview(r::QuadTreeRenderer; from_x::Bool = true)
    mat = r.grid_material isa Array ? r.grid_material : Array(r.grid_material)
    if from_x
        side = dropdims(maximum(mat, dims = 1), dims = 1)    # (y, z)
        return Gray.(clamp.(reverse(side, dims = 1)', 0.0f0, 1.0f0))  # (z, y), y up
    else
        side = dropdims(maximum(mat, dims = 3), dims = 3)    # (x, y)
        return Gray.(clamp.(reverse(side, dims = 1)', 0.0f0, 1.0f0))  # (y, x), y up
    end
end

# ╔═╡ 95103afa-e049-4088-b2ee-c6cdc65180a0
debug_sideview(renderer)          # elevation from the side

# ╔═╡ 57184b8c-e34b-4fbf-a870-02f795e1396e
"""
    debug_occupancy_stats(r::QuadTreeRenderer)

Counts of occupied voxels and per-slab occupancy (how many cells are filled at
each gy) — quick numeric confirmation that the buffer is being filled.
"""
function debug_occupancy_stats(r::QuadTreeRenderer)
    mat = r.grid_material isa Array ? r.grid_material : Array(r.grid_material)
    d = size(mat, 1)
    occ = count(!iszero, mat)
    per_gy = [count(!iszero, @view mat[:, gy, :]) for gy in 1:d]
    top = findlast(!iszero, per_gy)
    return (occupied = occ, fraction = occ / (d^3),
            top_gy = top === nothing ? 0 : top,
            max_gy = maximum(per_gy), argmax_gy = argmax(per_gy))
end

# ╔═╡ 2f4fa50d-d1d5-4349-8411-3891b24ca0c5
debug_occupancy_stats(renderer)   # numbers

# ╔═╡ Cell order:
# ╟─d697c7c5-664d-4273-a24a-78823aab6bae
# ╠═1d39e7ee-a6e3-11f1-3258-19edf57342c6
# ╟─740595ee-0af4-4c8a-96f0-0343f7004a24
# ╟─df1e063b-08f7-4c22-a980-a756cac6c563
# ╠═20aaf2ed-350d-4585-aa12-6fc010d67bd2
# ╠═cff3abed-e02a-4918-80e4-3ebeba0fc59c
# ╠═5e44a860-09b7-44fe-b805-bb50058a081c
# ╠═dcad0f17-d957-4a43-87c4-f1334be5b59b
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bebb
# ╠═d6108d2b-1d6d-4530-aafe-d2ee3b1c9c6e
# ╟─a4186c8c-f1ad-479a-a5fc-5274b4344528
# ╠═14a33876-0998-47a2-a7ce-96cace0cd335
# ╠═8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
# ╠═67bb0b77-f540-480a-aa42-0188d0df1ca4
# ╠═5f309061-321a-412d-a1dd-f1da047568f4
# ╠═95103afa-e049-4088-b2ee-c6cdc65180a0
# ╠═2f4fa50d-d1d5-4349-8411-3891b24ca0c5
# ╠═96230e38-9edf-4fa5-ab2e-a80b699371cf
# ╠═f5927a6a-566a-4d8e-af45-f6f530b8b9fa
# ╠═57184b8c-e34b-4fbf-a870-02f795e1396e
