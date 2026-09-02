# QuadTreeRenderer — Julia-native, AcceleratedKernels-backed replacement for the
# Taichi voxel renderer. Included from gm.jl (same GranularScenes module scope),
# so it shares access to write_obstacles!, project_qt!, QuadTree from geometry.jl.

using CUDA
using AcceleratedKernels
import AcceleratedKernels as AK

using StaticArrays
using LinearAlgebra: norm, cross, dot

# tested CPU/AK render core (camera, DDA, accumulation, normalization)
include("voxel_render_core.jl")


export QuadTreeRenderer, render!, set_obstacles!, reset_voxels!, observe_pixels,
       logpdf, random, save_img, depth_map, depth_map_array

"""
    QuadTreeRenderer(grid_res::Int, image_res::Tuple{Int,Int}, up::SVector{3,Float64};
                     exposure = 1.0, voxel_edges = 0.005)

Julia-native, AcceleratedKernels-backed replacement for the Taichi voxel renderer.
Owns:
  - `grid_material::CuArray{Float32, 3}`  (0 = empty, 1 = obstacle, 2 = light)
  - `depth_buffer::CuArray{Float32, 2}` (image_res)
  - `rendered::CuArray{Float32, 3}` (image_res × 1, depth-normalized)
  - `bbox::CuArray{Float32, 1}` (6)  [min_xyz; max_xyz]

Backend is chosen automatically by AK from the CuArray type; CPU `Array` fallback
works too (pass Array buffers) for debugging without a GPU.
"""
mutable struct QuadTreeRenderer{MA<:AbstractArray, MFA<:AbstractArray}
    grid_res::Int
    image_res::Tuple{Int,Int}
    voxel_dx::Float32
    voxel_inv_dx::Float32
    exposure::Float32
    voxel_edges::Float32
    obstacle_height::Int         # voxel columns rise this many cells from the floor

    grid_material::MA
    grid_material_reset::MA

    depth_buffer::MFA            # Float32 image_res
    rendered::MFA                # Float32 image_res × 1 (depth-normalized)
    noise_buffer::MFA            # Float32 image_res × 1 (preallocated gaussian noise)

    bbox::Vector{Float32}        # host 6-vector [minx,miny,minz,maxx,maxy,maxz] — always host;
    rand_buffer::MA              # Float32 grid_res÷2 × grid_res÷2

    leaf_inds::AbstractVector{Int32}     # device-packed leaf cell indices, 1-D
    leaf_weights::AbstractVector{Float32}  # device-packed leaf cell weights, 1-D
    max_leaf_cells::Int          # capacity of leaf_inds / leaf_weights

    # host-side preallocated pack buffers (CPU arrays; avoid per-call allocs)
    pack_inds::Vector{Int32}
    pack_wts::Vector{Float32}
    pack_n::Int                  # number of used entries after last pack

    camera_pos::SVector{3,Float32}   # host-side camera state (not GPU arrays; read once/frame)
    look_at::SVector{3,Float32}
    up::SVector{3,Float32}
    fov::Float32
    floor_height::Float32
    light_direction::SVector{3,Float32}
    light_direction_noise::Float32
end

const DEFAULT_UP = SVector(0.0f0, 1.0f0, 0.0f0)
const EPS = Float32(1E-4)
const INF = Float32(1E10)
const MAX_RAY_DEPTH = 10
const DIS_LIMIT = 200.0f0

"""
    backend_arrays(::Type{T}, grid_res, image_res) -> (grid, depth, rendered, bbox, randb, camera, look, up, fov, floorh, ldir, lnoise)

Factory returning the right array type for the current backend. On a machine with
CUDA available, use CuArray buffers (AK auto-selects CUDA backend). Fall back to
plain Array for CPU-only debugging.
"""
function buffer_allocator(use_cuda::Bool)
    if use_cuda
        (T, n, m=1, k=1) -> (CuArray{T}(undef, n, m, k))
    else
        (T, n, m=1, k=1) -> (Array{T}(undef, n, m, k))
    end
end

function QuadTreeRenderer(;grid_res::Int = 128,
                          image_res::Tuple{Int,Int} = (512,512),
                          up::SVector{3,Float32} = SVector(0.0f0, 1.0f0, 0.0f0),
                          exposure::Float32 = 1.0f0, voxel_edges::Float32 = 0.005f0,
                          obstacle_height::Int = -1,          # -1 → default grid_res ÷ 6 (Taichi oheight n//6)
                          camera_pos::SVector{3,Float32} = SVector(0.4f0, 40.0f0, 60.0f0),
                          fov::Float32 = 0.56f0,
                          look_at::SVector{3, Float32} = SVector(0.0f0, -5.0f0, -60.0f0),
                          use_cuda::Bool = CUDA.functional())
    alloc = buffer_allocator(use_cuda)
    m = grid_res
    w, h = image_res

    grid_material = alloc(Float32, m, m, m)
    grid_material_reset = alloc(Float32, m, m, m)
    depth_buffer = alloc(Float32, w, h, 1)
    rendered = alloc(Float32, w, h, 1)
    noise_buffer = alloc(Float32, w, h, 1)
    bbox = zeros(Float32, 6)     # host-only; read on the CPU every render, never on the GPU
    rand_buffer = alloc(Float32, m ÷ 2, m ÷ 2, 1)

    # max-allocated pack buffers: worst case is every grid cell a leaf cell
    max_cells = m * m
    pack_inds = Vector{Int32}(undef, max_cells)
    pack_wts = Vector{Float32}(undef, max_cells)
    leaf_inds = use_cuda ? CuArray{Int32}(undef, max_cells) : Vector{Int32}(undef, max_cells)
    leaf_weights = use_cuda ? CuArray{Float32}(undef, max_cells) : Vector{Float32}(undef, max_cells)

    # zero-initialize all device/host buffers before first use
    fill!(grid_material, 0.0f0)
    fill!(grid_material_reset, 0.0f0)
    fill!(depth_buffer, 0.0f0)
    fill!(rendered, 0.0f0)
    fill!(noise_buffer, 0.0f0)
    fill!(bbox, 0.0f0)
    fill!(rand_buffer, 0.0f0)

    upv = SVector{3,Float32}(up[1], up[2], up[3])
    oh = obstacle_height < 0 ? grid_res ÷ 6 : obstacle_height
    oh = min(oh, grid_res)          # clamp: never taller than the grid
    floor_height = 0.0f0
    light_direction = SVector{3,Float32}(0.0f0, 1.0f0, 0.0f0)
    light_direction_noise = 0.0f0

    QuadTreeRenderer(grid_res, image_res, 1.0f0, 1.0f0, exposure, voxel_edges, oh,
                     grid_material, grid_material_reset,
                     depth_buffer, rendered, noise_buffer, bbox, rand_buffer,
                     leaf_inds, leaf_weights, max_cells,
                     pack_inds, pack_wts, 0,
                     camera_pos, look_at, upv, fov, floor_height,
                     light_direction, light_direction_noise)
end

# camera / light accessors (kept Interface compatible with TaichiScene)
function set_camera_pos!(r::QuadTreeRenderer, x::Real, y::Real, z::Real)
    r.camera_pos = SVector{3,Float32}(x, y, z)
    nothing
end
function set_look_at!(r::QuadTreeRenderer, x::Real, y::Real, z::Real)
    r.look_at = SVector{3,Float32}(x, y, z)
    nothing
end
function set_up!(r::QuadTreeRenderer, x::Real, y::Real, z::Real)
    r.up = SVector{3,Float32}(x, y, z)
    nothing
end
function set_fov!(r::QuadTreeRenderer, f::Real)
    r.fov = Float32(f)
    nothing
end
function set_directional_light!(r::QuadTreeRenderer, direction::NTuple{3,Real},
                                noise::Real)
    d = SVector{Float32,3}(direction)
    dn = norm(d)
    r.light_direction = SVector{3,Float32}(d[1]/dn, d[2]/dn, d[3]/dn)
    r.light_direction_noise = Float32(noise)
    nothing
end
function set_floor!(r::QuadTreeRenderer, height::Real)
    r.floor_height = Float32(height)
    nothing
end

# ---------------- kernels (AcceleratedKernels) --------------------

# (CPU/GPU agnostic; AK picks backend from array)
function _reset_voxels_kernel!(material, material_reset)
    AK.foreachindex(material) do I
        material[I] = material_reset[I]
    end
    nothing
end

function reset_voxels!(r::QuadTreeRenderer)
    _reset_voxels_kernel!(r.grid_material, r.grid_material_reset)
    fill!(r.depth_buffer, 0.0f0)
    fill!(r.rendered, 0.0f0)
    nothing
end

# set_obstacles: for each (i,j) in rand_buffer with val > eps, fill the column
# from y in -hn : oheight-1 with material=val
function _set_obstacles_kernel!(material,
                                rand_buffer,
                                obstacle_height::Int)
    @inbounds begin
        nx, ny = size(rand_buffer, 1), size(rand_buffer, 2)
        n = max(nx, ny)
        hn = n ÷ 2
        oheight = -hn + obstacle_height     # floor at -hn, rise obstacle_height cells
        AK.foreachindex(rand_buffer) do I
            i = (I - 1) % nx + 1
            j = (I - 1) ÷ nx + 1
            val = rand_buffer[i, j]
            if val > EPS
                x = i - hn
                z = j - hn
                for y in (-hn):(oheight - 1)
                    # voxel index offset to center grid: + grid_res÷2
                    gx = x + n÷2
                    gy = y + n÷2
                    gz = z + n÷2
                    if 1 <= gx <= size(material,1) && 1 <= gy <= size(material,2) && 1 <= gz <= size(material,3)
                        material[gx, gy, gz] = val
                    end
                end
            end
        end
    end
    nothing
end

function set_obstacles!(r::QuadTreeRenderer, omap::AbstractMatrix)
    # ensure rand_buffer receives the map
    copyto!(r.rand_buffer, reshape(omap, size(r.rand_buffer,1), size(r.rand_buffer,2)))
    _set_obstacles_kernel!(r.grid_material, r.rand_buffer, r.obstacle_height)
    # keep the bbox fresh: obstacle columns occupy the omap's occupied cells,
    # rising obstacle_height from the floor (world y = -d/2).
    d = grid_res(r); dx = r.voxel_dx
    hn = -Float32(d) * dx / 2
    occ = findall(>(EPS), omap)
    if isempty(occ)
        half = Float32(d) * dx / 2
        r.bbox[1] = r.bbox[2] = -half; r.bbox[3] = hn
        r.bbox[4] = r.bbox[5] = half; r.bbox[6] = hn + Float32(r.obstacle_height) * dx
        return nothing
    end
    xs = Float32[ (i - 1 - d÷2) * dx for (i, _) in occ ]
    zs = Float32[ (j - 1 - d÷2) * dx for (_, j) in occ ]
    r.bbox[1] = minimum(xs); r.bbox[2] = minimum(zs)
    r.bbox[3] = hn
    r.bbox[4] = maximum(xs) + dx; r.bbox[5] = maximum(zs) + dx
    r.bbox[6] = hn + Float32(r.obstacle_height) * dx
    nothing
end

# ---------------------------------------------------------------------------
# Design A: direct quadtree -> GPU grid (no dense CPU omap intermediate)
# ---------------------------------------------------------------------------

"""
    pack_leaf_cells!(inds::Vector{Int32}, wts::Vector{Float32}, lv, d::Int64) -> Int

Allocation-free packer: writes packed `(linear_index, weight)` pairs into the
preallocated `inds`/`wts` buffers and returns the number of used entries.
Capacity is `length(inds)`; callers must preallocate at `d*d` (the rendering
worst case, when every grid cell is a leaf cell). Cell indices are linear
column-major over the `d × d` grid; weights are thresholded exactly as in
`project_qt!` (`w > 0.025 ? w : 0.0`).

Throws `BoundsError` if a quadtree exceeds the buffer capacity.
"""
function pack_leaf_cells!(inds::Vector{Int32}, wts::Vector{Float32}, lv, d::Int64)
    n = 1
    for x in lv
        idx = node_to_idx(x.node, d)      # same footprint as project_qt!
        w = weight(x)
        w = w > 0.025f0 ? Float32(w) : 0.0f0
        for li in idx
            inds[n] = li
            wts[n] = w
            n += 1
        end
    end
    n -= 1          # number of used entries
    if n > length(inds)
        throw(BoundsError(inds, n))
    end
    return n
end

"""
    _project_qt_to_grid_kernel!(grid, leaf_inds, leaf_weights, n)

GPU/CPU kernel: for each packed (cell_index, weight) pair among the first `n`
entries, write the weight into `grid` at the position determined by the linear
cell index.

`leaf_inds` are linear indices over the `d × d` occupancy grid; the kernel
reconstructs `(col, row)` from the index and writes every vertical column
(gy ∈ 1:d) at that (gx, gz) with the weight — mirroring Taichi's
`set_obstacles` column-fill rule and `project_qt!`'s plane mapping.
"""
# GPU-only projection kernel.  Requires a real array to iterate so
# KernelAbstractions can resolve a backend — Base.OneTo(n) has none
# (get_backend(::Base.OneTo) is unimplemented).  leaf_inds/leaf_weights are
# fixed-capacity device buffers; only entries 1:n are valid, so the body
# guards on `i > n` (reading the *index* of the tail is safe — the garbage
# values are never used).
function _project_qt_to_grid_kernel!(grid,
                                     leaf_inds,
                                     leaf_weights,
                                     n::Int,
                                     obstacle_height::Int)
    d = size(grid, 1)
    AK.foreachindex(leaf_inds) do i
        i > n && return nothing
        li = leaf_inds[i]
        w = leaf_weights[i]
        w == 0.0f0 && return nothing
        col = ((li - 1) % d) + 1
        row = (li - 1) ÷ d + 1
        # vertical column: rise obstacle_height cells from the floor (bottom rows)
        for gy in 1:obstacle_height
            grid[col, gy, row] = w
        end
    end
    nothing
end

# Host-only projection: plain Julia loop, no AK.  Correct and fastest when
# grid/leaf buffers are CPU Arrays (leaf arrays are host-resident by
# construction — pack_leaf_cells! fills them on the host).
function _project_qt_to_grid!(grid,
                              leaf_inds,
                              leaf_weights,
                              n::Int,
                              obstacle_height::Int = size(grid, 2))
    d = size(grid, 1)
    @inbounds for i in 1:n
        li = leaf_inds[i]
        w = leaf_weights[i]
        w == 0.0f0 && continue
        col = ((li - 1) % d) + 1
        row = (li - 1) ÷ d + 1
        # vertical column: fill all y at this (gx, gz)
        for gy in 1:obstacle_height
            grid[col, gy, row] = w
        end
    end
    nothing
end

"""
    write_obstacles!(r::QuadTreeRenderer, qt::QuadTree)

Direct quadtree → GPU grid path: packs `qt.leaves` into flat
(cell_index, weight) arrays on the host, uploads them, then runs
the `_project_qt_to_grid_kernel!` to write straight into
`r.grid_material`. No intermediate dense `d×d` CPU matrix.
"""
function write_obstacles!(r::QuadTreeRenderer, qt::QuadTree)
    d = grid_res(r)
    lv = qt.leaves
    n = pack_leaf_cells!(r.pack_inds, r.pack_wts, lv, d)
    r.pack_n = n
    if r.grid_material isa CuArray
        # GPU: upload the used prefix (bulk H2D), then run the device kernel.
        # pack_inds/pack_wts are dense host Vectors of capacity d*d; resize!
        # to the used prefix is O(1) (length-only), copyto! is a bulk transfer.
        resize!(r.pack_inds, n)
        resize!(r.pack_wts, n)
        copyto!(r.leaf_inds, r.pack_inds)
        copyto!(r.leaf_weights, r.pack_wts)
        resize!(r.pack_inds, d * d)   # restore capacity for next pack
        resize!(r.pack_wts, d * d)
        _project_qt_to_grid_kernel!(r.grid_material, r.leaf_inds, r.leaf_weights, n, r.obstacle_height)
    else
        # CPU: host-resident buffers already hold the packed prefix; project
        # natively (no AK needed — this is a small scatter, not a pixel loop).
        _project_qt_to_grid!(r.grid_material, r.pack_inds, r.pack_wts, n, r.obstacle_height)
    end
    recompute_bbox!(r)   # keep bbox fresh after every obstacle write
    nothing
end

function grid_res(r::QuadTreeRenderer)
    r.grid_res
end

# ---------------------------------------------------------------------------
# Ray-marching helpers (device-inlined; called from the AK kernel below).
# Ported from renderer.py (dda_voxel / render).
#
# IMPORTANT (GPU constraint): kernel code must NOT construct SVector /
# StaticArray values — their constructors invoke dynamic tuple machinery
# (`jl_f_tuple` / `jl_f__svec_ref`) that the GPU compiler rejects. Kernel
# code therefore uses plain NTuple{3,Float32} and scalars only. SVectors
# are accepted as function inputs (constructed on the host) and converted
# to NTuples once, before the kernel launch.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# render! — integrated core path.
# The per-pixel pipeline (camera → DDA → transmittance accumulation →
# normalization) lives in components/voxel_renderer_core.jl and is tested
# there (C1–C7).  This file only adapts state: SVector camera → NTuple, and
# the (6,1,1) bbox array → NTuple{6,Float32}, then dispatches to the AK
# parallel render (backend chosen by array type).
# ---------------------------------------------------------------------------
function render!(r::QuadTreeRenderer)
    # host-side camera state (SVectors are fine here; kernel only sees NTuples)
    eye    = (r.camera_pos[1], r.camera_pos[2], r.camera_pos[3])
    target = (r.look_at[1],    r.look_at[2],    r.look_at[3])
    upv    = (r.up[1],         r.up[2],         r.up[3])

    # bbox is stored as a (6,1,1) array; the core expects a 6-tuple
    bbox_t = (r.bbox[1], r.bbox[2], r.bbox[3], r.bbox[4], r.bbox[5], r.bbox[6])

    # AK-parallel depth render into the (w,h,1) buffer (no views — AK
    # kernels over SubArrays of CuArrays fall back to scalar indexing)
    render_pixels_ak!(r.depth_buffer, r.grid_material, bbox_t,
                      r.voxel_inv_dx, r.voxel_dx, eye, target, upv, r.fov)

    # depth → [0,1] image (linear indexing is agnostic to the trailing 1-dim)
    normalize_depth!(r.rendered, r.depth_buffer)

    nothing
end

# recompute bbox from non-empty voxels (Taichi: atomic_min/max over grid)
function recompute_bbox!(r::QuadTreeRenderer)
    # Host-side bbox from the PACKED LEAVES (the authoritative, host-resident
    # set of occupied cells), not a scan of the device grid (which would be
    # scalar indexing on CuArray and is O(grid³) wasted work).
    # pack_inds/pack_wts hold the used prefix [1:n]; each linear index maps a
    # d×d occupancy cell to world XY via the same convention as the projection
    # kernel: (col,row) = (mod(li-1,d)+1, div(li-1,d)+1), centered at origin.
    n = r.pack_n
    d = grid_res(r)
    dx = r.voxel_dx
    minx = Float32(1e9); maxx = Float32(-1e9)
    miny = Float32(1e9); maxy = Float32(-1e9)
    minz = Float32(1e9); maxz = Float32(-1e9)
    @inbounds for i in 1:n
        w = r.pack_wts[i]
        w == 0.0f0 && continue
        li = r.pack_inds[i]
        col = (li - 1) % d + 1
        row = (li - 1) ÷ d + 1
        x = (col - 1 - d÷2) * dx
        y = (row - 1 - d÷2) * dx
        # vertical column: the cell spans y in [0, obstacle_height] (floor to
        # obstacle top); world z-extent comes from obstacle_height below.
        minx = min(minx, x); maxx = max(maxx, x + dx)
        miny = min(miny, y); maxy = max(maxy, y + dx)
    end
    if minx > maxx   # no leaves → default box around origin
        half = Float32(d) * dx / 2
        minx = miny = -half; maxx = maxy = half
    end
    # vertical extent: floor at gy=1 (world y = -d/2) rising obstacle_height cells
    y_floor = -(Float32(d) * dx / 2)
    minz = y_floor
    maxz = y_floor + Float32(r.obstacle_height) * dx   # top of the obstacles
    r.bbox[1] = minx; r.bbox[2] = miny; r.bbox[3] = minz
    r.bbox[4] = maxx; r.bbox[5] = maxy; r.bbox[6] = maxz
    nothing
end


# observe_pixels / random / logpdf — port of Scene.random / Scene.logpdf
function random(r::QuadTreeRenderer, var::Real)
    render!(r)
    # preallocated noise buffer; randn! works on both CuArray and Array
    noise_std = Float32(var)
    randn!(r.noise_buffer)
    r.noise_buffer .*= noise_std
    r.rendered .+ r.noise_buffer
end

function broadcast_logscore!(mu, x, noise::Float32)
    @assert size(mu) === size(x)

    noise_std = Float32(noise)
    log_noise_std = Float32(log(noise))
    
    AK.foreachindex(mu) do i
        z = (mu[i] - x[i]) / noise_std
        zsqr = z * z
        mu[i] = -0.5f0 * (zsqr + 1.8378773f0) - log_noise_std
    end

    AK.reduce(+, mu; init = zero(eltype(mu)))
end

function logpdf(r::QuadTreeRenderer, img, var::Real)
    render!(r)
    # move the observation to the active backend (host Array → CuArray on GPU)
    obs = (isa(r.rendered, CuArray) && !isa(img, CuArray)) ?
        CuArray(img) : img
    # mapreduce pairs img[I] and rendered[I]; reduces with +; backend-agnostic
    broadcast_logscore!(r.rendered, obs, var)
end

function observe_pixels(r::QuadTreeRenderer; var::Real = 0.001)
    # Return a noisy observation (like TaichiScene.random)
    random(r, var)
end

function save_img(img::AbstractArray, fname::AbstractString)
    arr = Array(img)
    # depth image: write PGM (P5) grayscale; depth normalized to [0,1]
    h, w = size(arr, 1), size(arr, 2)
    open(fname, "w") do io
        write(io, "P5\n$w $h\n255\n")
        for i in 1:w, j in 1:h
            v = clamp(round(Int, arr[i, j]*255), 0, 255)
            write(io, UInt8(v))
        end
    end
    nothing
end


function depth_map_array(depth_buffer)
    # Note the transpose (depth') so index 1 (u) maps to horizontal columns,
    # and index 2 (v) maps to vertical rows.
    depth = Array(depth_buffer)[:, :, 1]
    hits = depth .> 0.0f0
    d_display = zeros(Float32, size(depth))
    if any(hits)
        d_min, d_max = extrema(depth[hits])
        d_display[hits] .= one(Float32) .- (depth[hits] .- d_min) ./ max(d_max - d_min, Float32(1E-4))
    end

    # Transpose sime JuliaImages expects hxw and depth buffer is wxh
    Gray.(d_display')
end

function depth_map(r::QuadTreeRenderer)
    depth_map_array(r.depth_buffer)
end
