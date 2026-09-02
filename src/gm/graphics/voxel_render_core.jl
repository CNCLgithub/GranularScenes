# ============================================================================
# voxel_renderer_core.jl — consolidated component ladder (C1–C6).
#
# A voxel-based renderer core: camera → ray → DDA → multi-bounce
# accumulation → full per-pixel image → normalize_depth.
#
# Design rules (learned from GPUCompiler/AK failures):
#   - Pure Julia, Float32 everywhere, no Float64 leaks into hot paths
#   - NTuple{3,Float32} for 3-vectors; NO SVector construction anywhere
#     (StaticArrays constructors use jl_f_tuple/jl_f__svec_ref: not GPU-safe)
#   - No AK / CUDA / GPU code in this file — it is the CPU reference that the
#     AK kernel port (C7+) must match byte-for-byte
#   - Every function is unit-testable in isolation
# ============================================================================

# ---------------------------------------------------------------------------
# C1: scalar vector math
# ---------------------------------------------------------------------------

using Random: randn!

# normalize a 3-vector (NTuple{3,Float32} in/out)
function normalize3(v::NTuple{3,Float32})
    inv_len = 1.0f0 / sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    (v[1] * inv_len, v[2] * inv_len, v[3] * inv_len)
end

# cross product (NTuple{3,Float32} in/out)
function cross3(a::NTuple{3,Float32}, b::NTuple{3,Float32})
    (a[2] * b[3] - a[3] * b[2],
     a[3] * b[1] - a[1] * b[3],
     a[1] * b[2] - a[2] * b[1])
end

# dot product (scalar out)
function dot3(a::NTuple{3,Float32}, b::NTuple{3,Float32})
    a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

# ---------------------------------------------------------------------------
# C2: ray / AABB intersection (slab method)
# ---------------------------------------------------------------------------
# bbox as (xmin, ymin, zmin, xmax, ymax, zmax), Float32.
# Returns (t_near, t_far, hits).  Caller must ensure dir has no zero components.
# ---------------------------------------------------------------------------

function ray_aabb_intersection(eye::NTuple{3,Float32},
                               dir::NTuple{3,Float32},
                               bbox::NTuple{6,Float32})
    bmin = (bbox[1], bbox[2], bbox[3])
    bmax = (bbox[4], bbox[5], bbox[6])
    t_near = -Inf32
    t_far = Inf32
    hits = true
    for axis in 1:3
        origin = eye[axis]
        lo = bmin[axis]
        hi = bmax[axis]
        dir_comp = dir[axis]               # must be nonzero
        t1 = (lo - origin) / dir_comp
        t2 = (hi - origin) / dir_comp
        t_near = max(min(t1, t2), t_near)
        t_far = min(max(t1, t2), t_far)
    end
    if t_near > t_far
        hits = false
    end
    (t_near, t_far, hits)
end

# ---------------------------------------------------------------------------
# C3: camera model — basis + per-pixel ray direction
# ---------------------------------------------------------------------------
# Model (from Taichi renderer.py get_cast_dir):
#   forward = normalize(look_at - eye)          # camera view axis
#   right   = normalize(forward × up)           # camera right
#   up_dir  = normalize(right × forward)        # camera up (perpendicular)
#   for pixel (u, v) in an image of size (w, h):
#       fu = 2·fov·u/h − fov·aspect − 1e-5      # horizontal offset
#       fv = 2·fov·v/h − fov − 1e-5             # vertical offset
#       dir = normalize(forward + fu·right + fv·up_dir)
# ---------------------------------------------------------------------------

# camera basis from eye/look_at/up → (forward, right, up_dir)
function camera_basis(eye::NTuple{3,Float32},
                      target::NTuple{3,Float32},
                      up::NTuple{3,Float32})
    fwd = (target[1] - eye[1], target[2] - eye[2], target[3] - eye[3])
    fn = sqrt(fwd[1]^2 + fwd[2]^2 + fwd[3]^2)
    if fn == 0.0f0
        error("eye == target: undefined view direction")
    end
    fwd = (fwd[1] / fn, fwd[2] / fn, fwd[3] / fn)
    # right = normalize(forward × up)
    rx  = fwd[2]*up[3] - fwd[3]*up[2]
    ry  = fwd[3]*up[1] - fwd[1]*up[3]
    rz  = fwd[1]*up[2] - fwd[2]*up[1]
    rn  = sqrt(rx^2 + ry^2 + rz^2)
    if rn == 0.0f0
        error("forward parallel to up: degenerate camera")
    end
    right = (rx/rn, ry/rn, rz/rn)
    # up_dir = right × forward
    ux  = right[2]*fwd[3] - right[3]*fwd[2]
    uy  = right[3]*fwd[1] - right[1]*fwd[3]
    uz  = right[1]*fwd[2] - right[2]*fwd[1]
    un  = sqrt(ux^2 + uy^2 + uz^2)
    up_dir = (ux/un, uy/un, uz/un)
    (fwd, right, up_dir)
end

# per-pixel ray direction from a precomputed basis
function ray_direction(forward::NTuple{3,Float32},
                       right::NTuple{3,Float32},
                       up_dir::NTuple{3,Float32},
                       fov::Float32,
                       aspect::Float32,
                       u::Int, v::Int,
                       image_h::Int)
    fu = 2 * fov * u / image_h - fov * aspect - 1.0f-5
    fv = 2 * fov * v / image_h - fov - 1.0f-5
    dx = forward[1] + fu*right[1] + fv*up_dir[1]
    dy = forward[2] + fu*right[2] + fv*up_dir[2]
    dz = forward[3] + fu*right[3] + fv*up_dir[3]
    dn = sqrt(dx^2 + dy^2 + dz^2)
    (dx/dn, dy/dn, dz/dn)
end

# ---------------------------------------------------------------------------
# C4: DDA voxel traversal along a single ray
# ---------------------------------------------------------------------------
# Semantics (ported faithfully from Taichi renderer.py dda_voxel):
#   - world = cube [-1, 1]^3, grid_res n, cell dx = 2/n, inv_dx = n/2
#   - ray origin `eye`, direction `dir` (NORMALIZED), both NTuple{3,Float32}
#   - bbox = (xmin,ymin,zmin,xmax,ymax,zmax) occupied AABB (world units)
#   - starts at t_entry = max(t_near, 0) + 5·EPS inside the AABB
#   - steps voxels; first voxel with density > EPS after step 0 → HIT
#   - returns (hit_distance, hit_weight, hit_pos) or (Inf32, 0, origin_marker)
# ---------------------------------------------------------------------------

# NOTE: EPS defined once here; C5 uses the same value.
const EPS = 1.0f-4
const MAX_RAY_DEPTH = 10
const DIS_LIMIT = 200.0f0

function dda_voxel_march(material::AbstractArray{Float32,3},
                         bbox::NTuple{6,Float32},
                         voxel_inv_dx::Float32,
                         voxel_dx::Float32,
                         eye::NTuple{3,Float32},
                         ray_dir::NTuple{3,Float32})
    n = size(material, 1)
    # clamp near-zero direction components (avoid div-by-zero in rinv)
    dir = (abs(ray_dir[1]) < 1.0f-6 ? 1.0f-6 : ray_dir[1],
           abs(ray_dir[2]) < 1.0f-6 ? 1.0f-6 : ray_dir[2],
           abs(ray_dir[3]) < 1.0f-6 ? 1.0f-6 : ray_dir[3])
    rinv = (1.0f0 / dir[1], 1.0f0 / dir[2], 1.0f0 / dir[3])
    rsign = (dir[1] > 0 ? 1 : -1,
             dir[2] > 0 ? 1 : -1,
             dir[3] > 0 ? 1 : -1)

    # ---- AABB slab test (same as C2, inlined with clamped dir) ----
    bmin = (bbox[1], bbox[2], bbox[3])
    bmax = (bbox[4], bbox[5], bbox[6])
    t_near = -Inf32; t_far = Inf32; hits = true
    for axis in 1:3
        origin = eye[axis]; lo = bmin[axis]; hi = bmax[axis]; dc = dir[axis]
        t1 = (lo - origin) / dc; t2 = (hi - origin) / dc
        t_near = max(min(t1, t2), t_near)
        t_far  = min(max(t1, t2), t_far)
    end
    if t_near > t_far
        hits = false
    end

    hit_distance = Inf32
    hit_weight = 0.0f0
    hit_pos = (0.0f0, 0.0f0, 0.0f0)

    if hits
        t_entry = max(t_near, 0.0f0)
        t_start = t_entry + 5 * EPS
        pos = (eye[1] + dir[1] * t_start,
               eye[2] + dir[2] * t_start,
               eye[3] + dir[3] * t_start)
        grid_pos = (voxel_inv_dx * pos[1], voxel_inv_dx * pos[2], voxel_inv_dx * pos[3])
        voxel = (floor(Int, grid_pos[1]), floor(Int, grid_pos[2]), floor(Int, grid_pos[3]))

        # distance to next boundary per axis (direction units)
        travel = ((voxel[1] - grid_pos[1] + 0.5f0 + rsign[1] * 0.5f0) * rinv[1],
                  (voxel[2] - grid_pos[2] + 0.5f0 + rsign[2] * 0.5f0) * rinv[2],
                  (voxel[3] - grid_pos[3] + 0.5f0 + rsign[3] * 0.5f0) * rinv[3])
        step = 0
        tracing = true
        while tracing
            grid_index_x = voxel[1] + n ÷ 2 + 1
            grid_index_y = voxel[2] + n ÷ 2 + 1
            grid_index_z = voxel[3] + n ÷ 2 + 1
            inside = (1 <= grid_index_x <= n) && (1 <= grid_index_y <= n) && (1 <= grid_index_z <= n)
            density = 0.0f0
            if inside
                density = clamp(material[grid_index_x, grid_index_y, grid_index_z], 0.0f0, 1.0f0)
            end
            if step > 0 && density > EPS
                exit_travel = ((voxel[1] - grid_pos[1] + 0.5f0 - rsign[1] * 0.5f0) * rinv[1],
                               (voxel[2] - grid_pos[2] + 0.5f0 - rsign[2] * 0.5f0) * rinv[2],
                               (voxel[3] - grid_pos[3] + 0.5f0 - rsign[3] * 0.5f0) * rinv[3])
                hit_distance = max(max(exit_travel[1], exit_travel[2]), exit_travel[3]) * voxel_dx + t_entry
                hit_pos = (eye[1] + dir[1] * (hit_distance + 1.0f-3),
                           eye[2] + dir[2] * (hit_distance + 1.0f-3),
                           eye[3] + dir[3] * (hit_distance + 1.0f-3))
                hit_weight = density
                tracing = false
            else
                axis = (travel[1] <= travel[2] && travel[1] < travel[3]) ? 1 :
                       (travel[2] <= travel[1] && travel[2] <= travel[3]) ? 2 : 3
                travel = (travel[1] + (axis == 1 ? rsign[1] * rinv[1] : 0.0f0),
                          travel[2] + (axis == 2 ? rsign[2] * rinv[2] : 0.0f0),
                          travel[3] + (axis == 3 ? rsign[3] * rinv[3] : 0.0f0))
                voxel = (voxel[1] + (axis == 1 ? rsign[1] : 0),
                         voxel[2] + (axis == 2 ? rsign[2] : 0),
                         voxel[3] + (axis == 3 ? rsign[3] : 0))
                step += 1
            end
            if !inside
                tracing = false
            end
        end
    end

    (hit_distance, hit_weight, hit_pos)
end

# ---------------------------------------------------------------------------
# C5: multi-bounce depth accumulation (the per-pixel render loop)
# ---------------------------------------------------------------------------
# Ported from Taichi renderer.py render():
#   pos = eye; total_distance = 0; transmittance = 1; depth = 0
#   for bounce in 1:MAX_RAY_DEPTH
#       dist, w, new_pos = march(pos, dir)
#       total_distance += dist
#       depth += transmittance * w * total_distance
#       transmittance *= (1 - w)
#       pos = new_pos
#   end
#   depth += transmittance * total_distance
#
# NOTE: dda_voxel_march returns dist=Inf32 on a miss. 0·Inf = NaN, so we must
# treat a miss as "ray escaped": stop marching and return current depth.
# ---------------------------------------------------------------------------

# returns accumulated depth for one ray
function accumulate_depth(material::AbstractArray{Float32,3},
                          bbox::NTuple{6,Float32},
                          voxel_inv_dx::Float32,
                          voxel_dx::Float32,
                          eye::NTuple{3,Float32},
                          dir::NTuple{3,Float32})
    pos = eye
    total_distance = 0.0f0
    transmittance = 1.0f0
    depth = 0.0f0
    bounce = 0
    while bounce < MAX_RAY_DEPTH && transmittance > EPS && total_distance < DIS_LIMIT
        dist, w, new_pos = dda_voxel_march(material, bbox, voxel_inv_dx, voxel_dx, pos, dir)
        if !isfinite(dist)      # miss: escaped the density field
            break
        end
        total_distance += dist
        depth += transmittance * w * total_distance
        transmittance *= (1.0f0 - w)
        pos = new_pos
        bounce += 1
    end
    depth += transmittance * total_distance
    depth
end

# ---------------------------------------------------------------------------
# C6: full per-pixel render (CPU-serial reference)
# ---------------------------------------------------------------------------
# Loops over pixels calling C3 (camera) + C5 (accumulate).
# The AK port (C7) replaces ONLY the outer loops with AK.foreachindex while
# keeping this per-pixel body identical — hence testing this validates the
# future kernel.
# ---------------------------------------------------------------------------

# renders a (w × h) depth image into `depth_buffer` (Array{Float32,2})
function render_pixels!(depth_buffer::AbstractArray{Float32,2},
                        material::AbstractArray{Float32,3},
                        bbox::NTuple{6,Float32},
                        voxel_inv_dx::Float32,
                        voxel_dx::Float32,
                        eye::NTuple{3,Float32},
                        target::NTuple{3,Float32},
                        up::NTuple{3,Float32},
                        fov::Float32)
    image_h = size(depth_buffer, 2)
    image_w = size(depth_buffer, 1)
    aspect = Float32(image_w) / image_h

    fwd, right, up_dir = camera_basis(eye, target, up)

    for v in 1:image_h
        for u in 1:image_w
            dir = ray_direction(fwd, right, up_dir, fov, aspect, u, v, image_h)
            depth = accumulate_depth(material, bbox, voxel_inv_dx, voxel_dx, eye, dir)
            depth_buffer[u, v] = depth
        end
    end
    depth_buffer
end

# normalize depth to [0, 1]: rendered = max(0, (d - dmin) / dmax)
function normalize_depth!(rendered::AbstractArray, depth_buffer::AbstractArray,
                          dmin::Float32=1.3f0, dmax::Float32=2.3f0)
    function scale_depth(x)
        max(0.0f0, (x - dmin) / dmax)
    end
    AK.map!(scale_depth, rendered, depth_buffer)
    # for I in eachindex(rendered)
    #     rendered[I] = max(0.0f0, (depth_buffer[I] - dmin) / dmax)
    # end
    rendered
end

# ---------------------------------------------------------------------------
# C7: AK-parallelized per-pixel render (optional, requires AcceleratedKernels)
# ---------------------------------------------------------------------------
# The ONLY change from C6's render_pixels! is the outer loop:
#   - nested `for v / for u` becomes `AcceleratedKernels.foreachindex(...)`
#     with a LINEAR index I (AK yields linear indices, not CartesianIndex)
#   - the per-pixel body is byte-identical to C6
#   - array annotations are relaxed (untyped) so the same code compiles for
#     Array (CPU) and CuArray (GPU); kernel-called helpers are @inline
#
# `AcceleratedKernels` is referenced fully-qualified at call time, so this
# section can live in the same file as the pure-CPU reference: it only needs
# the package (or a test mock) in the calling environment.
# ---------------------------------------------------------------------------

@inline function pixel_accumulate(material,
                                  bbox::NTuple{6,Float32},
                                  voxel_inv_dx::Float32,
                                  voxel_dx::Float32,
                                  eye::NTuple{3,Float32},
                                  fwd::NTuple{3,Float32},
                                  right::NTuple{3,Float32},
                                  up_dir::NTuple{3,Float32},
                                  fov::Float32,
                                  aspect::Float32,
                                  u::Int, v::Int,
                                  image_h::Int)
    dir = ray_direction(fwd, right, up_dir, fov, aspect, u, v, image_h)
    accumulate_depth(material, bbox, voxel_inv_dx, voxel_dx, eye, dir)
end

# AK-parallel render (CPU Array or GPU CuArray, whichever is passed)
function render_pixels_ak!(depth_buffer,
                           material,
                           bbox::NTuple{6,Float32},
                           voxel_inv_dx::Float32,
                           voxel_dx::Float32,
                           eye::NTuple{3,Float32},
                           target::NTuple{3,Float32},
                           up::NTuple{3,Float32},
                           fov::Float32)
    image_h = size(depth_buffer, 2)
    image_w = size(depth_buffer, 1)
    aspect  = Float32(image_w) / image_h

    fwd, right, up_dir = camera_basis(eye, target, up)

    # full path: the do-block closure captures only isbits values
    AcceleratedKernels.foreachindex(depth_buffer) do I
        u = ((I - 1) % image_w) + 1
        v = ((I - 1) ÷ image_w) + 1
        depth = pixel_accumulate(material, bbox, voxel_inv_dx, voxel_dx, eye,
                                 fwd, right, up_dir, fov, aspect, u, v, image_h)
        depth_buffer[I] = depth
    end
    depth_buffer
end
