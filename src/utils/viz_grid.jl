################################################################################
# Rollout panels
#
# Frame helpers for inspecting the inference chain. Panels are kept at their
# native resolution (a `d×d` quadtree image, a 256×256 depth map) and upscaled
# only for display: `viz_tile` is an element-wise nearest-neighbour repeat, so
# a 16×16 occupancy grid becomes 16×16 blocks of 16 px, not a 16-fold tiling
# of the whole image.
################################################################################

export viz_tile, viz_depth, viz_err

const VIZ_TILE_SIZE = 256
const VIZ_BG        = RGB{Float64}(1, 1, 1)

"""
    viz_tile(img, T = VIZ_TILE_SIZE) -> Matrix{RGB{Float64}}

Element-wise nearest-neighbour upscale of `img` (any `Colorant` matrix: a
`d×d` quadtree cell image or a 256×256 depth map) to a `T×T` white canvas,
centred. The integer factor is `fld(T, max(size(img)...))` and the block size
is exact, so every source cell maps to a `k×k` square. Sources larger than `T`
are cropped from the centre rather than scaled.

`repeat(a, k, k)` would tile the whole array `k` times in each direction;
the `inner` keyword is what replicates individual elements.
"""
function viz_tile(img, T::Int = VIZ_TILE_SIZE)
    a = convert.(RGB{Float64}, img)
    k = max(1, fld(T, max(size(a)...)))
    up = repeat(a, inner = (k, k))
    out = fill(VIZ_BG, T, T)
    hu, wu = size(up)
    i0 = max(0, fld(T - hu, 2))
    j0 = max(0, fld(T - wu, 2))
    @inbounds for i = 1:min(hu, T), j = 1:min(wu, T)
        out[i0+i, j0+j] = up[i, j]
    end
    return out
end

"""
    viz_depth(buf; lo = nothing, hi = nothing) -> Matrix{Gray{Float64}}

Depth buffer (`w×h` or `w×h×1`, as returned by `qt_observe` / a vision trace)
to a grayscale image with near = bright, no-hit = white.

`lo`/`hi` set the depth range in world units; pass a fixed pair for frames that
are to be compared across iterations. When either is `nothing` the range falls
back to the per-image extrema of the hits, which makes each frame
self-normalising and therefore not comparable to its neighbours.
"""
function viz_depth(buf; lo = nothing, hi = nothing)
    z = Array(buf)
    z = ndims(z) == 3 ? z[:, :, 1] : z
    hits = z .> 0
    if lo === nothing || hi === nothing
        lo_, hi_ = any(hits) ? extrema(z[hits]) : (0.0, 1.0)
    else
        lo_, hi_ = lo, hi
    end
    v = clamp.(1 .- (Float64.(z) .- lo_) ./ max(hi_ - lo_, 1e-4), 0.0, 1.0)
    v[.!hits] .= 1.0
    return Gray.(permutedims(v))
end

"""
    viz_err(gt, pred; emax = 0.25) -> Matrix{Gray{Float64}}

Absolute difference of two grayscale images on a fixed `[0, emax]` scale,
white = exact, black = `emax` or worse. Fixed range so the panel does not
re-normalise per frame.
"""
function viz_err(gt::AbstractMatrix{<:Gray},
                 pred::AbstractMatrix{<:Gray};
                 emax::Float64 = 0.25)
    size(gt) == size(pred) ||
        throw(DimensionMismatch("$(size(gt)) vs $(size(pred))"))
    e = abs.([Float64(c.val) for c in gt] .- [Float64(c.val) for c in pred])
    return Gray.(1.0 .- clamp.(e ./ emax, 0.0, 1.0))
end
