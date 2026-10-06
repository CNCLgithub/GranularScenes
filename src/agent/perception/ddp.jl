# Depth-Map VAE with spatially-organized latent and occupancy decoder.
# Adapted from the Lux.jl Convolutional VAE tutorial:
# https://lux.csail.mit.edu/stable/tutorials/intermediate/5_ConvolutionalVAE
#
# This file contains model definition, losses, and visualization helpers only.
# The training loop lives in notebooks/ddp.jl (Pluto notebook).
#
# Latent: 32x32x1 grid (spatially organized, per-cell Gaussians). The mu and
#         logvar projections are 3x3 convs to 1 channel, so each latent cell
#         summarizes a local neighborhood of the final feature map.
#
# Stage 1: train the Beta-VAE (encoder + depth decoder) on 256x256 depth maps.
# Stage 2: freeze encoder and depth decoder; train the occupancy decoder
#          mapping the 32x32x1 latent grid -> 16x16 occupancy grid. The latent
#          grid is 32x32 (camera frame); the occupancy target is 16x16
#          (top-down frame), so the decoder must both downsample and learn the
#          view transform.
#
# Data loading is a stub; plug in your own file paths in `load_depth_dataset`.

using Lux,
    Reactant,
    Enzyme,
    Random,
    Statistics,
    Optimisers,
    MLUtils,
    ConcreteStructs,
    Printf

const xdev = reactant_device(; force=true)
const cdev = cpu_device()

# ---------------------------------------------------------------------------
# Encoder: 256x256x1 depth map -> (z [32x32x1], mu [32x32x1], logvar [32x32x1])
# ---------------------------------------------------------------------------
function encoder(rng=Random.default_rng(); image_shape::Dims{3}, max_num_filters::Int)
    return @compact(;
        embed=Chain(
            Chain(Conv((3, 3), image_shape[3] => max_num_filters ÷ 4; stride=2, pad=1),
                  BatchNorm(max_num_filters ÷ 4, leakyrelu)),
            Chain(Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=2, pad=1),
                  BatchNorm(max_num_filters ÷ 2, leakyrelu)),
            Chain(Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=2, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
        ),
        # 3x3 convs to 1 channel: each latent cell sees a 3x3 neighborhood of
        # the last feature map. No flatten, no Dense — the latent stays a grid.
        proj_mu=Conv((3, 3), max_num_filters => 1; stride=1, pad=1),
        proj_log_var=Conv((3, 3), max_num_filters => 1; stride=1, pad=1),
        rng
    ) do x
        y = embed(x)             # (H/8, W/8, F, B) = (32, 32, F, B)
        μ = proj_mu(y)           # (32, 32, 1, B)
        logσ² = proj_log_var(y)  # (32, 32, 1, B)
        T = eltype(logσ²)
        logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0))
        σ = exp.(logσ² .* T(0.5))
        ϵ = randn_like(Lux.replicate(rng), σ)
        z = ϵ .* σ .+ μ
        @return z, μ, logσ²
    end
end

# ---------------------------------------------------------------------------
# Depth decoder: 32x32x1 latent grid -> 256x256x1 depth map (fully convolutional)
# ---------------------------------------------------------------------------
function depth_decoder(; max_num_filters::Int)
    return @compact(;
        upchain=Chain(
            Chain(Conv((3, 3), 1 => max_num_filters; stride=1, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters => max_num_filters ÷ 2; stride=1, pad=1),
                  BatchNorm(max_num_filters ÷ 2, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters ÷ 2 => max_num_filters ÷ 4; stride=1, pad=1),
                  BatchNorm(max_num_filters ÷ 4, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 4; stride=1, pad=1),
                  BatchNorm(max_num_filters ÷ 4, leakyrelu)),
            Conv((3, 3), max_num_filters ÷ 4 => 1; stride=1, pad=1),
        )
    ) do x
        @return sigmoid(upchain(x))
    end
end

# ---------------------------------------------------------------------------
# Occupancy decoder: 32x32x1 latent grid -> 16x16 occupancy grid
#
# A stride-2 conv downsamples 32 -> 16, and the stack must also learn the
# camera-frame -> top-down-frame transform.
# ---------------------------------------------------------------------------
function occupancy_decoder(; max_num_filters::Int)
    return @compact(;
        chain=Chain(
            Conv((3, 3), 1 => max_num_filters ÷ 4; stride=1, pad=1),
            BatchNorm(max_num_filters ÷ 4, leakyrelu),
            Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 4; stride=2, pad=1),
            BatchNorm(max_num_filters ÷ 4, leakyrelu),
            Conv((3, 3), max_num_filters ÷ 4 => 1; stride=1, pad=1),
        )
    ) do x
        @return sigmoid(chain(x))
    end
end

# ---------------------------------------------------------------------------
# Full model: encoder + depth decoder + occupancy decoder sharing the latent
# ---------------------------------------------------------------------------
@concrete struct DepthOccVAE <:
    AbstractLuxContainerLayer{(:encoder, :depth_decoder, :occ_decoder)}
    encoder <: AbstractLuxLayer
    depth_decoder <: AbstractLuxLayer
    occ_decoder <: AbstractLuxLayer
end

function DepthOccVAE(rng=Random.default_rng(); image_shape::Dims{3},
                     max_num_filters::Int)
    enc = encoder(rng; image_shape, max_num_filters)
    depth_dec = depth_decoder(; max_num_filters)
    occ_dec = occupancy_decoder(; max_num_filters)
    return DepthOccVAE(enc, depth_dec, occ_dec)
end

function (m::DepthOccVAE)(x, ps, st)
    (z, μ, logσ²), st_enc = m.encoder(x, ps.encoder, st.encoder)
    x_rec, st_d = m.depth_decoder(z, ps.depth_decoder, st.depth_decoder)
    occ, st_o = m.occ_decoder(z, ps.occ_decoder, st.occ_decoder)
    return (x_rec, occ, μ, logσ²), (; encoder=st_enc, depth_decoder=st_d,
                                     occ_decoder=st_o)
end

function encode(m::DepthOccVAE, x, ps, st)
    (z, _, _), st_enc = m.encoder(x, ps.encoder, st.encoder)
    return z, (; encoder=st_enc, st.depth_decoder, st.occ_decoder)
end

function decode_depth(m::DepthOccVAE, z, ps, st)
    x_rec, st_d = m.depth_decoder(z, ps.depth_decoder, st.depth_decoder)
    return x_rec, (; depth_decoder=st_d, st.encoder, st.occ_decoder)
end

function decode_occ(m::DepthOccVAE, z, ps, st)
    occ, st_o = m.occ_decoder(z, ps.occ_decoder, st.occ_decoder)
    return occ, (; occ_decoder=st_o, st.encoder, st.depth_decoder)
end

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
# KL divergence for per-cell independent Gaussians over the spatial grid.
# Sum over spatial cells and channels, averaged over the batch.
function kldiv_per_cell(μ, logσ²)
    return -sum(1 .+ logσ² .- μ .^ 2 .- exp.(logσ²)) / 2 / size(μ, ndims(μ))
end

# Stage 1: VAE loss on depth maps only. The occupancy decoder branch is not
# evaluated, so no gradients flow into it (and no wasted compute).
function vae_loss_function(model, ps, st, X; β=1.0f0)
    (z, μ, logσ²), st_enc = model.encoder(X, ps.encoder, st.encoder)
    x_rec, st_d = model.depth_decoder(z, ps.depth_decoder, st.depth_decoder)
    depth_loss = MSELoss(; agg=mean)(x_rec, X)
    kldiv_loss = kldiv_per_cell(μ, logσ²)
    loss = depth_loss + β * kldiv_loss
    return loss, (; encoder=st_enc, depth_decoder=st_d, st.occ_decoder),
           (; x_rec, μ, logσ², depth_loss, kldiv_loss)
end

function occ_loss_function(model, ps, st, (z, O))
    occ, st_o = model(z, ps, st)
    occ_loss = BinaryCrossEntropyLoss(; agg=mean)(occ, O)
    return occ_loss, st_o, (; occ,)
end

# # ---------------------------------------------------------------------------
# # Data loading (stub)
# # ---------------------------------------------------------------------------
# # Replace this with your dataset loader. It must return an MLUtils.DataLoader of
# # (X, O) pairs where
# #   X : Float32[256, 256, 1, batch]  depth map in [0, 1]
# #   O : Float32[16, 16, 1, batch]    occupancy grid in {0, 1}
# @concrete struct DepthOccDataset
#     depth::AbstractArray{Float32,4}   # (256, 256, 1, N)
#     occ::AbstractArray{Float32,4}     # (16, 16, 1, N)
# end

# function Base.getindex(ds::DepthOccDataset, idxs::AbstractVector{<:Integer})
#     return (ds.depth[:, :, :, idxs], ds.occ[:, :, :, idxs])
# end

# function Base.length(ds::DepthOccDataset)
#     return size(ds.depth, 4)
# end

# function load_depth_dataset(; batchsize)
#     # TODO: load your real depth maps and occupancy grids here.
#     # Below is random dummy data so the script runs end-to-end.
#     N = parse(Bool, get(ENV, "CI", "false")) ? 512 : 4096
#     depth = rand(Float32, 256, 256, 1, N)
#     occ = Float32.(rand(Float32, 16, 16, 1, N) .> 0.5)
#     ds = DepthOccDataset(depth, occ)
#     return DataLoader(ds; batchsize, shuffle=true, partial=false)
# end

# ---------------------------------------------------------------------------
# Test-set visualization: 3x4 grid
# rows = scenes, columns = | GT depth | Recon Depth | GT occupancy | Recon Occ |
# ---------------------------------------------------------------------------
"""
    gray01(A) -> Matrix{Gray}

Clamp a Float32 matrix to [0, 1] and wrap it as a `Gray` image
(`Matrix{Gray{Float32}}`), displayable in Pluto and renderable via
ImageIO/FileIO (`save("x.png", img)`).
"""
function gray01(A::AbstractMatrix{<:Real})
    return Gray.(clamp.(A, 0.0f0, 1.0f0))
end

"""
    occ_gray(O; up) -> Matrix{Gray}

Occupancy map as a grayscale image, nearest-neighbor upsampled by an
integer factor `up` (16 for a 16x16 grid -> 256x256) so it can be tiled
into the panel next to the 256x256 depth maps.
"""
function occ_gray(O::AbstractMatrix{<:Real}; up::Int=16)
    big = repeat(clamp.(O, 0.0f0, 1.0f0); inner=(up, up))
    return Gray.(big)
end
function plot_vae_panels(X_cpu, O_cpu, x_rec_cpu, occ_cpu; filepath=nothing)
    n = size(X_cpu, 4)
    # Assemble the panel image: rows are scenes, columns are
    # [GT depth, Recon depth, GT occ, Recon occ]
    panel = hcat(
        vcat([gray01(X_cpu[:, :, 1, i])   for i in 1:n]...),
        vcat([gray01(x_rec_cpu[:, :, 1, i]) for i in 1:n]...),
        vcat([occ_gray(O_cpu[:, :, 1, i]; up=16) for i in 1:n]...),
        vcat([occ_gray(occ_cpu[:, :, 1, i]; up=16) for i in 1:n]...),
    )

    if filepath !== nothing
        try
            save(filepath, panel)
        catch e
            @warn "ImageIO not available; skipping PNG write" exception=e
        end
    end
    return panel
end
