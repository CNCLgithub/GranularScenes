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
function encoder(rng=Random.default_rng(); image_shape::Dims{3}, max_num_filters::Int,
                 latent_dim::Int=32)
    return @compact(;
        embed=Chain(
            Chain(Conv((3, 3), image_shape[3] => max_num_filters ÷ 4; stride=2, pad=1),
                  BatchNorm(max_num_filters ÷ 4, leakyrelu)),
            Chain(Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=2, pad=1),
                  BatchNorm(max_num_filters ÷ 2, leakyrelu)),
            Chain(Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=2, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
        ),
        # Flat-vector projection, matching the working PyTorch VAE:
        # each latent dim sees the whole feature map (no local patching).
        # Grid-form latent can be restored later by swapping in a conv head.
        proj=Dense(max_num_filters * 32 * 32 => 2 * latent_dim, gelu),
        rng
    ) do x
        y = embed(x)             # (H/8, W/8, F, B) = (32, 32, F, B)
        y_flat = reshape(y, :, size(y, 4))   # (F·H/8·W/8, B) = (2048, B)
        proj_out = proj(y_flat)              # (2·latent_dim, B)
        μ = proj_out[1:latent_dim, :]        # (latent_dim, B)
        s_raw = proj_out[latent_dim+1:end, :] # (latent_dim, B)
        σ = softplus.(s_raw) .+ 1f-8      # σ>0, ≈0.69 at init (matches PyTorch)
        ϵ = randn_like(Lux.replicate(rng), σ)
        z = ϵ .* σ .+ μ
        @return z, μ, σ
    end
end

# ---------------------------------------------------------------------------
# Depth decoder: 32x32x1 latent grid -> 256x256x1 depth map (fully convolutional)
# ---------------------------------------------------------------------------
function depth_decoder(; max_num_filters::Int, latent_dim::Int=32,
                       grid_hw::Tuple{Int,Int}=(32, 32))
    return @compact(;
        # z-head: flat latent -> (16, 16, F/4) feature map (PyTorch Linear+Unflatten)
        z_head=Dense(latent_dim => (max_num_filters ÷ 4) * grid_hw[1] * grid_hw[2]),
        upchain=Chain(
            Chain(Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 2; stride=1, pad=1),
                  BatchNorm(max_num_filters ÷ 2, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters ÷ 2 => max_num_filters; stride=1, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters => max_num_filters; stride=1, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
            Chain(Upsample(2),
                  Conv((3, 3), max_num_filters => max_num_filters; stride=1, pad=1),
                  BatchNorm(max_num_filters, leakyrelu)),
            Conv((3, 3), max_num_filters => 1; stride=1, pad=1),
        )
    ) do x
        # x is flat z: (latent_dim, B)
        h = z_head(x)                                        # (F/4·16·16, B)
        h = reshape(h, grid_hw[1], grid_hw[2], max_num_filters ÷ 4, size(x, 2))
        @return sigmoid(upchain(h))
    end
end

# ---------------------------------------------------------------------------
# Occupancy decoder: 32x32x1 latent grid -> 16x16 occupancy grid
#
# A stride-2 conv downsamples 32 -> 16, and the stack must also learn the
# camera-frame -> top-down-frame transform.
# ---------------------------------------------------------------------------
function occupancy_decoder(; max_num_filters::Int, latent_dim::Int=32)
    return @compact(;
        # z-head: flat latent -> (32, 32, 1) grid consumed by the conv stack
        z_head=Dense(latent_dim => 32 * 32),
        chain=Chain(
            Conv((3, 3), 1 => max_num_filters ÷ 4; stride=1, pad=1),
            BatchNorm(max_num_filters ÷ 4, leakyrelu),
            Conv((3, 3), max_num_filters ÷ 4 => max_num_filters ÷ 4; stride=2, pad=1),
            BatchNorm(max_num_filters ÷ 4, leakyrelu),
            Conv((3, 3), max_num_filters ÷ 4 => 1; stride=1, pad=1),
        )
    ) do x
        h = z_head(x)                        # (32*32, B)
        h = reshape(h, 32, 32, 1, size(x, 2))
        @return sigmoid(chain(h))
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

function DepthOccVAE(rng=Random.default_rng(); image_shape::Dims{3}, max_num_filters::Int,
                     latent_dim::Int=32, grid_hw::Tuple{Int,Int}=(32, 32))
    enc = encoder(rng; image_shape, max_num_filters, latent_dim)
    depth_dec = depth_decoder(; max_num_filters, latent_dim, grid_hw)
    occ_dec = occupancy_decoder(; max_num_filters, latent_dim)
    return DepthOccVAE(enc, depth_dec, occ_dec)
end

function (m::DepthOccVAE)(x, ps, st)
    (z, μ, σ), st_enc = m.encoder(x, ps.encoder, st.encoder)
    x_rec, st_d = m.depth_decoder(z, ps.depth_decoder, st.depth_decoder)
    occ, st_o = m.occ_decoder(z, ps.occ_decoder, st.occ_decoder)
    return (x_rec, occ, μ, σ), (; encoder=st_enc, depth_decoder=st_d,
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
# KL divergence for diagonal Gaussians with σ = softplus(s):
#   KL = ½ Σ_d (μ_d² + σ_d² − log σ_d² − 1), averaged over the batch.
function kldiv_diag(μ, σ)
    σ² = σ .^ 2
    return sum(μ .^ 2 .+ σ² .- 2 .* log.(σ) .- 1) / 2 / size(μ, ndims(μ))
end

# Stage 1: VAE loss on depth maps only. The occupancy decoder branch is not
# evaluated, so no gradients flow into it (and no wasted compute).
function vae_loss_function(model, ps, st, X; β=1.0f0)
    (z, μ, σ), st_enc = model.encoder(X, ps.encoder, st.encoder)
    x_rec, st_d = model.depth_decoder(z, ps.depth_decoder, st.depth_decoder)
    depth_loss = MSELoss(; agg=mean)(x_rec, X)
    kldiv_loss = kldiv_diag(μ, σ)
    loss = depth_loss + β * kldiv_loss
    return loss, (; encoder=st_enc, depth_decoder=st_d, st.occ_decoder),
           (; depth_loss, kldiv_loss)
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
    @show size(X_cpu)
    @show size(x_rec_cpu)
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
