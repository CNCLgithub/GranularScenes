# Depth-Map VAE with occupancy decoder — Flux implementation.
#
# This replaces the earlier Lux/Reactant version (abandoned 2026-10-07: stale
# Reactant compile traces, shape-1-batch BatchNorm issues). The architecture
# mirrors pydeps/og_proposal/vae.py, the reference implementation that worked:
#
#   encoder:  Conv5x5 s2 p2 -> swish, x3 (256 -> 128 -> 64 -> 32)
#             flatten -> Dense(FEAT*32*32 -> LATENT) -> mu | scale
#             scale = softplus(s) + 1e-8   (sigma > 0)
#   decoder:  Dense(LATENT -> FEAT*32*32) -> reshape (32,32,FEAT)
#             -> ConvTranspose5x5 s2, x3 -> sigmoid
#   occupancy decoder:
#             flat latent -> Dense(LATENT -> 32*32) -> reshape (32,32,1)
#             -> conv downsample -> (16,16) sigmoid
#
# Latent: flat 32-dim global code. The spatially-organized grid variant was
# tested and rejected (IoU 0.070 vs 0.38 for the flat code; see
# scripts/nn/{vae,occ}_spatial.jl for the negative result).
#
# Stage 1: train the VAE (encoder + depth decoder) on 256x256 depth maps
#          (scripts/nn/vae_flux.jl).
# Stage 2: freeze encoder/mu/scale; train the occupancy decoder mapping the
#          latent -> 16x16 occupancy grid (scripts/nn/occ_flux.jl). The
#          latent is a camera-frame code; the occupancy target is a top-down
#          frame, so the decoder learns the view transform.
#
# This file contains model definitions, losses, and visualization helpers only.
# The training loops live in scripts/nn/*.jl.

using Flux,
    Optimisers,
    Zygote,
    Functors,
    Random,
    Statistics,
    CUDA,
    FileIO,
    ImageCore,
    Printf

const LATENT = 32
const HIDDEN = 16
const FEAT   = HIDDEN ÷ 4        # channels at the 32x32 bottleneck

# ---------------------------------------------------------------------------
# VAE: encoder + depth decoder sharing a flat 32-dim latent
# ---------------------------------------------------------------------------
struct DepthVAE
    encoder::Chain
    mu::Dense
    scale::Dense
    decoder::Chain
end
@functor DepthVAE

function DepthVAE(rng::AbstractRNG=Random.default_rng())
    encoder = Chain(
        Conv((5, 5), 1 => HIDDEN; stride=2, pad=2), swish,
        Conv((5, 5), HIDDEN => 8;  stride=2, pad=2), swish,
        Conv((5, 5), 8 => FEAT;    stride=2, pad=2), swish,
        Flux.flatten,                             # (FEAT*32*32, B)
    )
    mu    = Dense(FEAT * 32 * 32 => LATENT)
    scale = Dense(FEAT * 32 * 32 => LATENT)
    decoder = Chain(
        Dense(LATENT => FEAT * 32 * 32),
        x -> reshape(x, 32, 32, FEAT, :),
        ConvTranspose((5, 5), FEAT => HIDDEN; stride=2, pad=SamePad()), swish,
        ConvTranspose((5, 5), HIDDEN => 8;    stride=2, pad=SamePad()), swish,
        ConvTranspose((5, 5), 8 => 1;         stride=2, pad=SamePad()),
        sigmoid,
    )
    return DepthVAE(encoder, mu, scale, decoder)
end

encode(m::DepthVAE, x::AbstractArray{Float32,4}) =
    (m.mu(m.encoder(x)), softplus.(m.scale(m.encoder(x))) .+ 1f-8)

# Reparameterized sampling; eps drawn on device (CUDA.randn when available).
# The RNG draw is cut from the AD graph with Zygote.@ignore — gradients flow
# through z = mu + sigma .* eps with eps treated as a leaf.
sample_noise(μ::AbstractArray{Float32}) =
    CUDA.functional() ? Zygote.@ignore(CUDA.randn(Float32, size(μ))) :
                        Zygote.@ignore(randn(Float32, size(μ)))

function sample(m::DepthVAE, x::AbstractArray{Float32,4})
    μ, σ = encode(m, x)
    return μ .+ σ .* sample_noise(μ), μ, σ
end

decode_depth(m::DepthVAE, z::AbstractMatrix{Float32}) = m.decoder(z)

# ---------------------------------------------------------------------------
# Occupancy decoder: flat latent -> 16x16 occupancy grid
# ---------------------------------------------------------------------------
struct OccDecoder
    head::Dense
    tail::Chain
end
@functor OccDecoder

OccDecoder(rng::AbstractRNG=Random.default_rng()) =
    OccDecoder(
        Dense(LATENT => 32 * 32),
        Chain(
            Conv((3, 3), 1 => 16; stride=1, pad=1), leakyrelu,
            Conv((3, 3), 16 => 16; stride=2, pad=1), leakyrelu,
            Conv((3, 3), 16 => 1;  stride=1, pad=1),
            sigmoid,
        ),
    )

decode_occ(occ::OccDecoder, z::AbstractMatrix{Float32}) =
    occ.tail(reshape(occ.head(z), 32, 32, 1, size(z, 2)))

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
# KL(N(mu, diag(sigma^2)) || N(0, I)) per sample, averaged over the batch.
# sigma is parameterized by scale (softplus + 1e-8), so log sigma^2 = 2 log sigma.
function kldiv_diag(μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32})
    logσ² = 2f0 .* log.(σ)
    return mean(@. -0.5f0 * (1f0 + logσ² - μ^2 - exp(logσ²)))
end

# Stage 1: VAE loss on depth maps.
function vae_loss(m::DepthVAE, x::AbstractArray{Float32,4};
                  β::Float32=0.01f0)
    μ, σ = encode(m, x)
    z = μ .+ σ .* sample_noise(μ)
    x̂ = decode_depth(m, z)
    mse = mean((x̂ .- x) .^ 2)
    kl  = kldiv_diag(μ, σ)
    return mse + β * kl, mse, kl
end

# Stage 2: occupancy loss, deterministic mu path (no sampling noise), MSE on
# sigmoid outputs — the recipe that beat unweighted BCE (which collapsed to
# the 11% positive-rate prior) and stochastic z.
function occ_loss(occ::OccDecoder, z::AbstractMatrix{Float32},
                  O::AbstractArray{Float32,4})
    p̂ = decode_occ(occ, z)
    mse = mean((p̂ .- O) .^ 2)
    acc = mean((p̂ .> 0.5f0) .== (O .> 0.5f0))
    return mse, acc
end

# ---------------------------------------------------------------------------
# Visualization: gray-scale helpers for panels / PNG strips
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
        catch err
            @warn "could not save panel to $filepath" err
        end
    end
    return panel
end
