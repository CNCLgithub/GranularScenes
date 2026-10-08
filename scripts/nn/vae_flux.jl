# Flux-based VAE — drop-in replacement for the Lux/Reactant implementation.
# Mirrors pydeps/og_proposal/vae.py (the reference that "worked"):
#   encoder:  Conv5x5 s2 p2 -> SiLU, x3 (256 -> 128 -> 64 -> 32)
#             flatten -> Dense -> mu | scale ; scale = softplus + 1e-8
#   decoder:  Dense -> reshape 16ch @ 32x32 -> ConvTranspose5x5 s2 (x3) -> sigmoid
# No BatchNorm (matching the reference) => no batch-size-1 problems.
#
# Usage:  julia --project=. scripts/vae_flux.jl
# Env:    DDP_DATA, DDP_EPOCHS, DDP_BATCH, DDP_BETA, DDP_LR

using HDF5, Flux, Optimisers, Zygote, Random, Statistics, Printf, cuDNN
import CUDA

const DATA    = get(ENV, "DDP_DATA",   "/spaths/datasets/ddp_train_11f_32x32.hdf5")
const EPOCHS  = parse(Int,   get(ENV, "DDP_EPOCHS", "30"))
const BATCH   = parse(Int,   get(ENV, "DDP_BATCH",  "32"))
const BETA    = parse(Float32, get(ENV, "DDP_BETA", "0.001"))
const LR      = parse(Float32, get(ENV, "DDP_LR",   "1e-3"))
const HIDDEN  = 16
const LATENT  = 32
const IMG     = 256
const FEAT    = HIDDEN ÷ 4                  # channels at the 32x32 bottleneck

# ---------------------------------------------------------------- model ----
function make_vae(rng=Random.default_rng())
    encoder = Chain(
        Conv((5, 5), 1 => HIDDEN;      stride=2, pad=2), swish,
        Conv((5, 5), HIDDEN => 8;      stride=2, pad=2), swish,
        Conv((5, 5), 8 => FEAT;        stride=2, pad=2), swish,
        Flux.flatten,                             # (FEAT*32*32, B)
    )
    mu_proj    = Dense(FEAT * 32 * 32 => LATENT)
    scale_proj = Dense(FEAT * 32 * 32 => LATENT)
    decoder = Chain(
        Dense(LATENT => FEAT * 32 * 32),
        x -> reshape(x, 32, 32, FEAT, :),
        ConvTranspose((5, 5), FEAT => 8;  stride=2, pad=SamePad()), swish,
        ConvTranspose((5, 5), 8 => HIDDEN; stride=2, pad=SamePad()), swish,
        ConvTranspose((5, 5), HIDDEN => 1; stride=2, pad=SamePad()),
        sigmoid,
    )
    return (encoder=encoder, mu=mu_proj, scale=scale_proj, decoder=decoder)
end

function encode(m, x)
    h = m.encoder(x)                       # (FEAT*32*32, B)
    μ = m.mu(h)                            # (LATENT, B)
    σ = softplus.(m.scale(h)) .+ 1f-8      # strictly positive, like the reference
    return μ, σ
end

decode(m, z) = m.decoder(z)

function forward(m, x, rng)
    μ, σ = encode(m, x)
    z = μ .+ σ .* Zygote.@ignore CUDA.randn(Float32, size(μ))
    x̂ = decode(m, z)
    return μ, σ, x̂
end

# ---------------------------------------------------------------- loss -----
function vae_loss(m, x, rng)
    μ, σ, x̂ = forward(m, x, rng)
    mse = mean((x̂ .- x) .^ 2)
    # KL(N(μ, diag(σ²)) || N(0, I)) per sample, averaged over batch.
    # logσ² = 2 log σ since the reference parameterizes by scale, not logvar.
    logσ² = 2f0 .* log.(σ)
    kl = mean(@. -0.5f0 * (1f0 + logσ² - μ^2 - exp(logσ²)))
    return mse + BETA * kl, mse, kl
end

# ---------------------------------------------------------------- data -----
function load_data(path)
    h5open(path, "r") do f
        X = Float32.(read(f["depth"]))     # (256, 256, 1, N)
        O = Float32.(read(f["occ"]))
        X_min, X_max = extrema(X)
        X .-= X_min
        X .*= 1.0f0 / X_max
        @assert ndims(X) == 4 && size(X, 3) == 1
        return X, O
    end
end

# ---------------------------------------------------------------- main -----
function main()
    rng = Xoshiro(0)
    use_gpu = CUDA.functional()
    dev = use_gpu ? (x -> fmap(CUDA.cu, x)) : Flux.cpu
    use_gpu && @info "GPU: " * CUDA.name(CUDA.device())

    @printf "Loading %s ...\n" DATA
    X, _O = load_data(DATA)
    N = size(X, 4)
    @printf "dataset: %s, N=%d\n" size(X) N
    @printf "data variance: %.6f  (mean-predictor MSE baseline)\n" var(X)

    m = make_vae(rng) |> dev
    opt = Optimisers.setup(Optimisers.AdamW(LR), m)

    for epoch in 1:EPOCHS
        t0 = time()
        tot = mse_tot = kl_tot = 0f0
        nb = 0
        perm = randperm(rng, N)
        for i in 1:BATCH:N-BATCH+1
            x = X[:, :, :, perm[i:i+BATCH-1]] |> dev
            l, mse_l, kl_l = vae_loss(m, x, rng)
            gs = Zygote.gradient(m) do mm
                first(vae_loss(mm, x, rng))    # descend total loss only
            end |> first                        # model grad as NamedTuple tree
            opt, m = Optimisers.update(opt, m, gs)  # non-mutating: (state, new_model)
            tot += l; mse_tot += mse_l; kl_tot += kl_l; nb += 1
        end
        @printf "[flux vae] Epoch %2d, Loss: %.6f (mse %.6f, kl %.6f), Time: %.2fs\n" epoch tot/nb mse_tot/nb kl_tot/nb time()-t0
    end

    # --- quick check: variance explained by the reconstructions ---
    x = X[:, :, :, 1:min(256, N)] |> dev
    μ, σ, x̂ = forward(m, x, rng)
    resid = mean((cpu(x̂) .- x) .^ 2)
    @printf "recon MSE on first %d samples: %.6f ; var(X) = %.6f ; variance explained = %.1f%%\n" min(256, N) resid var(X) 100f0*(1 - resid/var(X))

end

main()
