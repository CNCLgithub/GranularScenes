# Stage 2: train the occupancy decoder on top of the frozen Flux VAE encoder.
# Companion to scripts/nn/vae_flux.jl. The VAE (encoder + mu + scale) is loaded
# from the stage-1 checkpoint and frozen; only the occupancy decoder is trained.
# Occupancy decoder mirrors src/agent/perception/ddp.jl `occupancy_decoder`:
#   flat latent -> Dense -> reshape (32,32,1) -> conv downsample -> (16,16) sigmoid.
# Loss: BCE. Features mirror vae_flux.jl: env config, per-epoch PNG, JLD2 weights.
#
# Usage:  julia --project=. scripts/nn/occ_flux.jl
# Env:    DDP_DATA, DDP_EPOCHS, DDP_BATCH, DDP_LR, DDP_VAE_CKPT

using HDF5, Flux, Optimisers, Zygote, Random, Statistics, Printf, cuDNN, FileIO, ImageCore, JLD2
import CUDA

const DATA     = get(ENV, "DDP_DATA",    "/spaths/datasets/ddp_train_11f_32x32.hdf5")
const EPOCHS   = parse(Int,   get(ENV, "DDP_EPOCHS",  "30"))
const BATCH    = parse(Int,   get(ENV, "DDP_BATCH",   "64"))
const LR       = parse(Float32, get(ENV, "DDP_LR",     "1e-3"))
const VAE_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxvae_e30.weights.jld2")
const HIDDEN   = 16
const LATENT   = 32
const FEAT     = HIDDEN ÷ 4                  # channels at the 32x32 bottleneck

# ------------------------------------------------------- frozen VAE (stage 1) ----
function make_vae(rng=Random.default_rng())
    encoder = Chain(
        Conv((5, 5), 1 => HIDDEN;      stride=2, pad=2), swish,
        Conv((5, 5), HIDDEN => 8;      stride=2, pad=2), swish,
        Conv((5, 5), 8 => FEAT;        stride=2, pad=2), swish,
        Flux.flatten,                             # (FEAT*32*32, B)
    )
    mu_proj    = Dense(FEAT * 32 * 32 => LATENT)
    scale_proj = Dense(FEAT * 32 * 32 => LATENT)
    return (encoder=encoder, mu=mu_proj, scale=scale_proj)
end

# ------------------------------------------------------- occupancy decoder ----
# flat latent -> (32,32,1) grid -> stride-2 conv downsample -> (16,16) occupancy.
# Mirrors occupancy_decoder() in src/agent/perception/ddp.jl (max_num_filters=64).
function make_occ(rng=Random.default_rng())
    head = Dense(LATENT => 32 * 32)
    tail = Chain(
        Conv((3, 3), 1 => 16; stride=1, pad=1), leakyrelu,
        Conv((3, 3), 16 => 16; stride=2, pad=1), leakyrelu,
        Conv((3, 3), 16 => 1;  stride=1, pad=1),
        sigmoid,
    )
    return (head=head, tail=tail)
end

function encode(m, x)
    h = m.encoder(x)
    μ = m.mu(h)
    σ = softplus.(m.scale(h)) .+ 1f-8
    return μ, σ
end

occ_forward(occ, z) = occ.tail(reshape(occ.head(z), 32, 32, 1, size(z, 2)))

# ---------------------------------------------------------------- loss -----
function occ_loss(occ, z, O)
    p̂ = occ_forward(occ, z)
    mse = mean((p̂ .- O) .^ 2)
    acc = mean((p̂ .> 0.5f0) .== (O .> 0.5f0))
    return mse, acc
end

# ---------------------------------------------------------------- data -----
function load_data(path)
    h5open(path, "r") do f
        X = Float32.(read(f["depth"]))     # (256, 256, 1, N)
        O = Float32.(read(f["occ"]))       # (16, 16, 1, N)
        X_min, X_max = extrema(X)
        X .-= X_min
        X .*= 1.0f0 / X_max
        @assert ndims(X) == 4 && size(X, 3) == 1
        @assert size(O) == (16, 16, 1, size(X, 4))
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
    X, O = load_data(DATA)
    N = size(X, 4)
    @printf "dataset: %s, occ %s, N=%d\n" size(X) size(O) N

    # --- frozen VAE from stage 1 ---
    @printf "Loading VAE weights from %s ...\n" VAE_CKPT
    vae = make_vae(rng) |> dev
    Flux.loadmodel!(vae, JLD2.load(VAE_CKPT, "ps") |> dev; filter = k -> k in (:encoder, :mu, :scale))

    # --- trainable occupancy decoder ---
    occ = make_occ(rng) |> dev
    opt = Optimisers.setup(Optimisers.AdamW(LR), occ)

    ckpt_dir = "/spaths/checkpoints"
    mkpath(ckpt_dir)
    tag = "fluxocc_e$(EPOCHS)"
    viz_i = 1:min(8, N)
    viz_x = X[:, :, :, viz_i] |> dev
    viz_O = O[:, :, :, viz_i] |> dev

    for epoch in 1:EPOCHS
        t0 = time()
        tot = acc_tot = 0f0
        nb = 0
        perm = randperm(rng, N)
        for i in 1:BATCH:N-BATCH+1
            idx = perm[i:i+BATCH-1]
            x = X[:, :, :, idx] |> dev
            O_b = O[:, :, :, idx] |> dev
            # frozen encoder: z sampled once per batch, no grad through the VAE
            μ, σ = encode(vae, x)
            z = μ   # deterministic path, mirrors pydeps/og_proposal tasks.py forward()

            l, acc = occ_loss(occ, z, O_b)
            gs = Zygote.gradient(occ) do oo
                first(occ_loss(oo, z, O_b))
            end |> first
            opt, occ = Optimisers.update(opt, occ, gs)
            tot += l; acc_tot += acc; nb += 1
        end
        @printf "[flux occ] Epoch %2d, Loss: %.6f (acc %.4f), Time: %.2fs\n" epoch tot/nb acc_tot/nb time()-t0

        # per-epoch visualization: occupancy target | 4px gap | prediction
        μ_viz, _ = encode(vae, viz_x)
        p̂ = occ_forward(occ, μ_viz)   # deterministic: viz with μ (no sampling noise)
        rows = map(1:length(viz_i)) do i
            gt = cpu(viz_O[:, :, 1, i]); pr = cpu(p̂[:, :, 1, i])
            clamp.(hcat(gt, fill(0.5f0, size(gt, 1), 4), pr), 0f0, 1f0)
        end
        img = hcat(rows...)'
        save(joinpath(ckpt_dir, "$tag.e$(epoch).occ.png"), Gray.(clamp.(img, 0f0, 1f0)))
    end

    # --- save weights + final visualization ---
    jldsave(joinpath(ckpt_dir, "$tag.weights.jld2"); ps=Flux.state(occ))
    @printf "saved %s\n" joinpath(ckpt_dir, "$tag.weights.jld2")

    μ_viz, _ = encode(vae, viz_x)
    p̂ = occ_forward(occ, μ_viz)
    rows = map(1:length(viz_i)) do i
        gt = cpu(viz_O[:, :, 1, i]); pr = cpu(p̂[:, :, 1, i])
        clamp.(hcat(gt, fill(0.5f0, size(gt, 1), 4), pr), 0f0, 1f0)
    end
    img = hcat(rows...)'
    save(joinpath(ckpt_dir, "$tag.occ.png"), Gray.(clamp.(img, 0f0, 1f0)))

    # --- final metric on a held-out slice ---
    idx = max(1, N - 255):N
    μ, σ = encode(vae, X[:, :, :, idx] |> dev)
    z = μ   # deterministic path
    l, acc = occ_loss(occ, z, O[:, :, :, idx] |> dev)
    # BCE of the all-0.5 predictor for scale; occupancy is often sparse
    O_slice = O[:, :, :, idx] |> dev
    base = mean((O_slice .- mean(O_slice)) .^ 2)
    @printf "final MSE %.6f (variance baseline %.6f), acc %.4f on %d samples\n" l base acc length(idx)
end

main()
