# Stage 2: train the occupancy decoder on top of the frozen Flux VAE encoder.
# Companion to scripts/nn/vae_flux.jl. The VAE (DepthVAE from
# src/agent/perception/ddp.jl) is loaded from the stage-1 checkpoint and
# frozen; only the OccDecoder is trained. Loss: MSE on sigmoid outputs,
# deterministic mu path (z = mu) — the recipe that beat unweighted BCE
# (which collapsed to the 11% positive-rate prior) and stochastic z.
#
# Usage:  julia --project=. scripts/nn/occ_flux.jl
# Env:    DDP_DATA, DDP_EPOCHS, DDP_BATCH, DDP_LR, DDP_VAE_CKPT
#
# Dependencies (Project.toml): Flux Optimisers Zygote Functors CUDA cuDNN
#                              HDF5 FileIO ImageCore JLD2

using HDF5, Flux, Optimisers, Zygote, Random, Statistics, Printf, cuDNN, FileIO, ImageCore, JLD2
using Flux: state
import CUDA

using GranularScenes: DepthVAE, OccDecoder, encode, sample, decode_occ,
                      vae_loss, occ_loss

const DATA     = get(ENV, "DDP_DATA",    "/spaths/datasets/ddp_train_11f_32x32.hdf5")
const EPOCHS   = parse(Int,   get(ENV, "DDP_EPOCHS",  "30"))
const BATCH    = parse(Int,   get(ENV, "DDP_BATCH",   "64"))
const LR       = parse(Float32, get(ENV, "DDP_LR",     "1e-3"))
const VAE_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxvae_e30.weights.jld2")

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
    @printf "dataset: %s, N=%d\n" size(X) N
    @printf "occupied fraction: %.4f (predict-prior MSE baseline: %.6f)\n" mean(O .> 0.5f0) mean((mean(O) .- O).^2)

    # Frozen VAE: load stage-1 weights, keep on device, never in the optimizer.
    svae = DepthVAE(rng) |> dev
    Flux.loadmodel!(svae, JLD2.load(VAE_CKPT, "ps") |> dev;
                    filter = k -> k in (:encoder, :mu, :scale))
    @info "Loaded frozen VAE from " VAE_CKPT

    occ = OccDecoder(rng) |> dev
    opt = Optimisers.setup(Optimisers.AdamW(LR), occ)

    ckpt_dir = "/spaths/checkpoints"
    mkpath(ckpt_dir)
    tag = "fluxocc_e$(EPOCHS)"
    viz_i = 1:min(8, N)
    viz_x = X[:, :, :, viz_i] |> dev
    viz_O = O[:, :, :, viz_i] |> dev
    # fixed encoder outputs for visualization (deterministic mu path)
    z_viz = first(encode(svae, viz_x))

    for epoch in 1:EPOCHS
        t0 = time(); tot = acc_tot = 0f0; nb = 0
        perm = randperm(rng, N)
        for i in 1:BATCH:N-BATCH+1
            idx = perm[i:i+BATCH-1]
            x = X[:, :, :, idx] |> dev
            O_b = O[:, :, :, idx] |> dev
            z = first(encode(svae, x))   # frozen encoder, deterministic mu
            (l, acc), gs = Zygote.withgradient(occ) do oo
                (l, acc) = occ_loss(oo, z, O_b)   # loss == mse here
                (l, acc)
            end
            opt, occ = Optimisers.update(opt, occ, gs[1])
            tot += l; acc_tot += acc; nb += 1
        end
        @printf "[flux occ] Epoch %2d, Loss: %.6f (acc %.4f), Time: %.2fs\n" epoch tot/nb acc_tot/nb time()-t0

        # per-epoch visualization: gt | 4px gap | prediction, samples side by side
        p_viz = decode_occ(occ, z_viz)
        rows = map(1:length(viz_i)) do i
            gt = cpu(viz_O[:, :, 1, i]); pr = cpu(p_viz[:, :, 1, i])
            clamp.(hcat(gt, fill(1f0, size(gt, 1), 4), pr), 0f0, 1f0)
        end
        img = hcat(rows...)'
        save(joinpath(ckpt_dir, "$tag.e$(epoch).occ.png"), Gray.(clamp.(img, 0f0, 1f0)))
    end

    # --- save weights + final visualization --------------------------
    jldsave(joinpath(ckpt_dir, "$tag.weights.jld2"); ps=Flux.state(occ), opt_state=opt)
    @printf "saved %s\n" joinpath(ckpt_dir, "$tag.weights.jld2")
end

main()
