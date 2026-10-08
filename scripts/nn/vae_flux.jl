# Stage 1: train the depth VAE (Flux). Model definitions and losses come from
# model/loss definitions in src/agent/perception/ddp.jl (via `using GranularScenes`).
# This script holds only data loading and the training loop.
#
# Usage:  julia --project=. scripts/nn/vae_flux.jl
# Env:    DDP_DATA, DDP_EPOCHS, DDP_BATCH, DDP_BETA, DDP_LR
#
# Dependencies (Project.toml): Flux Optimisers Zygote Functors CUDA cuDNN
#                              HDF5 FileIO ImageCore JLD2

using HDF5, Flux, Optimisers, Zygote, Random, Statistics, Printf, cuDNN, FileIO, ImageCore, JLD2
using Flux: state
import CUDA

using GranularScenes: DepthVAE, OccDecoder, encode, sample, decode_occ,
                      vae_loss, occ_loss

const DATA    = get(ENV, "DDP_DATA",   "/spaths/datasets/ddp_train_11f_32x32.hdf5")
const EPOCHS  = parse(Int,   get(ENV, "DDP_EPOCHS", "30"))
const BATCH   = parse(Int,   get(ENV, "DDP_BATCH",  "64"))
const BETA    = parse(Float32, get(ENV, "DDP_BETA", "0.01"))
const LR      = parse(Float32, get(ENV, "DDP_LR",   "1e-3"))

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

    m = DepthVAE(rng) |> dev
    opt = Optimisers.setup(Optimisers.AdamW(LR), m)

    ckpt_dir = "/spaths/checkpoints"
    mkpath(ckpt_dir)
    tag = "fluxvae_e$(EPOCHS)"
    viz_x = X[:, :, :, 1:min(8, N)] |> dev   # fixed viz samples, same every epoch

    for epoch in 1:EPOCHS
        t0 = time()
        tot = mse_tot = kl_tot = 0f0
        nb = 0
        perm = randperm(rng, N)
        for i in 1:BATCH:N-BATCH+1
            x = X[:, :, :, perm[i:i+BATCH-1]] |> dev
            l, mse_l, kl_l = vae_loss(m, x; β=BETA)
            gs = Zygote.gradient(m) do mm
                first(vae_loss(mm, x; β=BETA))  # descend total loss only
            end |> first                        # model grad as NamedTuple tree
            opt, m = Optimisers.update(opt, m, gs)  # non-mutating: (state, new_model)
            tot += l; mse_tot += mse_l; kl_tot += kl_l; nb += 1
        end
        @printf "[flux vae] Epoch %2d, Loss: %.6f (mse %.6f, kl %.6f), Time: %.2fs\n" epoch tot/nb mse_tot/nb kl_tot/nb time()-t0

        # per-epoch visualization: same fixed samples each epoch, to see evolution
        _, _, x̂e = sample(m, viz_x)
        rows = map(1:size(viz_x, 4)) do i
            gt = cpu(viz_x[:, :, 1, i]); rec = cpu(x̂e[:, :, 1, i])
            clamp.(hcat(gt, fill(1f0, size(gt, 1), 4), rec), 0f0, 1f0)
        end
        img = hcat(rows...)'
        save(joinpath(ckpt_dir, "$tag.e$(epoch).recon.png"), Gray.(clamp.(img, 0f0, 1f0)))
    end

    # --- save weights + final reconstruction --------------------------

    # save raw parameters + optimizer state; reload via Flux.loadmodel!(model, ps)
    # (avoids JLD2's by-name function storage — no closures serialized)


    using Flux: state

    jldsave(joinpath(ckpt_dir, "$tag.weights.jld2"); ps=Flux.state(m), opt_state=opt)
    @printf "saved %s\n" joinpath(ckpt_dir, "$tag.weights.jld2")

    # side-by-side visualization: row i = (ground truth, reconstruction)
    xs = viz_x
    _, _, x̂s = sample(m, xs)
    rows = map(1:size(xs, 4)) do i
        gt  = cpu(xs[:, :, 1, i])
        rec = cpu(x̂s[:, :, 1, i])
        row = clamp.(hcat(gt, fill(1f0, size(gt, 1), 4), rec), 0f0, 1f0)  # gt | 4px gap | recon
        row
    end
    n = length(rows)
    img = hcat(rows...)   # each row-entry is 256x512 (gt|4px gap|rec); hcat them side by side
    img = img'             # transpose to (W, H) orientation for saving
    save(joinpath(ckpt_dir, "$tag.recon.png"), Gray.(clamp.(img, 0f0, 1f0)))
    @printf "saved %s and %s\n" joinpath(ckpt_dir, "$tag.weights.jld2") joinpath(ckpt_dir, "$tag.recon.png")

    # --- quick check: variance explained by the reconstructions ---
    x = X[:, :, :, 1:min(256, N)] |> dev
    μ, σ, x̂ = sample(m, x)
    resid = mean((cpu(x̂) .- cpu(x)) .^ 2)   # both operands host-side
    @printf "recon MSE on first %d samples: %.6f ; var(X) = %.6f ; variance explained = %.1f%%\n" min(256, N) resid var(X) 100f0*(1 - resid/var(X))

end

main()
