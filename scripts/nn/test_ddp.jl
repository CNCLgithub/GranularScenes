
using HDF5, Flux, Optimisers, Zygote, Random, Statistics, Printf, cuDNN, FileIO, ImageCore, JLD2
using Flux: state
import CUDA

using GranularScenes: DataDrivenState, qt_ddp

const DATA     = get(ENV, "DDP_DATA",    "/spaths/datasets/ddp_train_11f_32x32.hdf5")
const VAE_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxvae_e30.weights.jld2")
const OCC_CKPT = get(ENV, "DDP_VAE_CKPT", "/spaths/checkpoints/fluxocc_e30.weights.jld2")

const DATA_DEPTH_MIN = 3.971077f0
const DATA_DEPTH_MAX = 10.181246f0

# ---------------------------------------------------------------- data -----
function load_data(path)
    dataset = DDPSDataset(path)
    x, o = dataset[1]
    x .-= DATA_DEPTH_MIN
    x .*= 1.0f0 / DATA_DEPTH_MAX
    (x, o)
end


function main()
    rng = Xoshiro(0)
    use_gpu = CUDA.functional()
    dev = use_gpu ? (x -> fmap(CUDA.cu, x)) : Flux.cpu
    use_gpu && @info "GPU: " * CUDA.name(CUDA.device())

    @printf "Loading %s ...\n" DATA
    X, O = load_data(DATA)
    N = size(X, 4)

    dds = DataDrivenState(
        ;
        device = dev,
        vae_path = VAE_CKPT,
        occ_path = OCC_CKPT,
        var = 0.001
    )

    @info "Loaded frozen VAE from " VAE_CKPT
    @info "Loaded frozen OCC from " OCC_CKPT

    viz_x = X |> dev
    viz_O = O |> dev

    qt = qt_ddp(dds, viz_x)
    # GranularScenes.display_mat(X[:, :, 1, 1])
    display(qt_topdown(qt))
    @show GranularScenes.nleaves(qt)
end

main()
