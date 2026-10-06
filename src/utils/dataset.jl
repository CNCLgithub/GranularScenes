# HDF5 dataset stub for the depth-map / occupancy VAE.
#
# Layout (single file):
#   /depth : Float16, shape (256, 256, 1, N), chunked per-sample (1,1,1,256),
#            gzip-compressed
#   /occ   : UInt8,   shape (16, 16, 1, N),   chunked per-sample (1,1,1,16),
#            gzip-compressed
#
# Trials are generated one at a time and appended row-wise. The size N must be
# known a priori, so the file is preallocated with an extendable dataset of the
# final size; the writer just fills slots as trials complete.
#
# Writer usage:
#     ds = DDPSDatasetWriter("data.h5"; N=100_000)
#     for trial in sampler
#         write_trial!(ds, trial.depth, trial.occ)
#     end
#     close(ds)
#
# Reader usage (wraps into an indexable dataset for MLUtils.DataLoader):
#     ds = DDPSDataset("data.h5")
#     loader = DataLoader(ds; batchsize=32, shuffle=true)
export DDPSDataset, DDPSDatasetWriter, write_trial!

import HDF5
using HDF5: h5open, attrs, create_dataset 

const DEPTH_SIZE = (256, 256, 1)   # (H, W, C)
const OCC_SIZE = (16, 16, 1)       # (H, W, C)

# ---------------------------------------------------------------------------
# Writer: preallocate and fill trial-by-trial
# ---------------------------------------------------------------------------
"""
    DDPSDatasetWriter(path; N)

Preallocates an HDF5 file with extendable datasets of final length `N`.
Call `write_trial!(w, depth, occ)` once per completed trial; indices are
assigned sequentially. `close(w)` flushes and closes.
"""
mutable struct DDPSDatasetWriter
    file::HDF5.File
    depth::HDF5.Dataset
    occ::HDF5.Dataset
    count::Int
    max::Int
end

function DDPSDatasetWriter(path::AbstractString, N::Integer)
    f = h5open(path, "w")
    # Preallocate the full extent; chunk along the sample (4th) axis.
    depth = create_dataset(f, "depth", Float32, (DEPTH_SIZE..., N);
        chunk=(DEPTH_SIZE..., 1),
        shuffle=(), deflate=4)
    occ = create_dataset(f, "occ", UInt8, (OCC_SIZE..., N);
        chunk=(OCC_SIZE..., 1),
        shuffle=(), deflate=4)
    attrs(depth)["description"] = "camera-centric depth maps, [0,1] in Float16"
    attrs(occ)["description"] = "top-down occupancy grids, {0,1} in UInt8"
    attrs(occ)["N"] = N
    attrs(depth)["N"] = N
    return DDPSDatasetWriter(f, depth, occ, 0, N)
end

"""
    write_trial!(w, depth, occ) -> Int

Append one trial. `depth` must be Float32/Float16 array of size (256, 256, 1)
and `occ` a binary array of size (16, 16, 1). Returns the 1-based index
assigned to the trial.
"""
function write_trial!(w::DDPSDatasetWriter, depth, occ)
    i = w.count + 1
    N = w.max
    i > N && throw(ArgumentError("dataset full (N=$N); cannot write trial $i"))
    w.depth[:, :, :, i] = Float32.(depth)   
    w.occ[:, :, :, i] = UInt8.(clamp.(occ, 0, 1))
    w.count = i
    return i
end

Base.length(w::DDPSDatasetWriter) = w.count
Base.close(w::DDPSDatasetWriter) = close(w.file)

# ---------------------------------------------------------------------------
# Reader: indexable wrapper for MLUtils.DataLoader
# ---------------------------------------------------------------------------
"""
    DDPSDataset(path)

Open the HDF5 file read-only. The struct implements `length` and `getindex`
returning `(depth, occ)` pairs shaped (H, W, C, 1), ready for
`DataLoader(ds; batchsize, shuffle=true)`.
"""
struct DDPSDataset
    file::HDF5.File
    depth::HDF5.Dataset
    occ::HDF5.Dataset
end

function DDPSDataset(path::AbstractString)
    f = h5open(path, "r")
    return DDPSDataset(f, f["depth"], f["occ"])
end

Base.length(ds::DDPSDataset) = attrs(ds.depth)["N"]

function Base.getindex(ds::DDPSDataset, i::Integer)
    d = Float32.(ds.depth[:, :, :, i])   # (256, 256, 1)
    o = Float32.(ds.occ[:, :, :, i])     # (16, 16, 1)
    return (reshape(d, DEPTH_SIZE..., 1), reshape(o, OCC_SIZE..., 1))
end

# Batched access (used by DataLoader when it requests a vector of indices).
function Base.getindex(ds::DDPSDataset, idxs::AbstractVector{<:Integer})
    # d = Float32.(ds.depth[:, :, :, idxs])
    # o = Float32.(ds.occ[:, :, :, idxs])
    d = Float32.(read(ds.depth, :, :, :, idxs)
    o = Float32.(read(ds.occ, :, :, :, idxs)
    return (d, o)
end

Base.close(ds::DDPSDataset) = close(ds.file)

function Base.iterate(ds::DDPSDataset, i::Int = 1)
    i > length(ds) && return nothing
    (ds[i], i + 1)
end
