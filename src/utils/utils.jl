using JSON
using FileIO: save
using Colors
using ImageIO
using ImageCore: colorview
using ImageInTerminal
using LinearAlgebra: lmul!

export save_img_array,
    softmax,
    softmax!,
    read_json


# index arrays using sets. Order doesn't matter
# function Base.to_index(i::Set{T}) where {T}
#     Base.to_index(collect(T, i))
# end


#################################################################################
# Visuals
#################################################################################

function draw_mat(m::Matrix,
                  rotate::Bool = true,
                  c1=colorant"black",
                  c2=colorant"white")
    m = clamp.(m, 0.0, 1.0)
    img = weighted_color_mean.(m, c2, c1)
    img = rotate ? rotr90(img, 3) : img
    return img
end

function display_mat(m::Matrix;
                     rotate::Bool = true,
                     c1=colorant"black",
                     c2=colorant"white")
    display(draw_mat(m, rotate, c1, c2))
    return nothing
end

function display_img(m::Array{<:T, 3}) where {T<:Real}
    if size(m, 1) > 3
        m = permutedims(m, (3, 2, 1))
    end
    img = colorview(RGB, m)
    display(img)
    return nothing
end


#################################################################################
# IO
#################################################################################

function process_taichi_array(array::PyObject)
    arr = @pycall array.to_numpy()::Array{Float32, 3}
    reverse!(arr, dims = 1)
    reverse!(arr, dims = 2)
    return arr
end

function save_img_array(array::PyObject, path::String)
    save_img_array(process_taichi_array(array), path)
end

function save_img_array(array::Array{T}, path::String) where {T}
    x = permutedims(array, (3,2,1))
    clamp!(x, zero(T), one(T))
    img = colorview(RGB, x)
    save(path, img)
end


function _load_device()
    if torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else
        device = torch.device("cpu")
    end
    return device
end

"""
    read_json(path)
    opens the file at path, parses as JSON and returns a dictionary
"""
function read_json(path)
    local data
    open(path, "r") do f
        data = JSON.parse(f)
    end

    # converting strings to symbols
    sym_data = Dict()
    for (k, v) in data
        sym_data[Symbol(k)] = v
    end

    return sym_data
end

function _init_graphics()
    variants = @pycall mi.variants()::PyObject
    if "cuda_ad_rgb" in variants
        @pycall mi.set_variant("cuda_ad_rgb")::PyObject
    else
        @pycall mi.set_variant("scalar_rgb")::PyObject
    end
    return nothing
end

#################################################################################
# Math
#################################################################################

# function softmax(x; t::Float64 = 1.0)
#     x = x .- maximum(x)
#     exs = exp.(x ./ t)
#     sxs = sum(exs)
#     n = length(x)
#     isnan(sxs) || iszero(sxs) ? fill(1.0/n, n) : exs ./ sxs
# end

abstract type MarginalType end

"""Discrete values of n categories of type T"""
struct DiscreteMarginal{T} <: MarginalType
end

function check_format(m::DiscreteMarginal{T}, bfr, key) where {T}
    @assert length(bfr) > 0 "Cannot marginalize over no samples"
    sample = bfr[1][key]
    st = typeof(sample)
    @assert st <: T "Marginal type missmatch: $(st) ≠ $(T)"
    return nothing
end

"""Continuous values of n dimensions of type T"""
@with_kw struct ContinuousMarginal{T} <: MarginalType
    op::Function = (+)
    norm::Function = (x, n::Integer) -> x / n
    id::Function = (A::Type) -> zero(A)
end

function ContinuousMarginal(::Type{T}) where {T<:AbstractArray}
    ContinuousMarginal{T}(; id = (A::Type{T}) -> zero(eltype(A)))
end

function check_format(m::ContinuousMarginal{T}, bfr, key) where {T}
    n = length(bfr)
    @assert n > 0 "Cannot marginalize over no samples"
    sample = bfr[1][key]
    @assert typeof(sample) <: T "Marginal type missmatch"
    return nothing
end

function marginalize(m::DiscreteMarginal{T}, bfr, key::Symbol) where {T<:Bool}
    check_format(m, bfr, key)
    n = length(bfr)
    acc = 0.0
    for sample in bfr
        if sample[key]
            acc += 1
        end
    end
    acc / n
end

function marginalize(m::DiscreteMarginal{T}, bfr, key::Symbol) where {T}
    check_format(m, bfr, key)
    n = length(bfr)
    acc = Dict{T, Int}()
    for sample in bfr
        v = sample[key]
        acc[v] = get(acc, v, 0) + 1
    end
    map!(x -> x / n, values(acc))
    return acc
end

function marginalize(m::ContinuousMarginal{T}, bfr, key::Symbol) where {T<:AbstractArray}
    check_format(m, bfr, key)
    n = length(bfr)
    #REVIEW: eltype instead of f64?
    marginal = similar(bfr[1][key], Float64)
    fill!(marginal, m.id(T))
    for i = 1:n
        datum = bfr[i][key]
        for j = eachindex(marginal)
            marginal[j] = m.op(marginal[j], datum[j])
        end
    end
    @inbounds for j = eachindex(marginal)
        marginal[j] = m.norm(marginal[j], n)
    end
    marginal
end


function marginalize(m::ContinuousMarginal{T}, bfr, key::Symbol) where {T<:Real}
    check_format(m, bfr, key)
    n = length(bfr)
    marginal = m.id(T)
    for i = 1:n
        marginal = m.op(marginal, bfr[i][key])
    end
    m.norm(marginal, n)
end

function marginalize(bfr, key::Symbol)
    n = length(bfr)
    @assert n > 0
    T = typeof(bfr[1][key])
    marginalize(ContinuousMarginal(T), bfr, key)
end

function softmax(x::Array{<:Real}, t::Real = 1.0)
    out = similar(x)
    softmax!(out, x, t)
    return out
end

function softmax!(out::Array{<:Real}, x::Array{<:Real}, t::Real = 1.0)
    nx = length(x)
    maxx = maximum(x)
    sxs = 0.0

    if maxx == -Inf
        out .= 1.0 / nx
        return nothing
    end

    @inbounds for i = 1:nx
        out[i] = @fastmath exp((x[i] - maxx) / t)
        sxs += out[i]
    end
    rmul!(out, 1.0 / sxs)
    return nothing
end

function uniform_weights(x)::Vector{Float64}
    n = length(x)
    fill(1.0 / n, n)
end
# TODO: documentation
# deals with empty case
function safe_uniform_weights(x)::Vector{Float64}
    n = length(x) + 1
    ws = fill(1.0 / n, n)
    return ws
end

function fast_sigmoid(x::Float64)
    2 * x / (1.0 + abs(x))
end

const NEGLN2 = -log(2)

"""
Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
See [Maechler2012accurate] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function log1mexp(x::Float64)
    x  > NEGLN2 ? log(-expm1(x)) : log1p(-exp(x))
end

#################################################################################
# Room coordinate manipulation
#################################################################################


function add(r::GridRoom, f)
    g = Rooms.PathGraph(grid(Rooms.steps(r)))
    d = deepcopy(r.data)
    for idx = f
        d[idx] = obstacle_tile
    end
    Rooms.prune_edges!(g, d)
    GridRoom(r.steps, r.bounds, r.entrance,
             r.exits, g, d)
end


const unit_ci = CartesianIndex(1,1)

"""
Is coordinate `a` adjacent to `b`?
"""
function is_next_to(a::CartesianIndex{2}, b::CartesianIndex{2})
    d = abs.(Tuple(a - b))
    # is either left,right,above,below
    d == (1, 0) || d == (0, 1)
end

"""
Takes a tile in an `mxn` space and \"expands\" by `factor`
"""
function up_scale_inds(src::CartesianIndices{2}, dest::CartesianIndices{2},
                       factor::Int64, vs::Vector{Int64})
    result = Array{Int64, 3}(undef, factor, factor, length(vs))
    for i = 1:length(vs)
        result[:, :, i] = up_scale_inds(src, dest, factor, vs[i])
    end
    vec(result)
end
function up_scale_inds(src::CartesianIndices{2}, dest::CartesianIndices{2},
                       factor::Int64, v::Int64)
    kern = CartesianIndices((1:factor, 1:factor))
    offset = CartesianIndex(1,1)
    dest_l = LinearIndices(dest)
    dest_l[(src[v] - offset) * factor .+ kern]
end


function refine_space(state, params)
    lower_dims = size(state)
    upper_dims = level_dims(params, next_lvl)
    kernel_dims = Int64.(upper_dims ./ lower_dims)

    lower_ref = CartesianIndices(lower_dims) # dimensions of coarse state
    kernel_ref = CartesianIndices(kernel_dims) # dimensions of coarse state

    upper_lref = LinearIndices(upper_dims)
    lower_lref = LinearIndices(lower_dims)
    kernel_lref = LinearIndices(kernel_dims)

    kp = prod(kernel_dims) # number of elements in kernel

    # iterate over coarse state kernel
    next_state = zeros(T, upper_dims)
    for lower in lower_ref
        # map together kernel steps
        i = lower_lref[lower]
        c = CartesianIndex((Tuple(lower) .- (1, 1)) .* kernel_dims)
        # @show c
        # iterate over scalars for each kernel sweep
        _sum = 0.
        for inner in kernel_ref
            # @show inner
            j = kernel_lref[inner] # index of inner
            idx = c + inner # cart index in refined space
            if j < kp # still retreiving from prop
                val = @read(u[:outer => i => :inner => j => :x],
                            :continuous)
                _sum += val
            else # solving for the final value
                val = state[lower] * kp - _sum
            end
            next_state[idx] = val
        end
    end
end
