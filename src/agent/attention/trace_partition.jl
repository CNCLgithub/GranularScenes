export TracePartition,
    latent_size,
    get_coord,
    select_prop,
    WMPartition

""" A procedure to selectively index and process traces"""
abstract type TracePartition{T<:Gen.Trace} end

"""
    latent_size(p::TracePartition{T}, tr::T) where {T<:Gen.Trace}

Determine the number of latents exposed for selective processing.
"""
function latent_size end


"""
    get_coord(p::TracePartition{T}, tr::T, i::Int) where {T<:Gen.Trace}

Get the representational coordinate of latent `i` in the trace.
"""
function get_coord end


struct WMPartition{T} <: TracePartition{T} end

function latent_size(::WMPartition{T}, tr::T) where {T<:QTVisionTrace}
    schema, _... = get_args(tr)
    nleaves(schema)
end

function get_coord(p::WMPartition{T}, tr::T, idx::Int
                   ) where {T<:QTVisionTrace}
    qt = get_retval(tr)
    schema = qt.schema
    leaf = schema.leaves[idx]
    center(leaf, qt)
end

function get_coord(p::WMPartition{T}, tr::T, node::NodeId
                   ) where {T<:QTVisionTrace}
    qt = get_retval(tr)
    center(node, qt)
end

function select_prop(::WMPartition, ::QTVisionTrace, ::Int)
    node_ancestral_proposal
end
