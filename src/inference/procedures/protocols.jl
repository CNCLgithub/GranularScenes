using LinearAlgebra: rmul!
using DataStructures: PriorityQueue
using Base.Order: Ordering, ReverseOrdering, Reverse

using Printf: @printf

export AttentionProtocal,
    UniformProtocol,
    AdaptiveComputation

abstract type AttentionProtocol end

#################################################################################
# No attention
#################################################################################

struct UniformProtocol <: AttentionProtocol end

mutable struct UniformAux <: AuxillaryState
    accepts::Int64
    steps::Int64
    qt_idxs::PriorityQueue{Int64, <:Number, <:Ordering}
end

accepts(aux::UniformAux) = aux.accepts

function AuxState(p::UniformProtocol, trace)
    qt = get_retval(trace)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    for (i, n) = enumerate(qt.leaves)
        q[n.node.tree_idx] = -Inf
    end
    UniformAux(0, 0, q)
end

function select_node(p::UniformProtocol, aux::UniformAux)
    rand(keys(aux.qt_idxs))
end


function rw_block_init!(aux::UniformAux, p::UniformProtocol, trace)
    aux.accepts = 0
    aux.steps = 0
    return nothing
end

function rw_block_inc!(aux::UniformAux, p::UniformProtocol,
                       t::Gen.Trace, node, alpha)
    aux.qt_idxs[node] = logsumexp(aux.qt_idxs[node], 0.0)
    aux.steps += 1
    return nothing
end

function rw_block_accept!(aux::UniformAux, p::UniformProtocol,
                          t::Gen.Trace, node)
    aux.accepts += 1
    return nothing
end

function rw_block_complete!(aux::UniformAux, p::UniformProtocol,
                            t::Gen.Trace, node)
    # @unpack steps, accepts = aux
    # println("RW acceptance ratio for node $(node): $(accepts  / steps)")
    # aux.steps = 0
    return nothing
end

function sm_block_init!(aux::UniformAux, p::UniformProtocol)
    aux.accepts = 0
    return nothing
end

function sm_block_accept!(aux::UniformAux,
                          node, move)
    aux.accepts += 1
    update_queue!(aux.qt_idxs, node, move)
    return nothing
end

function sm_block_complete!(aux::UniformAux, p::UniformProtocol,
                            node, move)
    # println("Accepted move $(move) on node $(node)")
    return nothing
end


function attention_map(aux::UniformAux, tr::Gen.Trace)
    qt = get_retval(tr)
    n = max_leaves(qt)
    amap = Matrix{Float32}(undef, (n, n))
    for x in qt.leaves
        idxs = node_to_idx(x.node, n)
        v = aux.qt_idxs[x.node.tree_idx]
        for i = idxs
            amap[i] = v
        end
    end
    return amap
end

#################################################################################
# Adaptive Computation
#################################################################################

@with_kw struct AdaptiveComputation <: AttentionProtocol
    # Goal driven belief
    objective::Function = quad_tree_path
    # destance metrics for two task objectives
    distance::Function = (x, y) -> norm(x - y)

    # smoothing relative sensitivity for each tracker
    smoothness::Float64 = 1.0
end


mutable struct AdaptiveAux <: AuxillaryState
    delta_pi::Dict{Int64, Float64}
    delta_s::Dict{Int64, Float64}
    accepts::Int64
    steps::Int64
    temp::Float64
    queue::PriorityQueue{Int64, Float64, ReverseOrdering}
end

function AuxState(p::AdaptiveComputation, trace::QTTrace)
    (_, _, dpi) = p.objective(trace)
    # go through the current set of terminal nodes
    # and intialize priority
    ds = Dict{Int64, Float64}()
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    qt = get_retval(trace)
    lvs = leaves(qt)
    n = length(lvs)
    for i = 1:n
        x = @inbounds node(lvs[i])
        idx = tree_idx(x)
        ds[idx] = 0.0
        dpi[idx] = log(get(dpi, idx, 0.))
        q[idx] = dpi[idx]
    end
    AdaptiveAux(dpi, ds, 0, 0, 1.0, q)
end

accepts(aux::AdaptiveAux) = aux.accepts

function select_node(p::AdaptiveComputation, aux::AdaptiveAux)
    ks = collect(keys(aux.queue))
    raw = collect(values(aux.queue))
    # clamp!(raw, 0.01, Inf)
    ws = softmax(raw, aux.temp) # TODO: add as hyperparameter
    nidx = categorical(ws)
    node = ks[nidx]
    # @printf("Selected node: %d; GR: %3g Pr: %3g\n",
    #         node,
    #         round(aux.queue[node]; sigdigits=3),
    #         round(pselect; sigdigits=3),
    #         )
    return node
end

function rw_block_init!(aux::AdaptiveAux, p::AdaptiveComputation,
                        trace::Gen.Trace)
    aux.accepts = 0
    aux.steps = 0
    return nothing
end

function rw_block_inc!(aux::AdaptiveAux, p::AdaptiveComputation,
                       t, node, alpha)
    aux.steps += 1
    return nothing
end


function rw_block_accept!(aux::AdaptiveAux,
                          p::AdaptiveComputation,
                          t::Gen.Trace, node)
    aux.accepts += 1
    return nothing
end

function rw_block_extend!()
    # accept_ratio = accept_ct / rw_cycles
    # addition_rw_cycles = (delta_pi > 0) * floor(Int64, sm_cycles * accept_ratio)
end

function rw_block_complete!(aux::AdaptiveAux,
                            p::AdaptiveComputation,
                            t::Gen.Trace, n::Int)
    (_, _, grads) = p.objective(t)
    @unpack delta_pi, delta_s, steps, accepts = aux
    accept_ratio = accepts / steps
    delta_s[n] = logsumexp(delta_s[n], log(accept_ratio)) - log(2.0)
    delta_pi[n] = logsumexp(delta_pi[n], log(get(grads, n, 0.0))) - log(2.0)
    aux.queue[n] = delta_pi[n] + delta_s[n]
    # @printf("Node: %d, AR= %3g, GR=%3g, dPi=%3g, dS=%3g \n",
    #         n,
    #         round(accept_ratio; sigdigits=3),
    #         round(aux.queue[n]; sigdigits=3),
    #         round(delta_pi[n]; sigdigits=3),
    #         round(delta_s[n]; sigdigits=3),
    #         )
    return nothing
end


function sm_block_init!(aux::AdaptiveAux, p::AdaptiveComputation)
    aux.accepts = 0
    return nothing
end

function sm_block_accept!(aux::AdaptiveAux, n::Int)
    aux.accepts += 1
    return nothing
end

function sm_block_accept!(aux::AdaptiveAux, node, move)
    aux.accepts += 1
    update_queue!(aux.queue, node, move)
    return nothing
end

function sm_block_complete!(aux::AdaptiveAux, p::AdaptiveComputation,
                            n::Int, move)
    return nothing
end

function sm_block_complete!(aux::AdaptiveAux, p::AdaptiveComputation,
                            tr::Gen.Trace,
                            n::Int)
    (_, _, grads) = p.objective(tr)
    # go through the current set of terminal nodes
    # and intialize priority
    qt = get_retval(tr)
    lvs = leaves(qt)
    n = length(lvs)
    current = collect(keys(qt.mapping))
    to_purge = setdiff(keys(aux.queue), current)
    for idx = to_purge
        delete!(aux.queue, idx)
        delete!(aux.delta_s, idx)
        delete!(aux.delta_pi, idx)
    end
    for idx = current
        aux.delta_pi[idx] = haskey(grads, idx) ?
            log(grads[idx]) :
            get(aux.delta_pi, idx, -Inf)
        aux.delta_s[idx] = get(aux.delta_s, idx, 0.0)
        aux.queue[idx] = logsumexp(
            get(aux.queue, idx, -Inf),
            aux.delta_pi[idx] + aux.delta_s[idx]
        ) - log(2.0)
    end
    return nothing
end

function attention_map(aux::AdaptiveAux, t::Gen.Trace)
    # amap = Matrix{Float32}(aux.gr)
    qt = get_retval(t)
    n = max_leaves(qt)
    amap = Matrix{Float32}(undef, n, n)
    for x in qt.leaves
        idxs = node_to_idx(x.node, n)
        v = aux.queue[x.node.tree_idx]
        for i = idxs
            amap[i] = v
        end
    end
    return amap
end

#################################################################################
# Helpers
#################################################################################

function init_queue(tr::Gen.Trace, gr::Matrix, delta_pi)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    # go through the current set of terminal nodes
    # and intialize priority
    qt = get_retval(tr)
    lvs = leaves(qt)
    n = length(lvs)
    ml = max_leaves(qt)
    for i = 1:n
        x = @inbounds node(lvs[i])
        idx = tree_idx(x)
        dpi = log(get(delta_pi, idx, 0.0))
        q[idx] = dpi
        # sidx = node_to_idx(x, ml)
        # for mi = sidx
        #     gr[mi] = dpi - log(length(sidx))
        # end
    end
    return q
end

function update_queue!(queue, node::Int64, move::Split)
    prev_val = queue[node] + log(0.25)
    # copying parent's (node) value to children
    for i = 1:4
        cid = Gen.get_child(node, i, 4)
        queue[cid] = prev_val
    end
    delete!(queue, node)
    return nothing
end

function update_queue!(queue, node::Int64, move::Merge)
    # merge to parent, averaging siblings relevance
    parent = Gen.get_parent(node, 4)
    prev_val = -Inf
    for i = 1:4
        cid = Gen.get_child(parent, i, 4)
        prev_val = logsumexp(prev_val, queue[cid])
        delete!(queue, cid)
    end
    queue[parent] = prev_val
    return nothing
end

# REVIEW: not used?
function update_queue!(queue, gr::Matrix{Float64},
                       tr::Gen.Trace)
    qt = get_retval(tr)
    ml = max_leaves(qt)
    empty!(queue)
    for l = qt.leaves
        nde = node(l)
        idx = tree_idx(nde)
        mat_idxs = node_to_idx(nde, ml)
        queue[idx] = logsumexp(gr[mat_idxs])
    end
    return nothing
end

function viz_node(tr, node::Int64)
    qt = get_retval(tr)
    n = max_leaves(qt)
    leaves = qt.leaves
    m = Matrix{Bool}(undef, n, n)
    fill!(m, false)
    v = leaf_from_idx(qt, node).node
    for idx = node_to_idx(v, n)
        m[idx] = true
    end
    println("Selected node")
    display_mat(m)
end
