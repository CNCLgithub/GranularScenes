using LinearAlgebra: rmul!
using DataStructures: PriorityQueue
using Base.Order: Ordering, ReverseOrdering, Reverse

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
    @unpack steps, accepts = aux
    # println("RW acceptance ratio for node $(node): $(accepts  / steps)")
    aux.steps = 0
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
    objective::Function = qt_path_cost
    # destance metrics for two task objectives
    distance::Function = (x, y) -> norm(x - y)

    # smoothing relative sensitivity for each tracker
    smoothness::Float64 = 1.0
end


mutable struct AdaptiveAux <: AuxillaryState
    delta_pi::Float64
    delta_s::Float64
    objective::Any
    accepts::Int64
    steps::Int64
    sensitivities::Matrix{Float64} # TODO: remove, not used
    queue::PriorityQueue{Int64, Float64, ReverseOrdering}
end

function AuxState(p::AdaptiveComputation, trace)
    dims = first(get_args(trace)).dims
    ndims = prod(dims)
    sensitivities = zeros(dims)
    queue = init_queue(trace)
    AdaptiveAux(0., 0., 0., 0, 0,
                sensitivities,
                queue)
end

accepts(aux::AdaptiveAux) = aux.accepts

function select_node(p::AdaptiveComputation, aux::AdaptiveAux)

    ks = collect(keys(aux.queue))
    ws = collect(values(aux.queue))

    clamp!(ws, -100, Inf)
    ws = softmax(ws, 1.0)
    stop = 0.0
    node = 0
    gr = 0.0
    nidx = categorical(ws)
    node = ks[nidx]
    gr = aux.queue[node]
    # if bernoulli(0.25) # TODO: make hyper-parameter
    #     node = rand(keys(aux.queue))
    #     gr = aux.queue[node]
    # end
    # println("Adaptive Protocol: node $(node), relevance $(gr)")
    # i = 1
    # for kv = aux.queue
    #     i > 3 && break
    #     println("\t $(kv) | $(ws[i])")
    #     i += 1
    # end

    node
end

function rw_block_init!(aux::AdaptiveAux, p::AdaptiveComputation,
                        trace)
    aux.accepts = 0
    aux.delta_pi = 0.0
    aux.delta_s = 0.0
    aux.objective = p.objective(trace)
    return nothing
end

function rw_block_inc!(aux::AdaptiveAux, p::AdaptiveComputation,
                       t, node, alpha)
    obj_t_prime = p.objective(t)
    aux.delta_pi += p.distance(aux.objective, obj_t_prime)
    # ds = alpha < -20 ? 0.0 : exp(alpha)
    aux.delta_s = logsumexp(aux.delta_s, alpha)
    # aux.delta_s += exp(min(0.0, alpha))
    # aux.delta_s += -0.01
    aux.steps += 1
    return nothing
end


function rw_block_accept!(aux::AdaptiveAux,
                          p::AdaptiveComputation,
                          t::Gen.Trace, node)
    aux.accepts += 1
    # aux.delta_s += 0.01
    aux.objective = p.objective(t)
    return nothing
end

function rw_block_extend!()
    # accept_ratio = accept_ct / rw_cycles
    # addition_rw_cycles = (delta_pi > 0) * floor(Int64, sm_cycles * accept_ratio)
end

function rw_block_complete!(aux::AdaptiveAux,
                            p::AdaptiveComputation,
                            t, node)
    # compute goal-relevance
    @unpack delta_pi, delta_s, steps, accepts = aux
    delta_pi = log(delta_pi) - log(steps)
    delta_s = delta_s - log(steps)
    goal_relevance = delta_pi + delta_s
    # goal_relevance = log(delta_pi) + log(delta_s) - 2 * log(steps)

    # update aux state
    # qt = get_retval(t)
    # prod_node = traverse_qt(qt, node).node
    # sidx = node_to_idx(prod_node, max_leaves(qt))
    # for i = sidx
    #     aux.sensitivities[i] = goal_relevance
    # end
    aux.queue[node] = goal_relevance
    accept_ratio = accepts / steps
    # println("\t $(delta_pi) + $(delta_s) | AR=$(accept_ratio)")
    return nothing
end


function sm_block_init!(aux::AdaptiveAux, p::AdaptiveComputation)
    aux.accepts = 0
    return nothing
end

function sm_block_accept!(aux::AdaptiveAux, node, move)
    aux.accepts += 1
    update_queue!(aux.queue, node, move)
end

function sm_block_complete!(aux::AdaptiveAux, p::AdaptiveComputation,
                            node, move)
    # println("Accepted move $(move) on node $(node)")
    return nothing
end

function attention_map(aux::AdaptiveAux, t::Gen.Trace)
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

function init_queue(tr::Gen.Trace)
    qt = get_retval(tr)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    # go through the current set of terminal nodes
    # and intialize priority
    for n = qt.leaves
        q[n.node.tree_idx] = log(area(n.node))
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
    queue[parent] = prev_val # * 0.25
    return nothing
end
