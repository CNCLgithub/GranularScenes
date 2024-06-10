using LinearAlgebra: rmul!
using DataStructures: PriorityQueue
using Base.Order: ReverseOrdering, Reverse

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
    qt_idxs::PriorityQueue{Int64, Float64, ReverseOrdering}
end

accepts(aux::UniformAux) = aux.accepts

function AuxState(p::UniformProtocol, trace)
    qt = get_retval(trace)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    # go through the current set of terminal nodes
    # and intialize priority
    for n = qt.leaves
        q[n.node.tree_idx] = area(n.node)
    end
    UniformAux(0, q)
end

function select_node(p::UniformProtocol, aux::UniformAux)
    denom = sum(collect(values(aux.qt_idxs)))
    stop = 0.
    nidxs = 0
    for (nidx, w) = aux.qt_idxs
        stop += w / denom
        if rand() < stop
            node = nidx
            break
        end
    end
    if node === 0
        node, _ = first(aux.qt_idx)
    end
    # node, gr = first(aux.qt_idxs)
    println("Uniform Protocol: node $(node)")
    node
end

function rw_block_init!(aux::UniformAux, trace)
    aux.accepts = 0
    return nothing
end

function rw_block_inc!(aux::UniformAux, t::Gen.Trace, alpha)
    return nothing
end

function rw_block_accept!(aux::UniformAux, t::Gen.Trace)
    aux.accepts += 1
    return nothing
end

function rw_block_complete!(aux::UniformAux, p::UniformProtocol,
                            t::Gen.Trace, node)
    return nothing
end

function sm_block_init!(aux::UniformAux, p::UniformProtocol)
    aux.accepts = 0
    return nothing
end

function sm_block_accept!(aux::UniformAux, ::Gen.Trace,
                          node, move)
    aux.accepts += 1
    update_queue!(aux.qt_idxs, node, move)
    return nothing
end

function sm_block_complete!(aux::UniformAux, p::UniformProtocol,
                            t::Gen.Trace, node, move)
    println("Accepted move $(move) : $(aux.accepts > 0)")
    return nothing
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

    # number of steps for init kernel
    init_cycles::Int64 = 10
end


mutable struct AdaptiveAux <: AuxillaryState
    sensitivities::Matrix{Float64}
    queue::PriorityQueue{Int64, Float64, ReverseOrdering}
    node::Int64
end

function AuxState(p::AdaptiveComputation, trace)
    dims = first(get_args(trace)).dims
    ndims = prod(dims)
    sensitivities = zeros(dims)
    queue = init_queue(trace)
    AdatpiveAux(sensitivities,
                queue,
                0)
end

function select_node(p::AdaptiveComputation, aux::AdaptiveAux)
    node, gr = first(aux.queue)
end

function rw_block_init!()
    # accept_ct::Int64 = 0
    # delta_pi::Float64 = 0.0
    # delta_s::Float64 = 0.0
end

function rw_inc_block!()
        # obj_t_prime = objective(_t)
        # delta_pi += distance(obj_t, obj_t_prime)
        # delta_s += exp(clamp(alpha, -Inf, 0.))
end

function rw_block_extend!()
    # accept_ratio = accept_ct / rw_cycles
    # addition_rw_cycles = (delta_pi > 0) * floor(Int64, sm_cycles * accept_ratio)
end

function rw_block_complete!(aux, protocol)
    # compute goal-relevance
    total_cycles = rw_cycles + addition_rw_cycles
    delta_pi /= total_cycles
    delta_s /= total_cycles
    goal_relevance = delta_pi * delta_s

    # update aux state
    prod_node = traverse_qt(qt, node).node
    sidx = node_to_idx(prod_node, max_leaves(qt))
    aux.sensitivities[sidx] .= goal_relevance
    aux.queue[node] = goal_relevance
    aux.node = node

    println("\t delta pi: $(delta_pi)")
    println("\t delta S: $(delta_s)")
    println("\t goal relevance: $(goal_relevance)")
    println("\t rw acceptance ratio: $(accept_ratio)")
end

function sm_block_accept!(aux::AdaptiveAux, trace, node, move)
        accept_ct += 1
        update_queue!(aux, node, move)
end

function sm_block_complete!()
    println("\t accepted SM move: $(accept_ct == 1)")
    return nothing
end

function init_queue(tr::Gen.Trace)
    qt = get_retval(tr)
    q = PriorityQueue{Int64, Float64, ReverseOrdering}(Reverse)
    # go through the current set of terminal nodes
    # and intialize priority
    for n = qt.leaves
        q[n.node.tree_idx] = 0.1 * area(n.node)
    end
    return q
end

function update_queue!(queue, node::Int64, move::Split)
    prev_val = queue[node] * 0.25
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
    prev_val = 0.
    for i = 1:4
        cid = Gen.get_child(parent, i, 4)
        prev_val += queue[cid]
        delete!(queue, cid)
    end
    queue[parent] = prev_val
    return nothing
end
