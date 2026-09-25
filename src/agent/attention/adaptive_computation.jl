export AdaptiveComputation

################################################################################
# Adaptive Computation
################################################################################

@with_kw struct AdaptiveComputation{V} <: AttentionProtocol
    vis_partition::TracePartition{V} = WMPartition{V}()
    "Minimum number of moves"
    base_steps::Int64 = 3
    "Size of attention hash map"
    buffer_size::Int64 = 100
    "Distance metric over XY"
    map_metric::PreMetric = Euclidean()
    "Number of nearest neighbors"
    nns::Int64 = 5
    "Importance softmax temperature"
    itemp::Float64 = 3.0
    "Maximumal load"
    load::Int64 = 20
    "Load curve slope"
    load_m::Float64 = 20.0
    "Load curve intercept"
    load_x0::Float64 = 5.0
end

mutable struct AdaptiveAux <: MentalState{AdaptiveComputation}
    "Impact of C_k on decision-making"
    dPi::HashMap
    "Impact of C_k on perception"
    dS::HashMap
    "Array used for task-relevance integration indeces"
    nn_idxs::Vector{Int32}
    "Array used for task-relevance integration distances"
    nn_dists::Vector{Float64}
    "Statistic over average load"
    avg_load::Float64
end

AdaptiveAux(n::Int, k::Int) = AdaptiveAux(HashMap(S2V, Float64,   n), # dPi
                                          HashMap(S2V, Float64,   n), # dS
                                          zeros(Int32, k),
                                          zeros(Float64, k),
                                          0.0
                                          ) 

function AttentionModule(m::AdaptiveComputation)
    MentalModule(m, AdaptiveAux(m.buffer_size, m.nns))
end

Base.isempty(x::AdaptiveAux) = isempty(x.dPi) || isempty(x.dK)

function update_impact!(buffer::HashMap,
                        partition::TracePartition{T},
                        trace::T,
                        deltas::Dict
                        ) where {T<:Trace}
    for (k,v) = deltas
        coord = get_coord(partition, trace, k)
        push_sample!(buffer, coord, v)
    end
    return nothing
end

function update_impact!(buffer::HashMap,
                        partition::TracePartition{T},
                        trace::T,
                        j::Int,
                        delta::Float64,
                        ) where {T<:Trace}
    coord = get_coord(partition, trace, j)
    push_sample!(buffer, coord, delta)
    return nothing
end

function update_task_relevance!(att::MentalModule{A}
                                ) where {A<:AdaptiveComputation}
    attp, attstate = mparse(att)
    fit_map!(attstate.dPi, attp.map_metric)
    fit_map!(attstate.dS , attp.map_metric)
    return nothing
end

function load(p::AdaptiveComputation, deltas::Vector{Float64})
    x = logsumexp(deltas)
    x = (x - p.load_x0) / p.load_m
    l = p.load * exp(min(x, 0.0))
    # println("| Agg. Delta: $(round(x; digits=2)) | Load: $(l)")
    return l
end

function task_relevance!(aux::AdaptiveAux,
                         dPi::HashMap,
                         dS::HashMap,
                         partition::TracePartition{T},
                         trace::T,
                         ) where {T<:Gen.Trace}
    n = latent_size(partition, trace)
    # NOTE: case with empty estimate?
    # No info yet -> -Inf
    (isempty(dPi) || isempty(dS)) && return fill(-Inf, n)
    tr = Vector{Float64}(undef, n)
    # Preallocating reused arrays
    for i = 1:n
        coord = get_coord(partition, trace, i)
        _dpi  = integrate!(aux.nn_idxs, aux.nn_dists, coord, dPi)
        _ds   = integrate!(aux.nn_idxs, aux.nn_dists, coord, dS)
        @printf "| δπ: %.2f \t | δS: %.2f |\n" _dpi _ds
        # @printf "| δπ: %.2f \t |\n" _dpi
        tr[i] = _dpi + _ds
    end
    return tr
end

function step_module!(att::MentalModule{AdaptiveComputation},
                      vis::MentalModule{AdaptiveMH},
                      planning::MentalModule{AStarPlanner})
    update_task_relevance!(att)
    attend!(att, vis)
    attend!(att, planning, vis)
    return nothing
end

function attend!(att::MentalModule{AdaptiveComputation},
                 perception::MentalModule{AdaptiveMH})

    vprotocol, vstate = mparse(perception)
    aprotocol, astate = mparse(att)

    @unpack vis_partition, base_steps, itemp = aprotocol

    trace = vstate.samples[end]
    deltas = task_relevance!(astate, astate.dPi, astate.dS, vis_partition, trace)
    importance = softmax(deltas, itemp)
    # TODO: figure out load curve
    tload = aprotocol.load # load(aprotocol, deltas)
    # Number of latents to choose from
    n = length(importance)
    for step = 1:(base_steps + tload)
        idx = step <= base_steps ?
            uniform_discrete(1, n) :
            categorical(importance)
        prop = select_prop(vis_partition, trace, idx)
        new_trace, w = prop(trace, idx)
        # MH acceptance function
        if log(rand()) < w
            trace = new_trace
            # Add new trace to chain
            push!(vstate.samples, new_trace)
            # increment weight
            push!(vstate.weights, vstate.weights[end] + w)
            dS = min(w, 0.)
        else
            dS = -Inf
        end

        update_impact!(astate.dS, vis_partition, new_trace, idx, dS)
    end
    return nothing
end

function attend!(att::MentalModule{AdaptiveComputation},
                 planning::MentalModule{AStarPlanner},
                 perception::MentalModule{AdaptiveMH})

    aprotocol, astate = mparse(att)
    pprotocol, pstate = mparse(planning)
    vprotocol, vstate = mparse(perception)

    mass = logsumexp(vstate.weights)

    n = length(vstate.samples)
    for i = 1:n
        vtrace = vstate.samples[i]
        vweight = vstate.weights[i] - mass
        qt = get_retval(vtrace)
        score, path, dPi =
            qt_a_star(qt,
                      pprotocol.obstacle_cost,
                      pprotocol.ent, pprotocol.ext,
                      pprotocol.grad_tau)

        pstate.scores[i] = -score + vweight
        pstate.paths[i] = path

        update_impact!(astate.dPi, aprotocol.vis_partition, vtrace, dPi)
    end
    return nothing
end
