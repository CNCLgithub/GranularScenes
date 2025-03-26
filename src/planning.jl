export search_step!, train!, test_change!, test_same

using LinearAlgebra: lmul!
using Zygote: withgradient

function search_step!(c::AMHChain,
                      l::ChainLogger,
                      steps::Int = 50,
                      search_weight::Float64 = 1.0)
    train!(c, l, steps)
    result = (qt, lw, dpi_kl) = test(c)
    update_deltapi!(c, dpi_kl, search_weight * c.step)
    # viz_chain(c)
    return result
end

function train!(c::AMHChain, log::ChainLogger,
                steps::Int = 50)
    for _ = 1:steps
        Gen_Compose.is_finished(c) && break
        Gen_Compose.step!(c)
        Gen_Compose.report_step!(log, c)
        Gen_Compose.increment!(c)
        # viz_chain(c)
    end
    return nothing
end

function test_same(c::AMHChain, logger::ChainLogger, q1)
    prob_same(c, q1)
    # m = ContinuousMarginal{Float64}(;
    #                                 op = logsumexp,
    #                                 norm = (x, n) -> x - log(n),
    #                                 id = x -> -Inf)
    # marginalize(m, buffer(logger), :prob_same)
end

function test_change!(c::AMHChain, logger::ChainLogger,
               steps::Int = 50)
    for _ = 1:steps
        Gen_Compose.is_finished(c) && break
        change_step!(c)
        Gen_Compose.report_step!(logger, c)
        Gen_Compose.increment!(c)
    end
    m = ContinuousMarginal{Float64}(;
                                    op = logsumexp,
                                    norm = (x, n) -> x - log(n),
                                    id = x -> -Inf)
    marginalize(m, buffer(logger), :prob_change)
end

function test(c::AMHChain)
    (qt, loc_ws) = get_retval(c.state)
    ws = loc_ws ./ maximum(loc_ws)
    # ws = softmax(loc_ws)
    clamp!(ws, 0.01, 0.99)
    grads = map(x -> (1E-4 + abs(log(-x + 1) - log(x))), ws)
    # grads = map(x -> 1.0 / (1E-4 + abs(log(-x + 1) - log(x))), ws)
    # mx = argmax(loc_ws)
    (qt, ws, grads)
end

function update_deltapi!(c::AMHChain, dpi_kl::Vector{Float64},
                         weight::Float64 = 1.0)
    aux = auxillary(c)
    state = estimate(c)
    qt = get_retval(state)
    ml = max_leaves(qt)
    lw = log(weight)
    n = length(dpi_kl)
    lvs = leaves(qt)
    @assert length(lvs) === n "search gradients missmatch qt leaves"
    for i = 1:n
        nde = node(lvs[i])
        tidx = tree_idx(nde)
        delta = log(dpi_kl[i]) + lw
        aux.queue[tidx] = logsumexp(aux.queue[tidx], delta)
    end
    return nothing
end
