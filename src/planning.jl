export search_step!

using LinearAlgebra: lmul!
using Zygote: withgradient

function search_step!(c1::AMHChain, c2::AMHChain,
                      l1::ChainLogger, l2::ChainLogger,
                      steps::Int = 50,
                      search_weight::Float64 = 1.0)
    m1 = train!(c1, l1, steps)
    m2 = train!(c2, l2, steps)
    result = (klm, max_d, c, dpi_kl) = test(m1, m2)
    update_deltapi!(c1, dpi_kl, search_weight)
    update_deltapi!(c2, dpi_kl, search_weight)
    println("Attention with search")
    att = ex_attention(c1)
    lmul!(1.0 / maximum(att), att)
    display_mat(att; c2 = colorant"red")
    return result
end

function train!(c::AMHChain, log::ChainLogger,
                steps::Int = 50)
    for _ = 1:steps
        Gen_Compose.is_finished(c) && break
        Gen_Compose.step!(c)
        Gen_Compose.report_step!(log, c)
        Gen_Compose.increment!(c)
    end
    marginal = marginalize(buffer(log), :obstacles)
end

function test(m1::Matrix, m2::Matrix)
    max_d = 0.0
    c = CartesianIndex(0, 0)
    klm = similar(m1)
    dpi = similar(m1)
    @inbounds for i = CartesianIndices(m1)
        d, g = kl_wgrad(m1[i], m2[i])
        klm[i] = d
        dpi[i] = norm(g)
        if d > max_d
            max_d = d
            c = i
        end
    end
    lmul!(1.0 / max_d, klm)
    (klm, max_d, c, dpi)
end

function kl(p::Real, q::Real)
    p = clamp(p, 0.01, 0.99)
    q = clamp(q, 0.01, 0.99)
    log(1 - p) - log(1 - q) +
        p * (log(p) + log(1-q) - log(q) - log(1 - p))
end

kl_wgrad(p::Real, q::Real) = withgradient(kl, p, q)

function marginalize(bfr, key::Symbol)
    n = length(bfr)
    # @show n
    @assert n > 0
    marginal = similar(bfr[1][key])
    fill!(marginal, 0.0)
    for i = 1:n
        datum = bfr[i][key]
        for j = eachindex(marginal)
            marginal[j] += datum[j]
        end
    end
    lmul!(1.0 / n, marginal)
end

function update_deltapi!(c::AMHChain, dpi_kl::Matrix, weight::Float64 = 1.0)
    aux = auxillary(c)
    state = estimate(c)
    params = first(get_args(state))
    nc = params.dims[2] # num cols
    qt = get_retval(state)
    lw = log(weight)
    @inbounds for li = LinearIndices(dpi_kl)
        cell = room_to_leaf(qt, li, nc)
        i = tree_idx(node(cell))
        w = log(dpi_kl[li]) + lw
        aux.queue[i] = logsumexp(aux.queue[i], w)
    end
    return nothing
end
