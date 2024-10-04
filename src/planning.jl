export search_step!

using LinearAlgebra: lmul!
using Zygote: withgradient

function search_step!(c1::AMHChain, c2::AMHChain,
                      l1::ChainLogger, l2::ChainLogger,
                      steps::Int = 50,
                      search_weight::Float64 = 1.0)
    m1 = train!(c1, l1, steps)
    m2 = train!(c2, l2, steps)
    (klm, max_d, c, dpi_kl) = test(m1, m2)
    eloc = expected_loc(klm)
    result = (klm, max_d, c, eloc)
    weight = search_weight # * Gen_Compose.step(c1)
    # update_deltapi!(c1, dpi_kl, weight)
    # update_deltapi!(c2, dpi_kl, weight)
    viz_chain(l1)
    viz_chain(l2)
    # println("Expected $(eloc); Max KL: $(max_d), @ index $(c)")
    # @show weight
    # display_mat(klm)
    return result
end

function expected_loc(kl::Matrix)
    loc_x = 0.0
    loc_y = 0.0
    skl = sum(kl)
    @inbounds for ci = CartesianIndices(kl)
        x, y = Tuple(ci)
        w = kl[ci] / skl
        loc_x += w * x
        loc_y += w * y
    end
    return SVector{2, Float64}(loc_x, loc_y)
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
    sum_d = 0.0
    c = CartesianIndex(0, 0)
    klm = similar(m1)
    dpi = similar(m1)
    @inbounds for i = CartesianIndices(m1)
        d, g = kl_wgrad(m1[i], m2[i])
        klm[i] = d
        dpi[i] = norm(g)
        sum_d += d
        if d > max_d
            max_d = d
            c = i
        end
    end
    prop_d = max_d / sum_d
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

function update_deltapi!(c::AMHChain, dpi_kl::Matrix, weight::Float64 = 1.0)
    aux = auxillary(c)
    state = estimate(c)
    lw = log(weight)
    @assert size(dpi_kl) == size(aux.gr)
    @inbounds for li = LinearIndices(dpi_kl)
        # @show aux.gr[li]
        # @show log(dpi_kl[li])
        aux.gr[li] =
            logsumexp(aux.gr[li], log(dpi_kl[li]) + lw)
        # println("new aux.gr[li] $(aux.gr[li])")
    end
    update_queue!(aux.queue, aux.gr, state)
    return nothing
end
