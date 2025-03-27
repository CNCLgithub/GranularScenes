export broadcasted_bernoulli, broadcasted_uniform, labelled_categorical, beta_mixture

#################################################################################
# Generic distributions
#################################################################################

@dist function labelled_categorical(xs)
    n = length(xs)
    probs = fill(1.0 / n, n)
    index = categorical(probs)
    xs[index]
end

@dist function id(x)
    probs = ones(1)
    xs = fill(x, 1)
    index = categorical(probs)
    xs[index]
end

#################################################################################
# Broadcasted Distributions for efficiency
#################################################################################

struct BroadcastedBernoulli <: Gen.Distribution{AbstractArray{Bool}} end

const broadcasted_bernoulli = BroadcastedBernoulli()

function Gen.random(::BroadcastedBernoulli, ws::Array{Float64})
    result = BitArray(undef, size(ws))
    for i in LinearIndices(ws)
        result[i] = bernoulli(ws[i])
    end
    return result
end

function Gen.logpdf(::BroadcastedBernoulli, xs::AbstractArray{Bool},
                    ws::AbstractArray{Float64})
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(bernoulli, xs[i], ws[i])
    end
    return ll
end

(::BroadcastedBernoulli)(ws) = Gen.random(BroadcastedBernoulli(), ws)

is_discrete(::BroadcastedBernoulli) = true
Gen.has_output_grad(::BroadcastedBernoulli) = false
Gen.logpdf_grad(::BroadcastedBernoulli, value::Set, args...) = (nothing,)


struct BroadcastedUniform <: Gen.Distribution{AbstractArray{Float64}} end

const broadcasted_uniform = BroadcastedUniform()

function Gen.random(::BroadcastedUniform, ws::AbstractArray{Tuple{Float64, Float64}})
    result = Array{Float64}(undef, size(ws)...)
    for i in LinearIndices(ws)
        a,b = ws[i]
        result[i] = uniform(a, b)
    end
    return result
end

function Gen.logpdf(::BroadcastedUniform,
                    xs::Float64,
                    ws::AbstractArray{Tuple{Float64, Float64}})
    Gen.logpdf(uniform, xs, ws[1]...)
end

function Gen.logpdf(::BroadcastedUniform,
                    xs::AbstractArray{Float64},
                    ws::AbstractArray{Tuple{Float64, Float64}})
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(uniform, xs[i], ws[i]...)
    end
    return ll
end

(::BroadcastedUniform)(ws) = Gen.random(BroadcastedUniform(), ws)

function logpdf_grad(::BroadcastedUniform,
                     xs::Union{AbstractArray{Float64}, Float64},
                     ws::AbstractArray{Tuple{Float64, Float64}})
    # s = size(ws)
    # Gen.assert_has_shape(xs, s;
    #                  msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    inv_diff = 0.
    for i in LinearIndices(s)
        low, high = ws[i]
        inv_diff += 1.0 / (high - low)
    end
    (0., inv_diff, -inv_diff)
end


is_discrete(::BroadcastedUniform) = false
has_output_grad(::BroadcastedUniform) = true
has_argument_grads(::BroadcastedUniform) = (true, true)



struct BroadcastedPiecewiseUniform <: Gen.Distribution{AbstractArray{Float64}} end


const broadcasted_piecewise_uniform = BroadcastedPiecewiseUniform()

function Gen.random(::BroadcastedPiecewiseUniform,
                    ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    result = Array{Float64}(undef, size(ws)...)
    for i in LinearIndices(ws)
        a,b = ws[i]
        result[i] = piecewise_uniform(a, b)
    end
    return result
end

function Gen.logpdf(::BroadcastedPiecewiseUniform,
                    xs::Float64,
                    ws::AbstractArray{Tuple})
    Gen.logpdf(piecewise_uniform, xs, ws[1]...)
end

function Gen.logpdf(::BroadcastedPiecewiseUniform,
                    xs::AbstractArray{Float64},
                    ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    s = size(ws)
    Gen.assert_has_shape(xs, s;
                     msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    ll = 0
    for i in LinearIndices(s)
        ll += Gen.logpdf(piecewise_uniform, xs[i], ws[i]...)
    end
    return ll
end

(::BroadcastedPiecewiseUniform)(ws) = Gen.random(BroadcastedPiecewiseUniform(), ws)

function logpdf_grad(::BroadcastedPiecewiseUniform,
                     xs::Union{AbstractArray{Float64}, Float64},
                     ws::AbstractArray{Tuple{Vector{Float64}, Vector{Float64}}, N}) where {N}
    # s = size(ws)
    # Gen.assert_has_shape(xs, s;
    #                  msg="`xs` has size $(size(xs))` but `ws` is size $(s)")
    a, b = 0., 0.
    for i in LinearIndices(s)
        _, _a, _b = Gen.logpdf_grad(piecewise_uniform, xs[i], ws[i]...)
        a += _a
        b += _b
    end
    (0., a, b)
end


is_discrete(::BroadcastedPiecewiseUniform) = false
has_output_grad(::BroadcastedPiecewiseUniform) = true
has_argument_grads(::BroadcastedPiecewiseUniform) = (true, true)


#################################################################################
# mixture of betas
#################################################################################
struct BetaMixture <: Distribution{Float64} end

const beta_mixture = BetaMixture()

function Gen.random(::BetaMixture, w::Float64, a1::Float64, b1::Float64, a2::Float64, b2::Float64)
    a,b = bernoulli(w) ? (a1, b1) : (a2, b2)
    beta(a, b)
end

function Gen.logpdf(::BetaMixture,
                    x::Float64,
                    w::Float64, a1::Float64, b1::Float64, a2::Float64, b2::Float64)
    (x < 0 || x > 1) && return  -Inf
    lw = log(w)
    l1 = lw + logpdf(beta, x, a1, b1)
    l2 = log1mexp(lw) + logpdf(beta, x, a2, b2)
    logsumexp(l1, l2)
end



function logpdf_grad(::BetaMixture, x::Float64, w::Float64, a1::Float64, b1::Float64, a2::Float64, b2::Float64)
    beta1_logpdf = logpdf(beta, x, a1, b1)
    beta2_logpdf = logpdf(beta, x, a1, b2)
    beta1_grad = logpdf_grad(beta, x, a1, b1)
    beta2_grad = logpdf_grad(beta, x, a2, b2)
    w1 = 1. / (1. + exp(log(1. - w) + beta2_logpdf - log(w) - beta1_logpdf))
    w2 = 1. - w1
    x_deriv = w1 * beta1_grad[1] + w2 * beta2_grad[1]
    a1_deriv = w1 * beta1_grad[2]
    b1_deriv = w1 * beta1_grad[3]
    a2_deriv = w2 * beta2_grad[2]
    b2_deriv = w2 * beta2_grad[3]
    w_deriv = (exp(beta1_logpdf) - exp(beta2_logpdf)) / (w * exp(beta1_logpdf) + (1. - theta) * exp(beta2_logpdf))
    (x_deriv, w_deriv, a1_deriv, b1_deriv, a2_deriv, b2_deriv)
end

function random(::BetaMixture, theta::Real, alpha::Real, beta::Real)
    if bernoulli(theta)
        random(Beta(), alpha, beta)
    else
        random(uniform_continuous, 0., 1.)
    end
end

(::BetaMixture)(w, a1, b1, a2, b2) = Gen.random(BetaMixture(), w, a1, b1, a2, b2)

is_discrete(::BetaMixture) = false
has_output_grad(::BetaMixture) = true
has_argument_grads(::BetaMixture) = (true, true, true, true, true)
