export QTObserve,
    qt_observe

include("qt_render.jl")

struct QTObserve <: Gen.Distribution{Array{Float32}} end

const qt_observe = QTObserve()

function Gen.random(::QTObserve, r::QuadTreeRenderer, qt::QuadTree, var::Float32)
    write_obstacles!(r, qt)
    sampled = random(r, var)
    Array(sampled)
end

function Gen.logpdf(::QTObserve, x::Array{Float32}, r::QuadTreeRenderer, qt::QuadTree, var::Float32)
    write_obstacles!(r, qt)
    ls = logpdf(r, x, var)
    Float64(ls)
end

(::QTObserve)(r, qt, var) = Gen.random(qt_observe, r, qt, var)

is_discrete(::QTObserve) = false
Gen.has_output_grad(::QTObserve) = false
Gen.logpdf_grad(::QTObserve, value::Set, args...) = (nothing,)
