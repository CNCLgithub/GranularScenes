export qt_vision

#################################################################################
# Generative Model
#################################################################################

function node_lambda(params::QTVisionPrior, depth::Int)
    l_min, l_0 = params.lambda_min, params.lambda_0
    lambda = l_min + (l_0 - l_min)^(depth - 1)
end

function node_kappa(params::QTVisionPrior, depth::Int)
    params.k_0 * exp2(depth - 1)
end

@gen (static) function occupancy_prior(node, params::QTVisionPrior)
    # how deep are we? 
    lambda = node_lambda(params, node.level)
    kappa = node_kappa(params, node.level)

    mean ~ beta(lambda, lambda)
    w  ~ beta(kappa*mean, kappa*(1-mean))
end

function bind_weights(g::QTSchema, weights)::QuadTree
    QuadTree(g, Dict{NodeId, Float64}(zip(g.leaves, weights)))
end

@gen (static) function map_occupancy_weights(g::QTSchema,
                                             p::QTVisionPrior)
    # 1. extract leaf nodes
    leaves = g.leaves
    # 2. sample occupancy weights
    weights ~ Gen.Map(occupancy_prior)(leaves, Fill(p, length(leaves)))
    qt::QuadTree = bind_weights(g, weights)
    return qt
end

@gen (static) function qt_vision(g::QTSchema,
                                 p::QTVisionPrior,
                                 l::QTVisionLikelihood)
    qt ~ map_occupancy_weights(g, p)
    # predict pixels from occupancy probabilities
    depth ~ qt_observe(l.renderer, qt, l.pixel_var)
    return qt
end
