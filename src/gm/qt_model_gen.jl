export qt_model

#################################################################################
# Generative Model
#################################################################################

@gen function qt_model(params::QuadTreeModel)
    # sample quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    qt::QuadTree = QuadTree(root)

    # predict pixels from occupancy probabilities
    depth ~ qt_observe(params.renderer, qt, params.pixel_var)

    return qt
end

function node_lambda(params::QuadTreeModel, depth::Int)
    l_min, l_0 = params.lambda_min, params.lambda_0
    lambda = l_min + (l_0 - l_min)^(depth - 1)
end

function node_kappa(params::QuadTreeModel, depth::Int)
    params.k_0 * exp2(depth - 1)
end

@gen (static) function occupancy_prior(node, params::QuadTreeModel)
    # how deep are we? 
    lambda = node_lambda(params, node.level)
    kappa = node_kappa(params, node.level)

    mean ~ beta(lambda, lambda)
    w  ~ beta(kappa*mean, kappa*(1-mean))
end

function bind_weights(g::GranularitySchema, weights)

end

@gen (static) function map_occupancy_weights(g::GranularitySchema, params::QuadTreeModel)
    # 1. extract leaf nodes
    leaves = g.leaves
    # 2. sample occupancy weights
    weights ~ Gen.Map(occupancy_prior)(leaves, Fill(params, length(leaves)))
    qt::QuadTree = bind_weights(g, weights)
    return qt
end

@gen (static) function qt_local(params::QaudTreeModel, g::GranularitySchema)
    qt ~ map_occupancy_weights(g, params)
    # predict pixels from occupancy probabilities
    depth ~ qt_observe(params.renderer, qt, params.pixel_var)
    return qt
end
