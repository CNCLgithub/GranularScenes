export qt_model

#################################################################################
# Generative Model
#################################################################################
@gen static function uniform_change_prior(x::QTAggNode)
    b ~ uniform(0.0, 1.0)
    return b
end

@gen function qt_model(params::QuadTreeModel)
    # initialize quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    # qt struct
    qt::QuadTree = QuadTree(root)

    # change location prior
    loc_prior ~ Map(uniform_change_prior)(leaves(qt))

    qt_prime = apply_changes(qt, loc_prior)


    # predict pixels from occupancy probabilities
    {:img_a} ~ observe_pixels(params.renderer, qt, params.pixel_var)
    {:img_b} ~ observe_pixels(params.renderer, qt_prime,
                              params.pixel_var)

    result::Tuple{QuadTree, Array{Float64}} = (qt, loc_prior)
    return result
end

# shortest path given qt uncertainty
# qtpath::QTPath = qt_a_star(qt, params.obs_cost, params.entrance, params.exit)
#
# Model state
# result::QuadTreeState = QuadTreeState(qt, mu, var, qtpath)
