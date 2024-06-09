export qt_model

#################################################################################
# Generative Model
#################################################################################
@gen function qt_model(params::QuadTreeModel)
    # initialize quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    # qt struct
    qt::QuadTree = QuadTree(root)

    # predict pixels from occupancy probabilities
    {:pixels} ~ observe_pixels(params.renderer, qt, params.pixel_var)

    return qt
end

# shortest path given qt uncertainty
# qtpath::QTPath = qt_a_star(qt, params.obs_cost, params.entrance, params.exit)
#
# Model state
# result::QuadTreeState = QuadTreeState(qt, mu, var, qtpath)
