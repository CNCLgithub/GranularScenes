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
    depth ~ qt_observe(params.renderer, qt, params.pixel_var)

    return qt
end
