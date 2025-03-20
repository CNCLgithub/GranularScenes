export qt_model

#################################################################################
# Generative Model
#################################################################################
@gen static function uniform_change_prior(x::QTAggNode)
    # b ~ uniform(0.0, 1.0)
    b ~ beta(0.5, 5.5)
    return b
end

@gen static function no_change(qt::QuadTree)
    return 0
end

@gen static function qt_change(qt::QuadTree)
    loc_prior ~ Map(uniform_change_prior)(leaves(qt))
    idx = argmax(loc_prior)
    return idx
end

const qt_change_switch = Gen.Switch(no_change, qt_change)

@gen function qt_model(params::QuadTreeModel)
    # initialize quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    # qt struct
    qt::QuadTree = QuadTree(root)

    # change prior
    change ~ bernoulli(0.5)
    change_model = change ? 1 : 2
    # can be 0 (no change) or 0 < i <= length(qt)
    loc = qt_change_switch(change_model, qt)
    qt_prime = apply_changes(qt, loc)

    # predict pixels from occupancy probabilities
    {:img_a} ~ observe_pixels(params.renderer, qt, params.pixel_var)
    {:img_b} ~ observe_pixels(params.renderer, qt_prime,
                              params.pixel_var)

    result::Tuple{QuadTree, Int} = (qt, loc)
    return result
end
