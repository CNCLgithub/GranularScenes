export qt_model

#################################################################################
# Generative Model
#################################################################################
@gen function qt_no_change(qt::QuadTree)
    return 0
end

@gen function qt_change(qt::QuadTree)
    ws = change_weights(qt)
    leaf ~ categorical(ws)
    return leaf
end

const qt_change_switch = Gen.Switch(qt_no_change, qt_change)

@gen function qt_change_kernel(t::Int, qt::QuadTree, params::QuadTreeModel)
    # change prior
    change ~ bernoulli(0.5)
    switch_idx = change ? 2 : 1
    # loc =
    # 0 => no change
    # [1, n] => change at leaf
    location ~ qt_change_switch(switch_idx, qt)
    qt_prime::QuadTree = apply_changes(qt, location)

    {:img_b} ~ observe_pixels(params.renderer,
                              qt_prime,
                              params.pixel_var)

    return qt_prime
end

@gen function qt_model(t::Int, params::QuadTreeModel)
    # sample quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    qt::QuadTree = QuadTree(root)

    # first image (X_0)
    {:img_a} ~ observe_pixels(params.renderer, qt, params.pixel_var)

    # second image (X_1) (could be the same image)
    changes ~ Gen.Unfold(qt_change_kernel)(t, qt, params)

    return qt
end
