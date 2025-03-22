export run_inference, resume_inference, query_from_params, proc_from_params, extend_query

function ex_obstacles(c::AMHChain)
    qt = get_retval(c.state)
    project_qt(qt)
end

function ex_img(c::AMHChain)
    _, p = get_args(c.state)
    qt = get_retval(c.state)
    img = @pycall render(p.renderer, qt).to_numpy()::Array{Float32,3}
end

function ex_granularity(c::AMHChain)
    qt = get_retval(c.state)
    n = max_leaves(qt)
    m = Matrix{UInt8}(undef, n, n)
    for x in qt.leaves
        for idx = node_to_idx(x.node, n)
            m[idx] = x.node.level
        end
    end
    m
end

function ex_path(tr::Gen.Trace)
    qt = get_retval(tr)
    n = max_leaves(qt)
    cost, path = quad_tree_path(tr)
    leaves = qt.leaves
    m = Matrix{Bool}(undef, n, n)
    fill!(m, false)
    for i = path
        node = leaf_from_idx(qt, i).node
        for idx = node_to_idx(node, n)
            m[idx] = true
        end
    end
    m
end
function ex_path(c::AMHChain)
    ex_path(c.state)
end

function ex_attention(c::AMHChain)
    attention_map(c.auxillary, c.state)
end

function task_error(c::AMHChain, gt::Int)
    error = 0.0
    if gt == 0
        # no change; get coinflip should be false
        error += getindex(c.state, :changes => 1 => :change) ? 1.0 : 0.0
    else
        # change; earth movers distance
        # error += getindex(c.state, :changes => 1 => :change) ? 0.0 : 1.0
        error += loc_error(c, gt)
    end
    return error
end

function loc_error(c::AMHChain, gt::Int)
    qt = get_retval(c.state)
    ws = change_weights(qt)
    ml = max_leaves(qt)
    # How likely was the change at the GT?
    idx = idx_to_node_space(gt, ml)
    node_at_gt = node(traverse_qt(qt, idx))
    weight_at_gt = ws[qt.mapping[tree_idx(node_at_gt)]]
    # How likely was the change near the GT?
    le = 0.0
    for current = leaves(qt)
        le += nearby_error(qt, weight_at_gt, node_at_gt, ws,
                           node(current))
    end
    # Penalize for coarse granulatiy at the GT change
    # e.g., if the node is 4x4, then there is 1/4 chance of
    # localizing correctly
    penalty = exp2(2 * (node_at_gt.max_level - node_at_gt.level))
    le *= penalty
    return le
end

"""
Surrogate earth mover's distance between a given node
and the location of the GT change.
"""
function nearby_error(qt, weight_at_gt, node_at_gt, ws, current)
    d = dist(node_at_gt, current)
    w = max(0.0, (ws[qt.mapping[tree_idx(current)]] - weight_at_gt))
    iszero(d) ? 0.0 : w  / d
end

"""
    ptest(test, chain)

Evaluates a sample of the posterior predictive
distribution on a test image.
"""
function ptest(test::ChoiceMap, c::AMHChain)
    state = c.state
    tr, w = Gen.update(state, test)
    Gen.project(tr, select(:pixels))
    # return w
end

function query_from_params(room::GridRoom, gm_params::QuadTreeModel)

    obs = create_obs(gm_params, room)
    lm = LatentMap(Dict{Symbol, Any}(
        :attention => ex_attention,
        :obstacles => ex_obstacles,
        :granularity => ex_granularity,
        :path => ex_path,
        :img => ex_img,
        :score => c -> Gen.get_score(c.state),
        :likelihood => c -> Gen.project(c.state, select(:img_a)),
    ))
    # define the posterior over qt geometries
    query = Gen_Compose.StaticQuery(lm,
                                    qt_model,
                                    (0, gm_params),
                                    obs)
end

function extend_query(query::StaticQuery, room::GridRoom,
                      gt_change::Int)
    (_, gm_params) = query.args
    key = :changes => 1 => :img_b
    change_key = :changes => 1 => :change
    loc_key = :changes => 1 => :location
    obs = create_obs(gm_params, room, key)
    lm = LatentMap(Dict{Symbol, Any}(
        :attention => ex_attention,
        :obstacles => ex_obstacles,
        :granularity => ex_granularity,
        :path => ex_path,
        :img => ex_img,
        :score => c -> Gen.get_score(c.state),
        :likelihood => c -> Gen.project(c.state, select(:img_a, key)),
        :change => c -> getindex(c.state, change_key),
        :task_error => c -> task_error(c, gt_change),
    ))
    # define the posterior over qt geometries
    query = Gen_Compose.StaticQuery(lm,
                                    qt_model,
                                    (1, gm_params),
                                    obs)
end


function query_from_params(first::GridRoom, second::GridRoom,
                           gm_params::QuadTreeModel)

    obs = create_obs(gm_params, first, second)
    lm = LatentMap(Dict{Symbol, Any}(
        :attention => ex_attention,
        :obstacles => ex_obstacles,
        :change => ex_loc_change,
        :path => ex_path,
        # :granularity => ex_granularity,
        # :img => ex_img,
        :score => c -> Gen.get_score(c.state),
        :likelihood => c -> Gen.project(c.state, select(:img_a, :img_b)),
    ))
    # define the posterior over qt geometries
    query = Gen_Compose.StaticQuery(lm,
                                    qt_model,
                                    (gm_params,),
                                    obs)
end
