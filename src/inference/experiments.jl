export run_inference, resume_inference, query_from_params, proc_from_params

function ex_obstacles(c::AMHChain)
    qt = first(get_retval(c.state))
    project_qt(qt)
end

function ex_img(c::AMHChain)
    p = first(get_args(c.state))
    qt = first(get_retval(c.state))
    img = @pycall render(p.renderer, qt).to_numpy()::Array{Float32,3}
end

function ex_granularity(c::AMHChain)
    qt = first(get_retval(c.state))
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
    qt = first(get_retval(tr))
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

function ex_loc_change(tr::Gen.Trace)
    qt, loc_ws = get_retval(tr)
    n = max_leaves(qt)
    leaves = qt.leaves
    m = Matrix{Float32}(undef, n, n)
    for (vi, x) in enumerate(qt.leaves)
        lw = loc_ws[vi]
        for idx = node_to_idx(node(x), n)
            m[idx] = lw
        end
    end
    m
end
function ex_loc_change(c::AMHChain)
    ex_loc_change(c.state)
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
        :likelihood => c -> Gen.project(c.state, select(:pixels)),
    ))
    # define the posterior over qt geometries
    query = Gen_Compose.StaticQuery(lm,
                                    qt_model,
                                    (gm_params,),
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
