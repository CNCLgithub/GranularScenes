export run_inference, resume_inference, query_from_params, proc_from_params

function ex_obstacles(c::AMHChain)
    qt = get_retval(c.state)
    project_qt(qt)
end

function ex_img(c::AMHChain)
    p = first(get_args(c.state))
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

function ex_path(c::AMHChain)
    qt = get_retval(c.state)
    n = max_leaves(qt)
    path = quad_tree_path(c.state)
    leaves = qt.leaves
    m = Matrix{Bool}(undef, n, n)
    fill!(m, false)
    for e in path.edges
        src_node = leaves[src(e)].node
        for idx = node_to_idx(src_node, n)
            m[idx] = true
        end
        dst_node = leaves[dst(e)].node
        for idx = node_to_idx(dst_node, n)
            m[idx] = true
        end
    end
    m
end

function ex_attention(c::AMHChain)
    attention_map(c.auxillary, c.state)
end

function query_from_params(room::GridRoom, path::String; kwargs...)

    _lm = Dict{Symbol, Any}(
        :attention => ex_attention,
        :obstacles => ex_obstacles,
        :granularity => ex_granularity,
        :path => ex_path,
        :img => ex_img,
    )
    latent_map = LatentMap(_lm)

    gm_params = QuadTreeModel(room
                              ;read_json(path)...,
                              kwargs...)

    obs = create_obs(gm_params, room)
    # add_init_constraints!(obs, gm_params, room)

    # Rooms.viz_room(room)

    # compiling further observations for the model
    query = Gen_Compose.StaticQuery(latent_map,
                                    qt_model,
                                    (gm_params,),
                                    obs)
end
