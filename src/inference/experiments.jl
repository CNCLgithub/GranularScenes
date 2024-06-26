export run_inference, resume_inference, query_from_params, proc_from_params

function ex_choicemap(c::AMHChain)
    ex_choicemap(c.state)
end
function ex_choicemap(tr::Gen.Trace)
    s = Gen.complement(select(:viz, :instances))
    choices = get_choices(tr)
    get_selected(choices, s)
end

function ex_projected(c::AMHChain)
     st = get_retval(c.state)
     deepcopy(st.qt.projected)
end

function ex_img_mu(c::AMHChain)
    st = get_retval(c.state)
    deepcopy(st.img_mu)
end

function ex_granularity(c::AMHChain)
    st = get_retval(c.state)
    n = size(st.qt.projected, 1)
    m = Matrix{Int64}(undef, n, n)
    for x in st.qt.leaves
        idx = node_to_idx(x.node, n)
        m[idx] .= x.node.level
    end
    m
end

function ex_path(c::AMHChain)
    params = first(get_args(c.state))
    qt = get_retval(c.state)
    path = quad_tree_path(c.state)
    n = first(params.dims)
    leaves = qt.leaves
    m = fill(false, (n,n))
    for e in path.edges
        src_node = leaves[src(e)].node
        idx = node_to_idx(src_node, n)
        m[idx] .= true
        dst_node = leaves[dst(e)].node
        idx = node_to_idx(src_node, n)
        m[idx] .= true
    end
    m
end

function ex_attention(c::AMHChain)
    @unpack auxillary = c
    Dict(:sensitivities => deepcopy(auxillary.sensitivities),
         :node => deepcopy(auxillary.node))
end

function query_from_params(room::GridRoom, path::String; kwargs...)

    _lm = Dict{Symbol, Any}(
        # :projected => ex_projected,
        # :granularity => ex_granularity,
        # :path => ex_path,
        # :img_mu => ex_img_mu,
        # :attention => ex_attention
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

# function proc_from_params(gt::Room,
#                           model_params::ModelParams, proc_path::String,
#                           vae::String, ddp::String; kwargs...)

#     # init ddp
#     img = image_from_instances([gt], model_params)
#     ddp_params = DataDrivenState(;vae_weights = vae,
#                                  ddp_weights = ddp)
#     all_selection = FunctionalScenes.all_selections_from_model(model_params)
#     ddp_args = ((ddp_params, img), all_selection)

#     selections = FunctionalScenes.selections(model_params)
#     proc = FunctionalScenes.load(AttentionMH, selections, proc_path;
#                                  ddp_args = ddp_args,
#                                  kwargs...)
# end
