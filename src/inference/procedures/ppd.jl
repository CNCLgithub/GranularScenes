export generate_cm_from_ppd

function generate_cm_from_ppd(marginal::Matrix{<:Real},
                              model_params, var::Float64,
                              min_depth = 2)
    println("Posterior Marginal")
    display_mat(marginal)
    head = model_params.start_node
    d = model_params.dims[2]
    max_depth = model_params.max_depth
    # Iterate through QT
    cm = choicemap()
    queue = [model_params.start_node]
    nodes = 0
    while !isempty(queue)
        head = pop!(queue)
        idx = node_to_idx(head, d)
        mu = mean(marginal[idx])
        sd = length(idx) == 1 ? 0. : std(marginal[idx], mean = mu)
        # @show sd
        # split = sd > ddp_params.var && head.level < head.max_level
        # restricting depth of nn
        split = head.level < min_depth || (sd > var && head.level < max_depth)
        cm[:trackers => (head.tree_idx, Val(:production)) => :produce] = split
        if split
            # add children to queue
            append!(queue, produce_qt(head))
        else
            # terminal node, add aggregation choice
            cm[:trackers => (head.tree_idx, Val(:aggregation)) => :mu] = mu
            nodes += 1
        end
    end
    println("PPD yielded $nodes qt leaves")
    return cm
end
