export DataDrivenState,
    generate_cm_from_ddp

using Statistics: std

@with_kw struct DataDrivenState
    # neural network details
    config_path::String
    device::PyObject = _load_device()
    nn::PyObject = _init_dd_state(config_path, device)

    # proposal variables
    var::Float64 = 0.05

end


function _init_dd_state(config_path::String, device::PyObject)
    nn = @pycall vox.init_dd_state(config_path, device)::PyObject
    nn.to(device)
    return nn
end


# `timg`: taichi field
function process_ddp_input(timg::PyObject, device::PyObject)
    x = @pycall timg.to_numpy()::Array{Float32, 3}
    T = eltype(x)
    clamp!(x, zero(T), one(T))
    x = permutedims(x, (3,2,1))
    y = @pycall torch.tensor(x, device = device)::PyObject
    y = @pycall y.unsqueeze(0)::PyObject
    return y
end

function generate_cm_from_ddp(ddp_params::DataDrivenState,
                              timg, model_params,
                              min_depth::Int64 = 1,
                              max_depth::Int64 = 5)
    @unpack nn, device, var = ddp_params

    img = process_ddp_input(timg, device)
    x = @pycall nn.forward(img)::PyObject
    state = @pycall x.detach().squeeze(0).cpu().numpy()::Matrix{Float64}
    println("Data-driven state")
    display_mat(state)
    head = model_params.start_node
    d = model_params.dims[2]
    # Iterate through QT
    cm = choicemap()
    queue = [model_params.start_node]
    nodes = 0
    while !isempty(queue)
        head = pop!(queue)
        idx = node_to_idx(head, d)
        mu = mean(state[idx])
        sd = length(idx) == 1 ? 0. : std(state[idx], mean = mu)
        # @show sd
        # split = sd > ddp_params.var && head.level < head.max_level
        # restricting depth of nn
        split = head.level < min_depth || (sd > ddp_params.var && head.level < max_depth)
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
    # for i = 1:nodes
    #     cm[:loc_prior => i => :b] = 0.0
    # end
    println("DDP yielded $nodes qt leaves")
    return cm
end
