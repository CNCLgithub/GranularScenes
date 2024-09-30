using Accessors: setproperties

export plan,
    integrate_update,
    get_plan

function plan(qt::QuadTree, occ_cost::Float64,
              ent::Int64, ext::Int64,
              maxsteps::Int=20)


    @unpack root, leaves, mapping = qt
    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = ent_node.node.tree_idx

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = ext_node.node.tree_idx

    # @show ext_idx
    heuristic = n -> -log(dist(node(n), node(ext_node)) + 1E-5)
    root = AStarNode(ext_idx,
                     heuristic,
                     qt,
                     occ_cost,
                     ent_idx,
                     0,
                     1, maxsteps)
    horizon = simulate(astar_recurse, (root, 1))
end

function integrate_update(plan_tr::Gen.RecurseTrace,
                          qt::QuadTree)

    # adjust horizon size
    d = depth(plan_tr)
    # given new perceived state, determine how well the
    # current sequence of action works out
    new_tr, nw = splice_plan(plan_tr, 1, d, (;qt=qt))

    # compute probability of the current sequence of actions
    # for the prevous state
    prev_w::Float64 = 0.0
    k = depth(new_tr)
    for t = 1:k
        # @show t
        prev_w += Gen.project(plan_tr.production_traces[t],
                              select(:action))
    end
    nw - prev_w
end

function splice_plan(source::Gen.RecurseTrace,
                     origin::Int64, # where to cut the new trace
                     steps::Int64,
                     node_data::NamedTuple)

    n = depth(source)
    maxsteps = origin + steps

    # extract the new start node
    node = get_node(source, origin) # REVIEW
    node = setproperties(node;
                         node_data...,
                         maxsteps = maxsteps)
    # copy over previously planned choices
    cm = choicemap()
    for i = origin:(min(n, maxsteps))
        addr = (i, Val(:production)) => :action
        cm[addr] = source[addr]
    end
    args = (node, origin)
    new_tr, w = Gen.generate(astar_recurse, args, cm)
end

#################################################################################
# Horizon interface
#################################################################################

function depth(tr::Gen.RecurseTrace)
    ks = keys(tr.production_traces)
    maximum(ks)
end

function get_step(tr::Gen.RecurseTrace, i::Int64)
    subtrace = tr.production_traces[i]
    get_retval(subtrace).value
end
function get_node(tr::Gen.RecurseTrace, i::Int64)
    subtrace = tr.production_traces[i]
    last(get_args(subtrace))
end

function horizon_length(tr::Gen.RecurseTrace, t::Int64)
    # ks = keys(tr.production_traces)
    # @show ks
    depth(tr) - t
end

function plan_reward(tr::Gen.RecurseTrace)
    get_retval(tr)
end

function get_plan(tr::Gen.RecurseTrace)
    root, _ = get_args(tr)
    d = depth(tr)
    steps = Vector{Int64}(undef, d + 1)
    for i = 1:d
        steps[i] = get_node(tr, i).loc
    end
    steps[end] = first(get_step(tr, d))
    return steps
end

#################################################################################
# AStarRecurse
#################################################################################

struct AStarNode
    dest::Int64
    "(QTAggNode) -> value"
    heuristic::Function
    qt::QuadTree
    obs_cost::Float64
    "Current node location"
    loc::Int64
    prev::Int64
    step::Int64
    maxsteps::Int64
end

function AStarNode(n::AStarNode, next::Int64)
    setproperties(n;
                  loc = next,
                  prev = n.loc,
                  step = n.step + 1)
end

@gen (static) function astar_production(n::AStarNode)
    # look ahead one step
    next_states, rewards, weights =  explore(n)
    action = @trace(categorical(weights), :action)
    reward = rewards[action]
    # predicted state associated with action
    next_state = next_states[action]
    node = AStarNode(n, next_state)
    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    # if `next_state` fails, the trace will terminate
    # and go back to the planner to regenerate a new branch
    w = production_weight(node, reward)
    s = @trace(bernoulli(w), :produce) # REVIEW: make deterministic?
    children::Vector{AStarNode} = s ? [node] : AStarNode[]
    result::Production{Tuple{Int64, Float64}, AStarNode} =
        Production((next_state, reward), children)
    return result
end

@gen static function astar_aggregation(agg_args::Tuple{Int64, Float64},
                                       children::Vector{Float64})
    r = last(agg_args)
    reward::Float64 = isempty(children) ?
        r :
        logsumexp(r,  first(children))
    return reward
end

function production_weight(n::AStarNode, r::Float64)
    (n.loc == n.dest || n.step >= n.maxsteps) ? 0.0 : 1.0
end

function explore(n::AStarNode)
    if n.loc == 1
        # no navigation possible
        return ([1], [0.], [1.])
    end
    # @show n.loc => n.dest
    cur_state = leaf_from_idx(n.qt, n.loc)
    prev_state = n.prev === 0 ? cur_state : leaf_from_idx(n.qt, n.prev)
    adjs = adjacent_leaves(n.qt, n.loc)
    nn = length(adjs)
    values = Vector{Float64}(undef, nn)
    costs = Vector{Float64}(undef, nn)
    weights = Vector{Float64}(undef, nn)
    @inbounds for (i, idx) = enumerate(adjs)
        state = leaf_from_idx(n.qt, idx)
        if node(state).tree_idx == n.prev
            values[i] = -Inf
            costs[i] = 1.0
            # rewards[i] = -Inf
            continue
        end
        v = n.heuristic(state)
        # v += 1.0 * dist(node(prev_state), node(state))
        values[i] = v
        # d = dist(cur_state.node, state.node)
        c = (weight(state) * length(state.node) +
            weight(cur_state) * length(cur_state.node))
        c *= n.obs_cost
        costs[i] = c
    end
    rewards = values .- costs
    # @show adjs
    # @show values
    # @show costs
    # @show rewards
    softmax!(weights, rewards, 0.1)
    # @show weights
    return (adjs, rewards, weights)
end

const astar_recurse = Recurse(astar_production,
                              astar_aggregation,
                              1, # max children
                              AStarNode,# U (production to children)
                              Tuple{Int64, Float64},# V (production to aggregation)
                              Float64) # W (aggregation to parents)
