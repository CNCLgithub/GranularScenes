

function plan(qt::QuadTree, occ_cost::Float64=1.0,
              maxsteps::Int=20)
    @unpack root, leaves, mapping = qt
    length(leaves) == 1 && return QTPath(first(leaves))
    # adjacency, distance matrix
    ad, _ = nav_graph(leaves, 1.0)

    g = SimpleGraph(ad)

    # map entrance and exit in room to qt
    row_d::Int64 = max_leaves(qt.root.node)
    ent_point = idx_to_node_space(ent, row_d)
    ent_node = traverse_qt(root, ent_point)
    ent_idx = mapping[ent_node.node.tree_idx]

    ext_point = idx_to_node_space(ext, row_d)
    ext_node = traverse_qt(root, ext_point)
    ext_idx = mapping[ext_node.node.tree_idx]

    # L2 heuristic used in A*
    heuristic = a_star_heuristic(leaves, ext_node, 1.0)

    root = AStarNode(heuristic,
                     qt,
                     g,
                     ent_idx,
                     0, max_steps)

    horizon = simulate(astar_recurse, (root, 0))
end


function replan(wm::W, ws::WorldState{<:W},
                heuristic,
                t::Int64,
                max_steps::Int64
                ) where {W <: VGDLWorldModel}
    # initialize world state at current time
    gs = game_state(ws)
    player = gs.scene.dynamic[1]
    args = (0, ws, wm)
    tr, _ = Gen.generate(vgdl_wm, args)
    # planning node points to current world
    # state and heuristic
    start_node = AStarNode(heuristic,
                           length(actionspace(player)),
                           tr,
                           t,
                           t + max_steps,
                           )

    args = (start_node, t)
    ptr, _ = Gen.generate(astar_recurse, args)
    ptr
end


function splice_plan(source::Gen.Trace,
                     origin::Int64, # where to cut the new trace
                     steps::Int64,
                     node_data::NamedTuple = NamedTuple())

    n = depth(source)
    maxsteps = origin + steps

    # @show origin
    # @show maxsteps
    # @show keys(source.production_traces)
    # extract the new start node
    node = get_node(source, origin) # REVIEW
    node = setproperties(node;
                         node_data...,
                         maxsteps = maxsteps)
    # @show node.step
    # @show first(get_args(node.state))

    # copy over previously planned choices
    cm = choicemap()
    for i = origin:(min(n, maxsteps))
        addr = (i, Val(:production)) => :action
        cm[addr] = source[addr]
    end

    args = (node, origin)
    new_tr, w = Gen.generate(astar_recurse, args, cm)
    new_node = get_node(new_tr, origin)
    # @show new_node.step
    # @show first(get_args(new_node.state))
    new_tr, w
end

function shift_plan(plan_tr::Gen.Trace,
                    current::Int64,
                    steps::Int64)
    # println("shift plan")
    n = depth(plan_tr)
    origin = min(current, n)
    # node_data = (; step = origin)
    splice_plan(plan_tr, origin, steps) #, node_data)
end

function integrate_update(plan_tr::Gen.Trace,
                          state::Gen.Trace,
                          current::Int64,
                          maxsteps::Int64)

    # println("integrate update")
    # update with new state
    node_data = (; state = state)
    # adjust horizon size
    n = depth(plan_tr)
    k = min(n - current, maxsteps)
    # given new perceived state, determine how well the
    # current sequence of action works out
    new_tr, nw = splice_plan(plan_tr, current, k, node_data)

    # compute probability of the current sequence of actions
    # for the prevous state
    # TODO: project is not implemented for `Recurse`
    # prev_w = Gen.project(plan_tr, selection)
    prev_w::Float64 = 0.0
    k = depth(new_tr)
    # @show k
    # @show(keys(new_tr.production_traces))
    for t = current:k
        # @show t
        prev_w += Gen.project(plan_tr.production_traces[t],
                              select(:action))
    end
    # (new_tr, nw - prev_w)
    nw - prev_w
end

#################################################################################
# Horizon interface
#################################################################################

function select_subgoal(horizon::Gen.RecurseTrace)
    final_step = get_retval(horizon)
    @unpack heuristics = final_step
    # display(heuristics)
    n = length(heuristics[1]) # subgoals
    v = fill(-Inf, n)
    @inbounds for a = 1:length(heuristics)
        v = max.(v, heuristics[a])
    end
    sgi = argmax(v)
end

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

function consolidate(sgs::Vector{<:Goal}, agraph)
    GoalGradients(sgs, agraph)
end

function horizon_length(tr::Gen.RecurseTrace, t::Int64)
    # ks = keys(tr.production_traces)
    # @show ks
    depth(tr) - t
end


function planned_action(pl::TheoryBasedPlanner, t::Int64)
    tr = pl.horizon
    step = get_step(tr, t)
    # @show step.heuristics
    step.action
end

function get_heursitic(tr::Gen.RecurseTrace)
    snode, _ = get_args(tr)
    snode.heuristic # `GoalGradient` function
end

function get_gradient_values(tr::Gen.RecurseTrace, t::Int64)
    step = get_step(tr, t)
    step.heuristics # gradient values
end

function map_gradients!(attention,
                        horizon,
                        # pl::TheoryBasedPlanner{<:W},
                        current_t::Int64
                        ) #where {W<:VGDLWorldModel}
    # @unpack horizon = pl
    # get subgoals + gradients for time t
    hstep = get_step(horizon, current_t)
    action = hstep.action
    values = hstep.heuristics[action]
    goal_gradients = get_heursitic(horizon)
    sub_goals = goal_gradients.subgoals
    wm = WorldMap(hstep.node.state, goal_gradients.affordances)
    n = length(sub_goals)
    # project each subgoal's gradient to world map
    for i = 1:n
        sg = sub_goals[i]
        idx = project_ref(reference(sg), wm)
        v = values[i]
        write_delta_pi!(attention, idx, v)
    end

    return nothing
end

#################################################################################
# AStarRecurse
#################################################################################

struct AStarNode
    "(state) -> value"
    heuristic::Function
    qt::QuadTree
    graph::SimpleGraph
    "Current node location"
    loc::Int64
    step::Int64
    maxsteps::Int64
end

@gen (static) function astar_production(n::AStarNode)
    # look ahead one step
    next_states, rewards, weights =  explore(n)
    action = @trace(categorical(weights), :action)
    reward = rewards[action]
    # predicted state associated with action
    node = AStarNode(n, next_states[action])
    # determine if new state satisfies any goals
    # or if the planning budget is exhausted
    # if `next_state` fails, the trace will terminate
    # and go back to the planner to regenerate a new branch
    w = production_weight(node, reward)
    s = @trace(bernoulli(w), :produce) # REVIEW: make deterministic?
    children::Vector{AStarNode} = s ? [node] : AStarNode[]
    result::Production{Float64, AStarNode} =
        Production(reward, children)
    return result
end

@gen static function astar_aggregation(r::Float64,
                                       children::Vector{Float64})
    reward::Float64 = isempty(children) ?
        r :
        logsumexp(r,  first(children))
    return reward
end

function production_weight(n::AStarNode, r::Float64)
    (n.step >= n.maxsteps || iszero(r)) ? 0.0 : 1.0
end

function explore(n::AStarNode)
    cur_state = leaf_from_idx(n.qt, n.loc)
    # cv = n.heuristic(cur_state)
    next_states = neighbors(n.graph, n.loc)
    nn = length(next_states)
    rewards = Vector{Float64}(undef, nn)
    weights = Vector{Float64}(undef, nn)
    @inbounds for (i, state) = enumerate(next_states)
        v = n.heuristic(state)
        d = contact(cur_state.node, state.node) ?
            dist(cur_state.node, state.node) : Inf
        c = d * (weight(state) * length(state.node) +
            weight(cur_state) * length(cur_state))
        rewards[i] = v / d
    end
    softmax!(weights, rewards)
    return (next_states, rewards, weights)
end

const astar_recurse = Recurse(astar_production,
                              astar_aggregation,
                              1, # max children
                              AStarNode,# U (production to children)
                              Float64,# V (production to aggregation)
                              Float64) # W (aggregation to parents)

#################################################################################
# Visualization
#################################################################################

function get_horizon_state(pl::TheoryBasedPlanner,
                           horizon::Gen.RecurseTrace,
                           t::Int64)
    world_trace = get_step(horizon, t).node.state
    world_state(world_trace)
end

function render_horizon(pl::TheoryBasedPlanner)
    horizon = pl.horizon
    # display(get_choices(horizon))
    steps = collect(keys(horizon.production_traces))
    states = map(t -> get_horizon_state(pl, pl.horizon, t),
                 steps)

    agent = get_player(pl.world_model, first(states))
    gr = graphics(pl.world_model)
    mean(map(st -> render(gr, st), states))
end

function viz_planning_module(agent::GenAgent{W,V,P,M,A},
                             path::String="") where {P<:TheoryBasedPlanner,
                                                     W, V, M, A}
    img = render_horizon(agent.planning)
    if path != ""
        save(path, repeat(render_obs(img), inner = (10,10)))
    end
    viz_obs(img)
    return nothing
end
