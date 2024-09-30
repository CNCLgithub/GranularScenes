export quad_tree_path,
    qt_path_cost

"""
Given a trace, returns the objective over paths
"""
function quad_tree_path(tr::Gen.Trace)
    params = first(get_args(tr))
    qt = get_retval(tr)
    a = first(params.entrance)
    b = first(params.exit)
    w = params.obs_cost
    qt_a_star(qt, w, a, b)
    # plan(qt, w, a, b)
end

function qt_path_cost(tr::Gen.Trace)::Float64
    # qt_path = quad_tree_path(tr)
    # c = 0.0
    # for e in qt_path.edges
    #     c += qt_path.dm[src(e), dst(e)]
    # end
    # return c
    path_tr = quad_tree_path(tr)
    plan_reward(path_tr)
end

function quad_tree_cross_cost(t_prime::Gen.Trace, path::Vector{Int64})
    params = first(get_args(t_prime))
    qt = get_retval(t_prime)
    src = leaf_from_idx(qt, path[1])
    if length(path) == 1
        return traversal_cost(src, src, params.obs_cost)
    end
    cost = 0.0
    for i = 2:length(path)
        dst = leaf_from_idx(qt, path[i])
        cost += traversal_cost(src, dst, params.obs_cost)
        src = dst
    end
    return cost
end


function delta_pi(plan::Tuple, t_prime::Gen.Trace)
    cost, path = plan
    cost_prime, path_prime = quad_tree_path(t_prime)
    cross_cost = quad_tree_cross_cost(t_prime, path)
    total = log(abs(cost_prime - cost) + abs(cost - cross_cost))
    # @show cost
    # @show cost_prime
    # @show cross_cost
    # @show total
    return total
end

function delta_pi(plan_tr::Gen.Trace, t_prime::Gen.Trace)
    qt = get_retval(t_prime)
    log_ratio = integrate_update(plan_tr, qt)
    abs(log_ratio) # farther from ratio of 1.0
end

function delta_s(alpha::Float64)
    exp(clamp(alpha, -Inf, 0))
end
