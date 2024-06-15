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
end

function qt_path_cost(tr::Gen.Trace)::Float64
    qt_path = quad_tree_path(tr)
    c = 0.0
    for e in qt_path.edges
        c += qt_path.dm[src(e), dst(e)]
    end
    return c
end

function delta_s(alpha::Float64)
    # abs(0.5 - exp(clamp(alpha, -Inf, 0)))
    exp(clamp(alpha, -Inf, 0))
end
