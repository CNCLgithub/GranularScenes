using Gen
using Parameters
using GranularScenes
using GranularScenes: rw_move, apply_random_walk
import Gen: get_child, get_parent

@gen function proposal(t::Gen.Trace, i::Int64)
    qt_addr = (i, Val(:aggregation)) => :mu
    mu::Float64 = t[qt_addr]
    dmu = 0.2
    {qt_addr} ~ uniform(max(0., mu - dmu), min(1., mu + dmu))
    return nothing
end

function mytest()


    # testing involution on prior
    center = zeros(2)
    dims = [1., 1.]
    max_level = 5
    start_node = QTProdNode(center, dims, 1, max_level, 1)
    display(start_node)

    cm = choicemap()
    # root node has children
    cm[(1, Val(:production)) => :produce] = true
    for i = 1:4
        # only one child of root has children
        cm[(i+1, Val(:production)) => :produce] = i == 1
        # child of 2 should not reproduce
        cm[(Gen.get_child(2, i, 4), Val(:production)) => :produce] = false
    end
    (trace, ls) = Gen.generate(quad_tree_prior, (start_node, 1), cm)
    choices = get_choices(trace)
    node = 4
    node_addr = (node, Val(:aggregation)) => :mu
    @time (new_trace, w) = apply_random_walk(trace, proposal, (node,))
    @show w
    return nothing
end

mytest();
