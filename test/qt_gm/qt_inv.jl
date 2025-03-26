using Gen
using Parameters
using FunctionalScenes
import Gen: get_child, get_parent



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

    # 1 -> 2 | [3,4,5] -> 6 | [7,8,9]
    node = 6 # first child of node 2
    translator = SymmetricTraceTranslator(qt_split_merge_proposal,
                                          (node,),
                                          # qt_sm_inv_manual)
                                          qt_involution)
    # @time (new_trace, w) = translator(trace, check = true)
    @time (new_trace, w) = translator(trace, check = false)

    # display(get_choices(new_trace))

    @show w
    return nothing
end

mytest();
