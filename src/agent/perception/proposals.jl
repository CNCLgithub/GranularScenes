
function select_node_uniform(trace::QTVisionTrace)
    schema, _... = get_args(trace)
    n = nleaves(schema)
    i = uniform_discrete(1, n)
    node_ancestral_proposal(trace, i)
end

function node_ancestral_proposal(trace::QTVisionTrace, i::Int)

    selection = select(:qt => :weights => i => :mean,
                       :qt => :weights => i => :w)

    new_trace, w, _... = regenerate(trace, selection)

    (new_trace, w)
end
