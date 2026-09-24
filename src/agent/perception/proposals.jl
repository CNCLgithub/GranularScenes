
function select_node_uniform(trace::QTVisionTrace)
    schema, _... = get_args(trace)
    n = nleaves(schema)
    i = uniform_discrete(1, n)
    select(:weights => i => :mean,
           :weights => i => :w)
end
