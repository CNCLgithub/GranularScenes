
function select_node_uniform(trace::QTVisionTrace)
    schema, _... = get_args(trace)
    n = nleaves(schema)
    i = uniform_discrete(1, n)
    select(:qt => :weights => i => :mean,
           :qt => :weights => i => :w)
end
