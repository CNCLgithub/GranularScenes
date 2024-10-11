export rw_move

@gen function qt_node_random_walk(t::Gen.Trace, addr::Pair)
    mu::Float64 = t[addr]
    low::Float64 = max(0., mu - 0.2)
    high::Float64 = min(1., mu + 0.2)
    # @show (low, high)
    {addr} ~ uniform(low, high)
end

function rw_move(t::Gen.Trace, i::Int64)
    qt = first(get_retval(t))
    qt_addr = :trackers => (i, Val(:aggregation)) => :mu
    (t, w1) = apply_random_walk(t, qt_node_random_walk, (qt_addr,))
    leaf_vec_idx = qt.mapping[i]
    loc_addr = :loc_prior => leaf_vec_idx => :b
    (t, w2) = apply_random_walk(t, qt_node_random_walk, (loc_addr,))
    (t, w1 + w2)
end

function rw_move(::NoChange, t::Gen.Trace, i::Int64)
    rw_move(t, i)
end

function rw_move(::Split, tr::Gen.Trace, node::Int64)
    result = 0.0
    for i = 1:4
        idx = Gen.get_child(node, i , 4)
        tr, w = rw_move(tr, idx)
        result += w
    end
    (tr, result)
end

function rw_move(::Merge, tr::Gen.Trace, node::Int64)
    rw_move(tr, Gen.get_parent(node, 4))
end

function compare_latents(a::Gen.Trace, b::Gen.Trace, node)
    addr = :trackers => (node, Val(:aggregation)) => :mu
    va = a[addr]
    vb = b[addr]
    println("RW from $(va) -> $(vb)")
    return nothing
end
