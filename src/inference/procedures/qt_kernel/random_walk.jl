export rw_move

@gen function qt_node_random_walk(t::QTTrace, i::Int64)
    phase, _ = get_args(t)
    if phase == 0 
        qt_addr = :trackers => (i, Val(:aggregation)) => :mu
        mu::Float64 = t[qt_addr]
        dmu = 0.2
        {qt_addr} ~ uniform(max(0., mu - dmu), min(1., mu + dmu))
    else
        # qt = get_retval(t)
        # leaf_vec_idx = qt.mapping[i]
        # loc_addr = :changes => 1 => leaf
        # mu = t[loc_addr]
        # dmu = 0.1
        # {loc_addr} ~ uniform(max(0., mu - dmu), min(1., mu + dmu))
    end
    return nothing
end

function rw_move(t::QTTrace, i::Int64)
    (t, w) = apply_random_walk(t, qt_node_random_walk, (i,))
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
