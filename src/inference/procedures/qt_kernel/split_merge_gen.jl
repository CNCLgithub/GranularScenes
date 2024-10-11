@gen function split_step(i::Int64,
                         temp::Float64,
                         n::Int64)
    hi::Float64 = min(1.0, temp)
    lo::Float64 = max(0.0, temp - (n - i))
    u_i::Float64 = {:mu} ~ uniform(lo, hi)
    res::Float64 = temp - u_i
    return res
end

@gen function split_kernel(mu::Float64)
    n = 4
    temp::Float64 = mu * n
    mus = {:steps} ~ Gen.Unfold(split_step)(3, temp, n)
    return mus
end

@gen function qt_split_merge_proposal(t::Gen.Trace, i::Int64)
    qt::QuadTree = get_retval(t)
    st::QTAggNode = traverse_qt(qt, i)
    # `st` could be parent if t' is a result of merge
    # since the original `i` would have been merged with its
    # sibling in t
    ref_idx = st.node.tree_idx
    # println("target: $(i), actual; $(ref_idx)")
    @assert ref_idx == i || ref_idx == Gen.get_parent(i, 4)

    # assuming that `i` is referencing a "balanced" node
    split = isempty(st.children)
    # splitting
    if split
        # refer to tree_idx since st could
        # be from parent in "merge" backward (split)
        mu = t[:trackers => (ref_idx, Val(:aggregation)) => :mu]
        {:split_kernel} ~ split_kernel(mu)
    end
    return split
end


@transform qt_involution (t, u) to (t_prime, u_prime) begin

    # Retrieve node and move type
    _, node = get_args(u)
    split = @read(u[], :discrete)

    if split
        # splitting node
        # update `t_prime` with samples from `u`
        # no backward choices (`u_prime`) as merge is deterministic
        @write(t_prime[:trackers => (node, Val(:production)) => :produce],
               true, :discrete)
        mu = @read(t[:trackers => (node, Val(:aggregation)) => :mu], :continuous)
        dof = 0.0
        # assigning to first 3 children
        for i = 1:3
            c_mu = @read(u[:split_kernel => :steps => i => :mu], :continuous)
            dof += c_mu
            cid = Gen.get_child(node, i, 4)
            @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
                   c_mu, :continuous)
            @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                   false, :discrete)
        end
        # computing residual for 4th child
        res = 4 * mu - dof
        cid = Gen.get_child(node, 4, 4)
        @write(t_prime[:trackers => (cid, Val(:aggregation)) => :mu],
               res, :continuous)
        @write(t_prime[:trackers => (cid, Val(:production)) => :produce],
                false, :discrete)

    else
        # Merge all children of `node`
        # update `t_prime` with the average of children in `t`
        # backward `u_prime` contains the original children in `t`
        mu = 0
        for i = 1:3
            cid = get_child(node, i, 4)
            cmu =  @read(t[:trackers => (cid, Val(:aggregation)) => :mu],
                         :continuous)
            mu += cmu
            @write(u_prime[:split_kernel => :steps => i => :mu],
                cmu, :continuous)
        end
        cid = get_child(node, 4, 4)
        cmu =  @read(t[:trackers => (cid, Val(:aggregation)) => :mu],
                        :continuous)
        mu += cmu
        mu *= 0.25
        @write(t_prime[:trackers => (node, Val(:aggregation)) => :mu],
               mu, :continuous)
        @write(t_prime[:trackers => (node, Val(:production)) => :produce],
               false, :discrete)
    end
end

is_involution!(qt_involution)


@gen function qt_branch_proposal(t::Gen.Trace, i::Int64)
    qt::QuadTree = first(get_retval(t))
    st::QTAggNode = traverse_qt(qt, i)
    # `st` could be parent if t' is a result of merge
    # since the original `i` would have been merged with its
    # sibling in t
    ref_idx = st.node.tree_idx
    parent_idx = Gen.get_parent(i, 4)
    isparent = ref_idx == parent_idx
    # println("target: $(i); actual: $(ref_idx); parent: $(parent_idx)")
    @assert ref_idx == i || isparent

    parent = isparent ? st : traverse_qt(qt, parent_idx)
    merge_weight = isparent ? 1.0 : dof(parent)

    select_parent ~ bernoulli(merge_weight)

    start_node = node(select_parent ? parent : st)
    start_idx = tree_idx(start_node)
    new_branch ~ quad_tree_prior(start_node, start_idx)
    return (start_idx,)
end


function qt_involution_incremental(trace, fwd_choices::ChoiceMap, fwd_ret::Tuple, proposal_args::Tuple)

    (subtree_idx, _...) = fwd_ret
    model_args = get_args(trace)

    # populate constraints
    constraints = choicemap()
    set_submap!(constraints, :trackers, get_submap(fwd_choices, :new_branch))

    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(trace, model_args, (NoChange(),), constraints)

    # populate backward assignment
    bwd_choices = choicemap()
    bwd_choices[:select_parent] = fwd_choices[:select_parent]
    set_submap!(bwd_choices, :new_branch,
                get_submap(discard, :trackers))
    (new_trace, bwd_choices, weight)
end
