export split_merge_move, balanced_split_merge

import Base.isapprox

include("split_merge_gen.jl")

function can_split(trace::Gen.Trace, node)
    qt = get_retval(trace)
    agg_node = traverse_qt(qt, node)
    prod_node = agg_node.node
    w = weight(agg_node) # NOTE: -Inf in prop if w = 0
    prod_node.max_level > prod_node.level && w > 1E-3
end

function construct_translator(::MoveDirection, node::Int64)
    error("not implemented")
end
function construct_translator(::Split, node::Int64)
    SymmetricTraceTranslator(qt_split_merge_proposal,
                             (node,),
                             qt_involution)
end
function construct_translator(::Merge, node::Int64)
    SymmetricTraceTranslator(qt_split_merge_proposal,
                             (Gen.get_parent(node, 4),),
                             qt_involution)
end

# for trace translator round trip
function Base.isapprox(a::PyObject,
                       b::PyObject; kwargs...)
    x::Int64 = py"($a == $b).sum()"
    n::Int64 = py"($a.size)"
    x == n
end

function split_merge_move(trace::Gen.Trace,
                          node::Int64,
                          direction::MoveDirection)
    # RJ-mcmc move over tracker resolution
    @debug "SM kernel - $node"
    translator = construct_translator(direction, node)
    (new_trace, w1) = translator(trace, check = false)
    # (new_trace, w1) = mytransform(translator, trace, check = false)
    if isinf(w1)
        @show get_depth(node)

        compare_latents(trace, new_trace, direction, node)

        if direction == split_move
            @show new_trace[:trackers => (node, Val(:production)) => :produce]
            for i = 1:4
                child = Gen.get_child(node, i, 4)
                @show new_trace[:trackers => (child, Val(:production)) => :produce]
                @show new_trace[:trackers => (child, Val(:aggregation)) => :mu]
            end
        end
        error("-Inf in $(direction) move on node $(node)")
    end
    (new_trace, w1)
end


function split_merge_move(trace::Gen.Trace,
                          node::Int64)
    @debug "SM kernel - $node"
    translator =
        SymmetricTraceTranslator(qt_branch_proposal,
                                 (node,),
                                 qt_involution_incremental)
    # passes =)  09/20/24
    (new_trace, w1) = translator(trace, check = false)
    # @show w1
    if isinf(w1)
        error("-Inf in $(direction) move on node $(node)")
    end
    (new_trace, w1)
end

function balanced_split_merge(t::Gen.Trace, tidx::Int64)::Bool
    qt = get_retval(t)
    st = traverse_qt(qt, tidx)
    # it's possible to not reach the node
    # in that case, not balanced?
    # REVIEW
    st.node.tree_idx === tidx || return false

    # Root node cannot merge
    st.node.tree_idx === 1 && return false

    # cannot split or merge if max depth
    st.node.level == st.node.max_level && return false
    # balanced if node is terminal : Split <-> Merge
    # and if siblings are all terminal : Merge <-> Split
    parent_idx = Gen.get_parent(tidx, 4)
    parent_st = traverse_qt(qt, parent_idx)
    siblings = parent_st.children
    all(x -> isempty(x.children), siblings) &&
        all(x -> weight(x) > 1E-3, siblings)
end

function compare_latents(a::Gen.Trace, b::Gen.Trace,
                         move::Split, node::Int64)
    addr = :trackers => (node, Val(:aggregation)) => :mu
    va = a[addr]
    print("Split from $(va) -> ")
    siblings = Vector{Float64}(undef, 4)
    for i = 1:4
        cid = Gen.get_child(node, i, 4)
        caddr = :trackers => (cid, Val(:aggregation)) => :mu
        vb = b[caddr]
        print("$(vb) ")
    end
    print("\n")
    return nothing
end

function compare_latents(a::Gen.Trace, b::Gen.Trace,
                         move::Merge, node::Int64)

    siblings = Vector{Float64}(undef, 4)
    # merge to parent, averaging siblings relevance
    parent = Gen.get_parent(node, 4)
    for i = 1:4
        sid = Gen.get_child(parent, i, 4)
        saddr = :trackers => (sid, Val(:aggregation)) => :mu
        siblings[i] = a[saddr]
    end
    baddr = :trackers => (parent, Val(:aggregation)) => :mu
    bv = b[baddr]

    println("Merge from $(siblings) -> $(bv)")

    return nothing
end
