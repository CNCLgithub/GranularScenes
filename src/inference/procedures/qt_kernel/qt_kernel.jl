import Gen.regenerate
import Base.show

function Gen.regenerate(trace::Gen.RecurseTrace{S,T,U,V,W,X,Y},
                    new_args::Tuple{U,Int},
                    argdiffs::Tuple,
                    selection::EmptySelection) where {S,T,U,V,W,X,Y}
    (trace, 0.0, Gen.NoChange())
end


function apply_random_walk(trace::Gen.Trace, proposal, proposal_args)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = Gen.assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    (new_trace, weight)
end


"""
Defines possible moves over QT
"""
abstract type MoveDirection end
struct Merge <: MoveDirection end
struct Split <: MoveDirection end
struct NoChange <: MoveDirection end
const merge_move = Merge()
const split_move = Split()
const no_change  = NoChange()


function Base.show(io::IO, ::Merge)
    print(io, "Merge")
end

function Base.show(io::IO, ::Split)
    print(io, "Split")
end


function downstream_selection(t::Gen.Trace, node::Int64)
    params = first(get_args(t))
    head::QTAggNode = t[:trackers]
    st::QTAggNode = traverse_qt(head, node)
    # associated rooms samples
    idxs = node_to_idx(st.node, params.dims[1])
    v = Vector{Pair}(undef, length(idxs) * params.instances)
    lis = LinearIndices((params.instances, length(idxs)))
    for i = 1:params.instances, j = 1:length(idxs)
        v[lis[i, j]] = :instances => i => :obstacle => idxs[j] => :flip
    end
    select(v...)
end

function downstream_selection(::Union{NoChange,Split}, t::Gen.Trace, node::Int64)
    downstream_selection(t, node)
end

function downstream_selection(::Merge, t::Gen.Trace, node::Int64)
    p = Gen.get_parent(node, 4)
    downstream_selection(t, p)
end

function all_downstream_selection(t::Gen.Trace)::Gen.Selection
    p = first(get_args(t))
    all_downstream_selection(p)
end

include("random_walk.jl")
include("split_merge.jl")
