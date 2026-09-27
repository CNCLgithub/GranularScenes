@with_kw struct MO <: GranularityProtocol
    
end

mutable struct MOState <: MentalState{MO}
    tick::Int
    schema::QTSchema
    time_integral::Dict{NodeId, Float64}
end

function step_module!(mo::MentalModule{MO},
                      attention::MentalModule{AdaptiveComputation},
                      vision::MentalModule{AdaptiveMH},
                      planning::MentalModule{AStarPlanner})

    mo_prot, mo_state = mparse(mo)
    
    # get the schema - common to all traces
    schema = mo_state.schema
    task_rel = deepcopy(mo_state.time_integral)

    split_p = split_prob(schema, task_rel)

    move = rand() < split_p ?
        split_kernel(schema, task_rel) :
        merge_kernel(schema, task_rel)

    apply_split_merge_move!(move, mo)
    apply_split_merge_move!(move, attention)
    apply_split_merge_move!(move, vision)
    apply_split_merge_move!(move, planning)
    return nothing
end

function split_prob(g::QTSchema, tr::Vector{Float64})
    # out of memory
    nleaves(g) < MAX_LEAVES || return 0.0

    # Only one leaf, can't merge
    nleaves(g) == 1 || return 1.0

    # 50/50 split-merge
    0.5
end

abstract type Reframe end

struct SplitMove <: Reframe
    node::NodeId
end

struct MergeMove <: Reframe
    node::NodeId
end

function split_kernel(g::QTSchema, tr::Vector{Float64})
    # REVIEW: mutating `tr` for speed, should communicate
    n = length(tr)
    @inbounds for i = 1:n
        node = g.leaves[i]
        if Int64(depth(node)) == g.max_level
            tr[i] = -Inf
        end
    end
    softmax!(tr)
    idx = categorical(tr)
    SplitMove(g.leaves[idx])
end

function merge_kernel(g::QTSchema, tr::Vector{Float64})
    # REVIEW: mutating `tr` for speed, should communicate
    n = length(tr)
    @inbounds for i = 1:n
        tr[i] *= -1
    end
    softmax!(tr)
    idx = categorical(tr)
    parent = parent_key(g.leaves[idx])
    MergeMove(parent)
end

function apply_split_merge_move!(move::SplitMove,
                                 mo::MentalModule{<:MO})
    node = move.node

    mo_prot, mo_state = mparse(mo)
    
    haskey(mo_state.time_integral, node) ||
        throw(ArgumentError("not a leaf: $node"))
    node.depth < max_level(mo_state.schema) ||
        throw(ArgumentError("leaf at max level: $node"))

    w = mo_state.time_integral[node]
    kids = [child_key(node, j) for j in 1:4]
    for k in kids
        mo_state.time_integral[k] = w
    end
    delete!(mo_state.time_integral, node)
    keep = filter(!=(node), mo_state.schema.leaves)
    append!(keep, kids)
    mo_state.schema.leaves = keep
    return nothing
end

function apply_move(move::SplitMove,
                    qt::QuadTree)
    n = move.node
    haskey(qt.weight_map, n) || throw(ArgumentError("not a leaf: $n"))
    n.depth < max_level(qt) || throw(ArgumentError("leaf at max level: $n"))
    
    w = qt.weight_map[n]
    new_weight_map = Dict{NodeId, Float64}()
    # add children
    for ki in 1:4
        k = child_key(n, ki)
        new_weight_map[k] = w
    end
    #copy over other nodes
    for (k, v) = qt.weight_map
        k == n && continue
        new_weight_map[k] = v
    end

    new_leaves = collect(NodeId, keys(new_weight_map))
    sort!(new_leaves)

    new_schema = QTSchema(max_level(qt),
                          bounds(qt),
                          new_leaves)

    QuadTree(new_schema, new_weight_map)
end

function apply_move(move::MergeMove,
                    qt::QuadTree)
    parent = move.node
    kids = [child_key(parent, j) for j in 1:4]
    all(k -> haskey(qt.weight_map, k), kids) ||
        throw(ArgumentError("Parent missing child"))

    pooled = 0.0
    for k in kids
        pooled += qt.weight_map[k]
    end
    pooled *= 0.25

    new_weight_map = Dict{NodeId, Float64}()
    # add parent
    new_weight_map[parent] = pooled
    #copy over other nodes
    for (k, v) = qt.weight_map
        in(k, kids) && continue
        new_weight_map[k] = v
    end

    new_leaves = collect(NodeId, keys(new_weight_map))
    sort!(new_leaves)

    new_schema = QTSchema(max_level(qt),
                          bounds(qt),
                          new_leaves)

    QuadTree(new_schema, new_weight_map)
end

function apply_move(move::Reframe,
                    trace::QTVisionTrace)

    _, rest_args... = get_args(trace)
    prev_qt = get_retval(trace)

    new_qt = apply_move(move, prev_qt)
    new_args = (new_qt.schema, rest_args...)


    lvs = leaves(new_qt)
    cm = choicemap()
    for i = 1:nleaves(new_qt)
        node = lvs[i]
        w = weight_of(new_qt, node)
        cm[:qt => :weights => i => :mean] = w
        cm[:qt => :weights => i => :w] = w
    end

    cm[:depth] = trace[:depth]


    trace, w = Gen.generate(qt_vision, new_args, cm)
end

function apply_split_merge_move!(move::Reframe,
                                 vision::MentalModule{<:AdaptiveMH})
    node = move.node
    vprot, vstate = mparse(vision)
    # Re-init all traces with:
    # - new schema
    # - transferred weights
    ns = length(vstate.samples)
    @inbounds for i = 1:ns
        trace = vstate.samples[i]
        vstate.samples[i], vstate.weights[i] =
            apply_move(move, trace)
    end
    return nothing
end

end
    apply_split_merge_move!(move, attention)
    apply_split_merge_move!(move, vision)
    apply_split_merge_move!(move, planning)
