
#################################################################################
# Split / merge — O(log n) bookkeeping
#################################################################################


"""
    split!(qt, n) -> nothing

Split leaf `n` into 4 children. Children inherit the parent's weight.
Order in `schema.leaves` is not maintained (not load-bearing).
"""
function split!(qt::QuadTree, n::NodeId)
    haskey(qt.weight_map, n) || throw(ArgumentError("not a leaf: $n"))
    n.depth < max_level(qt) || throw(ArgumentError("leaf at max level: $n"))
    w = qt.weight_map[n]
    kids = [child_key(n, j) for j in 1:4]
    for k in kids
        qt.weight_map[k] = w
    end
    delete!(qt.weight_map, n)
    keep = filter(!=(n), qt.schema.leaves)
    append!(keep, kids)
    qt.schema.leaves = keep
    return nothing
end
split!(qt::QuadTree, i::Int) = split!(qt, qt.schema.leaves[i])

"""
    merge_children!(qt, i) -> Bool

Given the *parent* node key of 4 sibling leaves (as a NodeId at depth d-1),
replace them with the single parent leaf carrying the pooled weight (mean of
the four).
"""
function merge_children!(qt::QuadTree, parent::NodeId)
    kids = [child_key(parent, j) for j in 1:4]
    all(k -> haskey(qt.weight_map, k), kids) || return nothing
    pooled = 0.0
    for k in kids
        pooled += qt.weight_map[k]
        delete!(qt.weight_map, k)
    end
    pooled *= 0.25
    qt.weight_map[parent] = pooled
    keep = filter(∉(kids), qt.schema.leaves)
    append!(keep, (parent,))
    qt.schema.leaves = keep
    return nothing
end
