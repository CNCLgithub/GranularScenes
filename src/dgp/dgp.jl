export DGP, GrowState, valid_spaces

"""A data generating procedure"""
abstract type DGP end

@with_kw struct FurnishState <: DGP
    template::GridRoom
    vm::PersistentVector{Bool}
    f::Set{Int64}
    count::Int64
    max_size::Int64 = 5
    max_count::Int64 = 10
end

function FurnishState(st::FurnishState, f::Set{Int64})
    _vm = collect(Bool, st.vm)
    _vm[f] .= false
    purge_around!(_vm, st.template, f)
    FurnishState(st.template,
                 PersistentVector(_vm),
                 f,
                 st.count + 1,
                 st.max_size,
                 st.max_count)
end


@with_kw struct GrowState <: DGP
    head::Int64
    vm::PersistentVector{Bool}
    g::Rooms.PathGraph
    current_depth::Int64
    max_depth::Int64 = 5
end

# function GrowState(head::Int64, vmap::BitMatrix,
#                    g::PathGraph)
#     GrowState(head, P)

function GrowState(ns, ni::Int64, st::GrowState)::GrowState
    @unpack vm, g, current_depth = st
    # done growing
    ni == 0 && return st
    # update head
    new_head = ns[ni]
    new_vm = assoc(vm, new_head, false)
    GrowState(new_head, new_vm, g, current_depth + 1,
              st.max_depth)
end

function neighboring_candidates(st::GrowState)::Vector{Int64}
    @unpack head, vm, g, current_depth, max_depth= st
    # reached max depth. terminate
    current_depth == max_depth && return Int64[]
    ns = neighbors(g, head)
    ns[vm[ns]]
end

function valid_spaces(r::Room)::PersistentVector{Bool} end

function valid_spaces(r::Room, vm::PersistentVector{Bool})
    PersistentVector{Bool}(valid_spaces(r) .& vm)
end

function valid_spaces(r::GridRoom)
    g = pathgraph(r)
    d = data(r)
    vec_d = vec(d)
    valid_map = Vector{Bool}(vec_d .== floor_tile)
    valid_map[entrance(r)] .= false
    valid_map[exits(r)] .= false

    # cannot block entrances
    for v in entrance(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end
    # cannot block exits
    for v in exits(r)
        ns = neighbors(g, v)
        valid_map[ns] .= false
    end

    # want to prevent "stitching" new pieces together
    # fvs = Set{Int64}(findall(vec_d .== obstacle_tile))
    fvs = findall(vec_d .== obstacle_tile)
    ds = gdistances(g, fvs)
    valid_map[ds .<= 1] .= false
    # purge_around!(valid_map, r, fvs)

    PersistentVector(valid_map)
end

# having to deal with type instability
function merge_prod(st::GrowState, children::Set{Int64})
    @unpack head = st
    union(children, head)
end
function merge_prod(st::GrowState, children::Vector{Set{Int64}})
    @unpack head = st
    isempty(children) ? Set(head) : union(first(children), head)
end

function merge_prod(st::FurnishState, children::Set{Int64})
    @unpack f = st
    # @show f
    union(children, f)
    # children
end
function merge_prod(st::FurnishState, children::Vector{Set{Int64}})
    @unpack f = st
    # @show f
    isempty(children) ? f : union(first(children), f)
    # isempty(children) ? f : first(children)
end


function is_floor(r::GridRoom, t::Int64)::Bool
    g = pathgraph(r)
    d = data(r)
    has_vertex(g, t) && d[t] == floor_tile
end


function purge_around!(vm::Vector{Bool}, r::GridRoom, f)
    # want to prevent "stitching" new pieces together
    m,n = size(data(r))
    for idx = f
        if idx - m >= 1
            vm[idx - m] = false
        end
        if idx - 1 >= 1
            vm[idx - 1] = false
        end
        if idx + m <= (m * n)
            vm[idx + m] = false
        end
        if idx + 1 <= (m * n)
            vm[idx + 1] = false
        end
    end
    return nothing
end


include("gen.jl")
# include("path_based/path_based.jl")
