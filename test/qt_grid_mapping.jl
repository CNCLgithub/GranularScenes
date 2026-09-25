import GranularScenes as GS
using StaticArrays: SVector
using Rooms
using JSON

# T1: round-trip point -> leaf -> bounds contains point
qt = GS.QuadTree(4, 5)
for p in (SVector(-0.46875, -0.03125), SVector(0.46875, 0.21875))
    n = GS.leaf_at(qt, p)
    b = GS.node_bounds(qt, n)
    @show p
    @show b
    @show GS.area(b)
    @assert b.xmin <= p[1] <= b.xmax && b.ymin <= p[2] <= b.ymax
    println(p, " -> ", n, " bounds=", b)
end
# Expect: start -> (x=0,y=3) i.e. left wall mid; goal -> (x=7,y=5) right wall below-mid.

# T2: leaf_lin_idxs! ↔ write_obstacles! agreement
d = GS.finest_grid(qt)
occ = Matrix{Float32}(undef, d, d); buf = Int64[]
GS.write_obstacles!(occ, qt, d)
buf2 = zeros(Int64, d*d)
k = GS.leaf_lin_idxs!(buf2, qt, first(qt.schema.leaves), d)
# T3: every idx in buf2, when decomposed, must hit a cell inside node_bounds
n = first(qt.schema.leaves)
b = GS.node_bounds(qt, n)
for i in 1:k
    li = buf2[i]; r = (li-1) % d + 1; c = (li-1) ÷ d + 1
    xc = (c - 0.5)/d - 0.5; yc = (r - 0.5)/d - 0.5
    @assert b.xmin <= xc <= b.xmax && b.ymin <= yc <= b.ymax
end

dataset = "window-0.1/2025-02-05_vifdDO"
function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx).json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end
# T4: corner indices of room map to correct node-space corners
# r = <your 16x16 GridRoom>   # fill in from notebook
r = load_room(1)
mk = GS.room_index_to_point(r, 1)            # (row 1, col 1)
mk2 = GS.room_index_to_point(r, 16*16)       # (row 16, col 16)
@assert mk ≈ SVector(-0.46875, -0.46875)     # top-left corner in tile coords
@assert mk2 ≈ SVector(0.46875, 0.46875)      # bottom-right

# T5: entrance/exit decode (the acceptance test)
println(GS.room_index_to_point(r, first(GS.entrance(r))))
println(GS.room_index_to_point(r, first(GS.exits(r))))
# Expect (-0.46875, -0.03125) and (0.46875, 0.21875) for steps=(16,16)
