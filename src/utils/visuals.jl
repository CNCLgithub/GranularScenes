export room_topdown


"""
    room_topdown(r::GridRoom; kw...) -> Matrix{RGB}

Top-down color map of a `GridRoom`. Uses the same linear-index convention as
the rest of the codebase: `data(gr)` is a matrix with `gr.steps[1]` rows ×
`gr.steps[2]` columns, indexed `data[row, col]` in Julia. Render the image
transposed if you want x (first spatial axis) as columns and y (second) as
rows in the display.

Colors: floor = light gray, wall = dark blue-gray, obstacle = near-black,
entrance tiles = green, exit tiles = purple.
"""
function room_topdown(r::GridRoom;
                      floor_color  = RGB(0.95, 0.95, 0.95),
                      wall_color   = RGB(0.0, 0.0, 0.35),
                      obstacle_color = RGB(0.05, 0.05, 0.05),
                      entrance_color = RGB(0.0, 0.6, 0.0),
                      exit_color     = RGB(0.7, 0.0, 0.7))
    dmat = data(r)
    img = map(dmat) do t
        if t == wall_tile
            wall_color
        elseif t == obstacle_tile
            obstacle_color
        else
            floor_color
        end
    end
    for i in entrance(r)
        checkbounds(Bool, img, i) && (img[i] = entrance_color)
    end
    for i in exits(r)
        checkbounds(Bool, img, i) && (img[i] = exit_color)
    end
    img
end

"""
    room_topdown!(img, r::GridRoom) -> Matrix{RGB}

In-place variant: overwrites a pre-allocated d×d image (d = steps[1] must
match rows) with the room's occupancy, leaving entrance/exit markers off.
"""
function room_topdown!(img::AbstractMatrix{<:Colorant}, r::GridRoom)
    dmat = data(r)
    size(dmat) == size(img) || throw(ArgumentError("Size mismatch"))
    for i in eachindex(img)
        img[i] = dmat[i] == wall_tile ? RGB(0.0, 0.0, 0.35) :
                 dmat[i] == obstacle_tile ? RGB(0.05, 0.05, 0.05) :
                                            RGB(0.95, 0.95, 0.95)
    end
    img
end
