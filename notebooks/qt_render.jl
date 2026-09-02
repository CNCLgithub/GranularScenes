### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 1d39e7ee-a6e3-11f1-3258-19edf57342c6
begin
	using Pkg
	Pkg.activate("..")
	
	using Gen
	using PlutoUI
	using Random
	using Printf
	using JSON
	using Colors
	# using StatProfilerHTML

	using Rooms
	
	using Revise
	using GranularScenes
end


# ╔═╡ d697c7c5-664d-4273-a24a-78823aab6bae
html"""
<style>
    @media screen {
        main {
            margin: 0 auto;
            max-width: 3000px;
            padding-left: max(100px, 10%);
            padding-right: max(100px, 10%);
        }
    }
	pluto-output {
    font-size: 1.2em; /* Adjust base text size */
    font-family: "Inter";
	}

pluto-output h1 {
    font-size: 2.5rem; /* Adjust header sizes */
	font-family: "Inter";
}

pluto-output h2 {
    font-size: 3.0rem;
}

cm-editor .cm-scroller,
.cm-editor .cm-content {
    font-family: "Fira Code", monospace !important;
    font-size: 18px !important; /* Adjust size here */
}
</style>
"""

# ╔═╡ 14a33876-0998-47a2-a7ce-96cace0cd335
dataset = "window-0.1/2025-02-05_vifdDO"

# ╔═╡ 8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
function load_room(idx::Int)

    base_path = "/spaths/datasets/$(dataset)/scenes"
    path = joinpath(base_path, "$(idx).json")
    local base_s
    open(path, "r") do f
        base_s = JSON.parse(f)
    end
    from_json(GridRoom, base_s)
end

# ╔═╡ 67bb0b77-f540-480a-aa42-0188d0df1ca4
function mytest()
    r = load_room(1)

    params = QuadTreeModel(r;
                           render_kwargs =
                               Dict(:image_res => (256,256),
                                    :use_cuda => false),
                           pixel_var = 0.001)

    @time (trace, ll) = generate(qt_model, (params,))
   
    img = trace[:depth][:, :, 1]
    img_min, img_max = extrema(img)
    scaled = (img .- img_min)
    scaled .*= one(Float32) / maximum(scaled)

    gray = Gray.(scaled)
end


# ╔═╡ a61eabea-5349-4121-a63f-cd7c9b52bebb
mytest()

# ╔═╡ Cell order:
# ╟─d697c7c5-664d-4273-a24a-78823aab6bae
# ╠═1d39e7ee-a6e3-11f1-3258-19edf57342c6
# ╠═14a33876-0998-47a2-a7ce-96cace0cd335
# ╠═8d9add3f-dbc5-47c5-8ac3-3a7dbfc4ef94
# ╠═67bb0b77-f540-480a-aa42-0188d0df1ca4
# ╠═a61eabea-5349-4121-a63f-cd7c9b52bebb
