export TaichiScene,
    render,
    TaichiObserve,
    observe_pixels,
    render

struct TaichiScene
    voxscene::PyObject
    voxel_buffer::PyObject # REVIEW: needed?
    obstacle_map::Matrix{Float32}
end

"""
    $(TYPEDSIGNATURES)

Initialize Taichi scene renderer with template room.
"""
function TaichiScene(template::GridRoom;
                     resolution = (100, 100),
                     window = false)
    h, w = Rooms.steps(template)
    n = max(h, w)
    vx = @pycall vox.vxl_scene(n, resolution, window=window)::PyObject
    @pycall vx.renderer.set_look_at(-0.001, -0.216, -0.391)::PyObject
    @pycall vx.renderer.set_camera_pos(-0.015, -0.031, -2.477)::PyObject
    # @pycall vx.renderer.set_look_at(-0.3, -0.2, -0.016)::PyObject
    # @pycall vx.renderer.set_camera_pos(1.78, -0.03, -0.015)::PyObject
    voxel_buffer = init_voxel_buffer!(vx, template)
    obstacle_map = zeros(Float32, h, w)
    TaichiScene(vx, voxel_buffer, obstacle_map)
end


function init_voxel_buffer!(vx::PyObject,
                            template::GridRoom)
    h, w = Rooms.steps(template)
    n = max(h, w)
    wall_map = Int32.(data(template) .== wall_tile)
    @pycall vx.set_exterior(wall_map)::PyObject
    @pycall vx.set_lights(n)::PyObject
    buffer = vx."voxel_buffer"
    return buffer
end


function write_obstacles!(m::Matrix{Float32}, gr::GridRoom)
    for i = eachindex(m)
        m[i] = data(gr)[i] == obstacle_tile
    end
    return nothing
end

function write_obstacles!(m::Matrix{Float32}, qt::QuadTree)
    fill!(m, 0.) # REVIEW: needed?
    project_qt!(m, qt.leaves)
    return nothing
end

function render(scene::TaichiScene, obj)
    vx = scene.voxscene
    # clear voxels
    @pycall vx.reset_voxels()::PyObject

    write_obstacles!(scene.obstacle_map, obj)
    @pycall vx.set_obstacles(scene.obstacle_map)::PyObject

    result = @pycall vx.render_scene()::PyObject

    return result
end

function window(scene::TaichiScene, obj)
    vx = scene.voxscene
    # clear voxels
    @pycall vx.reset_voxels()::PyObject
    write_obstacles!(scene.obstacle_map, obj)
    @pycall vx.set_obstacles(scene.obstacle_map)::PyObject
    @pycall scene.voxscene.finish()::PyObject
    return nothing
end

"""

A `Gen.Distribution` implementation of broadcastable image rendering in Tiachi
"""
struct TaichiObserve <: Gen.Distribution{PyObject} end

const observe_pixels = TaichiObserve()

function Gen.random(::TaichiObserve, s::TaichiScene, obj, var::Float32)
    vx = s.voxscene
    # clear voxels
    @pycall vx.reset_voxels()::PyObject
    write_obstacles!(s.obstacle_map, obj)
    @pycall vx.set_obstacles(s.obstacle_map)::PyObject
    result = @pycall vx.random(var)::PyObject
    return result
end

function Gen.logpdf(::TaichiObserve, img::PyObject, s::TaichiScene,
                    obj, var::Float32)
    vx = s.voxscene
    # clear voxels
    @pycall vx.reset_voxels()::PyObject
    write_obstacles!(s.obstacle_map, obj)
    @pycall vx.set_obstacles(s.obstacle_map)::PyObject
    # py"print($img.shape)"
    result = @pycall vx.logpdf(img, var)::Float64
    return result
end

(::TaichiObserve)(s, obj, var) = Gen.random(observe_pixels, s, obj, var)

is_discrete(::TaichiObserve) = false
Gen.has_output_grad(::TaichiObserve) = false
Gen.logpdf_grad(::TaichiObserve, value::Set, args...) = (nothing,)
