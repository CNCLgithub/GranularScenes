import taichi as ti
from taichi.math import (vec3, clamp, sign)
import numpy as np

from .math_utils import (eps, inf, out_dir, ray_aabb_intersection,
                         round_idx)

MAX_RAY_DEPTH = 5
use_directional_light = True

DIS_LIMIT = 200.0


@ti.data_oriented
class Renderer:
    def __init__(self, voxel_grid_res, image_res, up, voxel_edges,
                 exposure=1.0):
                 # exposure=3):
        self.image_res = image_res
        self.aspect_ratio = image_res[0] / image_res[1]
        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]
        self.current_spp = 0

        # self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.depth_buffer = ti.field(dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.voxel_color = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material = ti.field(dtype=ti.f32)
        self.voxel_color_reset = ti.Vector.field(3, dtype=ti.u8)
        self.voxel_material_reset = ti.field(dtype=ti.f32)

        self.light_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.light_direction_noise = ti.field(dtype=ti.f32, shape=())
        self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.cast_voxel_hit = ti.field(ti.i32, shape=())
        self.cast_voxel_index = ti.Vector.field(3, ti.i32, shape=())

        self.voxel_edges = voxel_edges
        self.exposure = exposure

        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.floor_height = ti.field(dtype=ti.f32, shape=())
        self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.voxel_grid_res = voxel_grid_res
        self.voxel_dx = 2 / voxel_grid_res
        self.voxel_inv_dx = 1 / self.voxel_dx
        # Note that voxel_inv_dx == voxel_grid_res iff the box has width = 1
        voxel_grid_offset = [-self.voxel_grid_res // 2 for _ in range(3)]

        # ti.root.dense(ti.ij, image_res).place(self.color_buffer)
        ti.root.dense(ti.ij, image_res).place(self.depth_buffer)
        ti.root.dense(ti.ijk,
                      self.voxel_grid_res).place(self.voxel_color,
                                                 self.voxel_material,
                                                 self.voxel_color_reset,
                                                 self.voxel_material_reset,
                                                 offset=voxel_grid_offset)

        self._rendered_image = ti.Vector.field(3, float, image_res)
        self.rand_buffer = ti.field(dtype=ti.f32)
        ti.root.dense(ti.ij, voxel_grid_res // 2).place(self.rand_buffer)

        self.set_up(*up)
        self.set_fov(0.23)

        self.floor_height[None] = 0
        self.floor_color[None] = (1, 1, 1)

    def set_directional_light(self, direction, light_direction_noise,
                              light_color):
        direction_norm = (direction[0]**2 + direction[1]**2 +
                          direction[2]**2)**0.5
        self.light_direction[None] = (direction[0] / direction_norm,
                                      direction[1] / direction_norm,
                                      direction[2] / direction_norm)
        self.light_direction_noise[None] = light_direction_noise
        self.light_color[None] = light_color

    @ti.func
    def inside_grid(self, ipos):
        return ipos.min() >= -self.voxel_grid_res // 2 and ipos.max(
        ) < self.voxel_grid_res // 2

    @ti.func
    def unsafe_query_density(self, ipos):
        ret = self.voxel_material[ipos]
        return ret

    @ti.func
    def query_density(self, ipos):
        inside = self.inside_grid(ipos)
        ret = 0.0
        if inside:
            ret = self.voxel_material[ipos]
        # else:
        #     ret = 1.0
        return ret

    @ti.func
    def _to_voxel_index(self, pos):
        p = pos * self.voxel_inv_dx
        voxel_index = ti.floor(p).cast(ti.i32)
        return voxel_index

    @ti.func
    def voxel_surface_color(self, pos):
        # REVIEW: relative position within voxel?
        p = pos * self.voxel_inv_dx
        p -= ti.floor(p)

        boundary = self.voxel_edges
        count = 0
        for i in ti.static(range(3)):
            if p[i] < boundary or p[i] > 1 - boundary:
                count += 1

        # REVIEW: only used for edge highlighting?
        f = 0.0
        if count >= 2:
            f = 1.0

        voxel_index = self._to_voxel_index(pos)
        voxel_color = ti.Vector([0.0, 0.0, 0.0])
        is_light = 0
        if self.inside_particle_grid(voxel_index):
            voxel_color = self.voxel_color[voxel_index] * (1.0 / 255)
            if self.voxel_material[voxel_index] == 2:
                is_light = 1

        return voxel_color * (1.3 - 1.2 * f), is_light

    @ti.func
    def ray_march(self, p, d):
        dist = inf
        if d[1] < -eps:
            dist = (self.floor_height[None] - p[1]) / d[1]

        return dist

    @ti.func
    def sdf_normal(self, p):
        return ti.Vector([0.0, 1.0, 0.0])  # up

    @ti.func
    def sdf_color(self, p):
        return self.floor_color[None]

    @ti.func
    def dda_voxel(self, eye_pos, d):
        for i in ti.static(range(3)):
            if abs(d[i]) < 1e-6:
                d[i] = 1e-6
        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        bbox_min = self.bbox[0]
        bbox_max = self.bbox[1]
        inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                                 d)

        normal = ti.Vector([0.0, 0.0, 0.0])
        hit_distance = inf
        weight = 1.0

        if inter:
            near = max(0, near)
            pos = eye_pos + d * (near + 5 * eps)
            o = self.voxel_inv_dx * pos
            ipos = int(ti.floor(o))
            dis = (ipos - o + 0.5 + rsign * 0.5) * rinv

            running = 1
            while running:

                # voxel weight
                w = clamp(self.query_density(ipos), 0.0, 1.0)

                if w > eps: # return distance of voxel surface
                    mini = (ipos - o + ti.Vector([0.5, 0.5, 0.5]) -
                            rsign * 0.5) * rinv
                    hit_distance = mini.max() * self.voxel_dx + near
                    weight = w
                    running = 0
                else: # no surface reached - continue ray
                    mm = ti.Vector([0, 0, 0])
                    if dis[0] <= dis[1] and dis[0] < dis[2]:
                        mm[0] = 1
                    elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                        mm[1] = 1
                    else:
                        mm[2] = 1
                    dis += mm * rsign * rinv
                    ipos += mm * rsign
                    normal = -mm * rsign

                # outside range of voxels - exit
                if not self.inside_particle_grid(ipos):
                    # hit_distance = DIS_LIMIT
                    # weight = 1.0
                    running = 0

        return hit_distance, weight, normal

    @ti.func
    def inside_particle_grid(self, ipos):
        pos = ipos * self.voxel_dx
        return self.bbox[0][0] <= pos[0] and pos[0] < self.bbox[1][
            0] and self.bbox[0][1] <= pos[1] and pos[1] < self.bbox[1][
                1] and self.bbox[0][2] <= pos[2] and pos[2] < self.bbox[1][2]

    @ti.func
    def next_hit(self, pos, d):
        normal = ti.Vector([0.0, 0.0, 0.0])
        # next voxel surface, possibly out of bounds
        hit_distance, weight, normal = self.dda_voxel(pos, d)

        # check if floor is closer
        ray_march_dist = self.ray_march(pos, d)
        if hit_distance < eps or ray_march_dist < hit_distance:
        # if ray_march_dist < hit_distance:
            hit_distance = ray_march_dist
            weight = 1.0 # don't go further
            normal = self.sdf_normal(pos + d * hit_distance)


        # hit_distance = max(0.0, hit_distance)

        return hit_distance, weight, normal

    @ti.kernel
    def set_camera_pos(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.camera_pos[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_up(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.up[None] = ti.Vector([x, y, z]).normalized()

    @ti.kernel
    def set_look_at(self, x: ti.f32, y: ti.f32, z: ti.f32):
        self.look_at[None] = ti.Vector([x, y, z])

    @ti.kernel
    def set_fov(self, fov: ti.f32):
        self.fov[None] = fov

    @ti.func
    def get_cast_dir(self, u, v):
        fov = self.fov[None]
        d = (self.look_at[None] - self.camera_pos[None]).normalized()
        # fu = (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
        fu = (2 * fov * (u) / self.image_res[1] -
              fov * self.aspect_ratio - 1e-5)
        # fv = 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5
        fv = 2 * fov * (v ) / self.image_res[1] - fov - 1e-5
        du = d.cross(self.up[None]).normalized()
        dv = du.cross(d).normalized()
        d = (d + fu * du + fv * dv).normalized()
        return d

    @ti.kernel
    def render(self):
        ti.loop_config(block_dim=256)
        for u, v in self.depth_buffer:
            cdir = self.get_cast_dir(u, v)
            pos = self.camera_pos[None]
            t = 0.0

            new_pos = ti.Vector([0.0, 0.0, 0.0]) + pos
            depth = 0.0
            acc_dist = 0.0
            mass = 1.0
            dist = -eps
            w = 0.0
            steps = 0

            # Tracing begin
            while steps < MAX_RAY_DEPTH and mass > eps and acc_dist < DIS_LIMIT:
                new_pos = new_pos + (dist + eps) * cdir
                dist, w, normal = self.next_hit(new_pos, cdir)
                acc_dist += dist
                depth += mass * w * acc_dist
                # if u == 34 and v == 5:
                #     print(f'{steps=} {pos=} {new_pos=} {mass=} {dist=} {w=} {acc_dist=} {depth=} {cdir=}')
                mass *= (1.0 - w)
                steps += 1

            # top off with final hit
            depth += mass * acc_dist

            self.depth_buffer[u, v] = depth

    @ti.kernel
    def set_obstacles(self):
        blue = vec3(0.3, 0.3, 0.9)
        nx,ny = self.rand_buffer.shape
        n = max(nx, ny)
        hn = n // 2
        oheight = 1 * n // 6 - hn

        for i,j in self.rand_buffer:
            if self.rand_buffer[i, j] > eps:
                x = i - hn
                z = j - hn
                for y in range(-hn, oheight):
                    self.set_voxel(round_idx(vec3(x, y, z)),
                                   self.rand_buffer[i, j],
                                   blue)


    @ti.kernel
    def random(self,
               result: ti.types.ndarray(ndim = 3, dtype = ti.f32),
               var: float):
        for i,j in self._rendered_image:
            for c in ti.static(range(3)):
                result[i,j,c] = self._rendered_image[i,j][c] + ti.randn(ti.f32) * var


    @ti.kernel
    def logpdf(self,
               img: ti.types.ndarray(ndim = 3, dtype=ti.f32),
               var: float) -> float:
        result = 0.
        for u, v in self._rendered_image:
            # for c in ti.static(range(3)):
            z = (img[u, v, 0] - self._rendered_image[u, v][0]) / var
            zsqr = z * z
            ls = -1 * (zsqr + ti.log(2*np.pi))/2 - ti.log(var)
            result += ls
        return result

    @ti.kernel
    def _render_to_image(self):
        ti.loop_config(block_dim=256)
        for i, j in self.depth_buffer:
            # v = 0.3 * ti.sqrt(self.depth_buffer[i, j])
            v = 0.25 * self.depth_buffer[i, j]
            for c in ti.static(range(3)):
                self._rendered_image[i, j][c] = v

    @ti.kernel
    def recompute_bbox(self):
        for d in ti.static(range(3)):
            self.bbox[0][d] = 1e9
            self.bbox[1][d] = -1e9
        for I in ti.grouped(self.voxel_material):
            if self.voxel_material[I] != 0:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.bbox[0][d], (I[d] - 1) * self.voxel_dx)
                    ti.atomic_max(self.bbox[1][d], (I[d] + 2) * self.voxel_dx)

    @ti.kernel
    def reset_voxels(self):
        # self.voxel_material.fill(0.)
        # self.voxel_color.fill(0)
        for I in ti.grouped(self.voxel_material):
            self.voxel_material[I] = self.voxel_material_reset[I]
            self.voxel_color[I] = self.voxel_color_reset[I]

    def reset_framebuffer(self):
        self.current_spp = 0
        # self.color_buffer.fill(0)
        self.depth_buffer.fill(0)

    def accumulate(self):
        self.render()
        self.current_spp += 1

    def fetch_image(self, normalize = True):
        # if normalize:
        #     self._render_to_image(self.current_spp)
        self._render_to_image()
        return self._rendered_image

    @staticmethod
    @ti.func
    def to_vec3u(c):
        c = ti.math.clamp(c, 0.0, 1.0)
        r = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i] * 255, ti.u8)
        return r

    @staticmethod
    @ti.func
    def to_vec3(c):
        r = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            r[i] = ti.cast(c[i], ti.f32) / 255.0
        return r

    @ti.func
    def set_voxel(self, idx, mat, color, reset = False):
        self.voxel_material[idx] = ti.cast(mat, ti.f32)
        self.voxel_color[idx] = self.to_vec3u(color)
        if reset:
            self.voxel_material_reset[idx] = self.voxel_material[idx]
            self.voxel_color_reset[idx] = self.voxel_color[idx]


    @ti.func
    def set_voxel_unsafe(self, idx, mat, color):
        self.voxel_material[idx] = mat
        self.voxel_color[idx] = color

    @ti.func
    def get_voxel(self, ijk):
        mat = self.voxel_material[ijk]
        color = self.voxel_color[ijk]
        return mat, self.to_vec3(color)
