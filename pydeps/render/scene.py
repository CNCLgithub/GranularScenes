import time
import os
from datetime import datetime
import numpy as np
import taichi as ti
from taichi.math import vec3, ivec3
from .renderer import Renderer
from .math_utils import np_normalize, np_rotate_matrix
import __main__


VOXEL_DX = 1 / 64
SCREEN_RES = (1280, 720)
TARGET_FPS = 30
UP_DIR = (0, 1, 0)
HELP_MSG = '''
====================================================
Camera:
* Drag with your left mouse button to rotate
* Press W/A/S/D/Q/E to move
====================================================
'''

MAT_LAMBERTIAN = 1
MAT_LIGHT = 2

class Camera:
    def __init__(self, window, up):
        self._window = window
        self._camera_pos = np.array((0.4, 0.5, 2.0))
        self._lookat_pos = np.array((0.0, 0.0, 0.0))
        self._up = np_normalize(np.array(up))
        self._last_mouse_pos = None

    @property
    def mouse_exclusive_owner(self):
        return True

    def update_camera(self):
        res = self._update_by_wasd()
        res = self._update_by_mouse() or res
        return res

    def _update_by_mouse(self):
        win = self._window
        if not self.mouse_exclusive_owner or not win.is_pressed(ti.ui.LMB):
            self._last_mouse_pos = None
            return False
        mouse_pos = np.array(win.get_cursor_pos())
        if self._last_mouse_pos is None:
            self._last_mouse_pos = mouse_pos
            return False
        # Makes camera rotation feels right
        dx, dy = self._last_mouse_pos - mouse_pos
        self._last_mouse_pos = mouse_pos

        out_dir = self._lookat_pos - self._camera_pos
        leftdir = self._compute_left_dir(np_normalize(out_dir))

        # REVIEW: what does `3` mean?
        scale = 3
        rotx = np_rotate_matrix(self._up, dx * scale)
        roty = np_rotate_matrix(leftdir, dy * scale)

        out_dir_homo = np.array(list(out_dir) + [0.0])
        new_out_dir = np.matmul(np.matmul(roty, rotx), out_dir_homo)[:3]
        self._lookat_pos = self._camera_pos + new_out_dir

        return True

    def _update_by_wasd(self):
        win = self._window
        tgtdir = self.target_dir
        leftdir = self._compute_left_dir(tgtdir)
        lut = [
            ('w', tgtdir),
            ('a', leftdir),
            ('s', -tgtdir),
            ('d', -leftdir),
            ('e', [0, -1, 0]),
            ('q', [0, 1, 0]),
        ]
        dir = np.array([0.0, 0.0, 0.0])
        pressed = False
        for key, d in lut:
            if win.is_pressed(key):
                pressed = True
                dir += np.array(d)
        if not pressed:
            return False
        dir *= 0.05
        self._lookat_pos += dir
        self._camera_pos += dir
        return True

    @property
    def position(self):
        return self._camera_pos

    @property
    def look_at(self):
        return self._lookat_pos

    @property
    def target_dir(self):
        return np_normalize(self.look_at - self.position)

    def _compute_left_dir(self, tgtdir):
        cos = np.dot(self._up, tgtdir)
        if abs(cos) > 0.999:
            return np.array([-1.0, 0.0, 0.0])
        return np.cross(self._up, tgtdir)


@ti.data_oriented
class Scene:
    def __init__(self,
                 voxel_edges=0.0,
                 exposure=3,
                 voxel_grid_res = 128,
                 screen_res = SCREEN_RES,
                 window = False):
        ti.init(arch=ti.cuda)
        # print(HELP_MSG)
        if window:
            self.window = ti.ui.Window("Taichi Voxel Renderer",
                                    screen_res,
                                    vsync=True)
        else:
            self.window = None

        self.camera = Camera(self.window, up=UP_DIR)
        self.renderer = Renderer(voxel_grid_res, # VOXEL_DX,
                                 image_res=screen_res,
                                 up=UP_DIR,
                                 voxel_edges=voxel_edges,
                                 exposure=exposure)

        self.renderer.set_camera_pos(*self.camera.position)


    @staticmethod
    @ti.func
    def round_idx(idx_):
        idx = ti.cast(idx_, ti.f32)
        return ti.Vector(
            [ti.round(idx[0]),
             ti.round(idx[1]),
             ti.round(idx[2])]).cast(ti.i32)

    @ti.func
    def set_voxel(self, idx, mat, color, reset = False):
        self.renderer.set_voxel(self.round_idx(idx), mat, color, reset)

    @ti.func
    def set_voxel_unsafe(self, idx, mat, color, reset = False):
        self.renderer.set_voxel(idx, mat, color, reset)

    @ti.func
    def get_voxel(self, idx):
        mat, color = self.renderer.get_voxel(self.round_idx(idx))
        return mat, color

    @ti.func
    def set_floor(self, height, color):
        self.renderer.floor_height[None] = height
        self.renderer.floor_color[None] = color

    def set_directional_light(self, direction, direction_noise, color):
        self.renderer.set_directional_light(direction, direction_noise, color)

    def set_background_color(self, color):
        self.renderer.background_color[None] = color


    @ti.kernel
    def set_exterior(self,
                     walls: ti.types.ndarray(dtype=ti.i32,
                                             ndim=2)):
        gray = vec3(1.0, 1.0, 1.0)
        n = max(*walls.shape)
        hn = n // 2
        height = 2 * n // 3 - hn

        self.set_floor(-hn + 0.05, (1.0, 1.0, 1.0))

        for I in ti.grouped(walls):
            if walls[I]:
                x = I[0] - hn
                z = I[1] - hn
                for y in range(-hn, height):
                    self.set_voxel(vec3(x, y, z), 1, gray,
                                   reset = True)

        # ceiling and floor
        for i in range(-hn, hn):
            for j in range(-hn, hn):
                self.set_voxel(vec3(i, height, j), 1, vec3(1, 1, 1),
                               reset = True)
                self.set_voxel(vec3(i, -hn, j), 1, vec3(1, 1, 1),
                               reset = True)

    @ti.kernel
    def set_lights(self, n:int):
        hn = n // 2
        height = 2 * n // 3 - hn
        # add lights to the ceiling
        for i in range(-n // 8, n // 8):
            for j in range(-n // 8, n // 8):
                pos = vec3(i, height, j)
                self.set_voxel(pos, 2, vec3(1, 1, 1), reset = True)


    @ti.kernel
    def set_obstacle(self,
                     i: int,
                     j: int,
                     val : ti.f32):
        blue = vec3(0.3, 0.3, 0.9)
        nx,ny = self.rand_buffer.shape
        n = max(nx, ny)
        hn = n // 2
        oheight = 1 * n // 6 - hn
        x = i - hn
        z = j - hn
        for y in ti.ndrange((-hn, oheight)):
            self.set_voxel(vec3(x, y, z),
                           val,
                           blue)



    def set_obstacles(self, omap):
        # t = time.time()
        self.renderer.rand_buffer.from_numpy(omap)
        self.renderer.set_obstacles()
        # print(f'elapsed `set_obstacles`: {time.time() - t}')
        # _blue = vec3(0.3, 0.3, 0.9)
        # blue = vec3(0.3, 0.3, 0.9)
        # # blue = self.renderer.to_vec3u(_blue)
        # #
        # self.renderer.rand_buffer.from_numpy(omap)

        # nx,ny = omap.shape
        # n = max(nx, ny)
        # hn = n // 2
        # oheight = 1 * n // 6 - hn

        # ti.loop_config(block_dim=8)
        # # for I in ti.grouped(omap):
        # for i,j in self.renderer.rand_buffer:
        #     if omap[i, j] > 0.:
        #         # x = i - hn
        #         # z = j - hn
        #         for y in range(-hn, oheight):
        #             self.set_voxel(vec3(i - hn, y, j - hn), omap[i, j], blue)

    def reset_voxels(self):
        self.renderer.reset_voxels()

    @property
    def voxel_buffer(self):
        return self.renderer.voxel_material

    @property
    def rand_buffer(self):
        return self.renderer.rand_buffer


    def random(self, var: float, spp:int):
        result = np.zeros((*self.renderer.image_res, 3),
                          dtype = np.float32)
        self.render_scene(spp)
        self.renderer.random(result, var)
        return result

    def logpdf(self, img, var, spp):
        self.render_scene(spp)
        return self.renderer.logpdf(img, var)


    def save_img(self, img, dirpath = os.getcwd()):
            timestamp = datetime.today().strftime('%Y-%m-%d-%H%M%S')
            main_filename = os.path.split(__main__.__file__)[1]
            fname = os.path.join(dirpath, f"{timestamp}.jpg")
            ti.tools.image.imwrite(img, fname)
            print(f"Screenshot has been saved to {fname}")


    def render_scene(self, spp : int):

        # t = time.time()
        self.renderer.recompute_bbox()
        self.renderer.reset_framebuffer()
        # elapsed = time.time() - t
        # print(f'reset time: {elapsed}')

        # t = time.time()
        for _ in range(spp):
            self.renderer.accumulate()
        # elapsed = time.time() - t
        # print(f'accumulate time: {elapsed}')

        # t = time.time()
        img = self.renderer.fetch_image()
        # elapsed = time.time() - t
        # print(f'image time: {elapsed}')
        return img


    def finish(self):
        self.renderer.recompute_bbox()
        canvas = self.window.get_canvas()
        spp = 1
        while self.window.running:
            should_reset_framebuffer = False

            if self.camera.update_camera():
                self.renderer.set_camera_pos(*self.camera.position)
                look_at = self.camera.look_at
                self.renderer.set_look_at(*look_at)
                should_reset_framebuffer = True

            if should_reset_framebuffer:
                self.renderer.reset_framebuffer()

            t = time.time()
            for _ in range(spp):
                self.renderer.accumulate()
            img = self.renderer.fetch_image()

            if self.window.is_pressed('c'):
                print(f"Camera position {self.camera.position}")
                print(f"Camera lookat {self.camera.look_at}")

            if self.window.is_pressed('p'):
                self.save_img(img)
            canvas.set_image(img)
            elapsed_time = time.time() - t
            # print(f'accumulate time (spp={spp}): {elapsed_time}')
            if elapsed_time * TARGET_FPS > 1:
                spp = int(spp / (elapsed_time * TARGET_FPS) - 1)
                spp = max(spp, 1)
            else:
                spp += 1
            self.window.show()
