import taichi as ti
from .scene import Scene


def vxl_scene(n:int, resolution:tuple((int, int)),
              window = False):
    s = Scene(voxel_grid_res = 2 * n,
              screen_res = resolution,
              window = window)
    return s
