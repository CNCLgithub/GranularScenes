#!/usr/bin/env python3

import time
from pydeps import vxl_scene
import numpy as np

obstacles = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype = np.float32).repeat(2, 0).repeat(2, 1)

obs_x, obs_y = np.where(obstacles)

def write_obs(scene, xs, ys):
    # scene.renderer.rand_buffer.from_numpy(obstacles)
    scene.set_obstacles(obstacles)
    # for (x, y) in zip(xs, ys):
    #     scene.set_obstacle(x, y, 1.0)

walls = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
], dtype = np.int32).repeat(2, 0).repeat(2, 1)

scene = vxl_scene(32, (128, 128), window = False)
# scene = vxl_scene(32, (720, 480), window = True)
scene.set_exterior(walls)
scene.set_lights(32)
scene.set_obstacles(obstacles)
scene.renderer.set_look_at(-0.3, -0.2, -0.016)
scene.renderer.set_camera_pos(1.78, -0.03, -0.015)
# scene.finish()


# for _ in range(100):
#     t = time.time()
#     scene.reset_voxels()
#     elapsed = time.time() - t
#     print(f'Reset time: {elapsed}')
#     t = time.time()
#     # scene.set_obstacles(obstacles)
#     write_obs(scene, obs_x, obs_y)
#     elapsed = time.time() - t
#     print(f'Write time: {elapsed}')
#     t = time.time()
#     # scene.set_obstacles(obstacles)
#     write_obs(scene, obs_x, obs_y)
#     elapsed = time.time() - t
#     print(f'Write time: {elapsed}')
#     t = time.time()
#     scene.render_scene(15)
#     elapsed = time.time() - t
#     print(f'Rendering time: {elapsed}')

# t = time.time()
# scene.reset_voxels()
# elapsed = time.time() - t
# print(f'Reset time: {elapsed}')
# t = time.time()
# scene.set_obstacles(obstacles)
# elapsed = time.time() - t
# print(f'Write time: {elapsed}')
# t = time.time()
# img = scene.render_scene(15)
# elapsed = time.time() - t
# print(f'Rendering time: {elapsed}')
# scene.save_img(img, dirpath = "test/screenshots/")

spp = 30
rimg = scene.render_scene(spp).to_numpy()
# rimg = scene.random(0.1, 15)
print(type(rimg))
print(rimg.shape)
scene.save_img(rimg, dirpath = "test/screenshots/")
t = time.time()
nsteps = 20
for _ in range(nsteps):
    logpdf = scene.logpdf(rimg, 0.1, spp)
    print(f'Logpdf: {logpdf}')

elapsed = (time.time() - t) / nsteps
print(f'Time per call: {elapsed}')
