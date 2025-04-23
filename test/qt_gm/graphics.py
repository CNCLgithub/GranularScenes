#!/usr/bin/env python3

import time
from pydeps import vxl_scene
import numpy as np



weights = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,.8,0,0,.8,0,0,0,0,0,0,0,0,0,0],
    [0,0,.8,0,0,.8,0,1,1,1,0,0,0,0,0,0],
    [0,0,.8,.8,.8,.8,0,1,1,1,0,0,0,0,0,0],
    [0,0,.8,0,0,.8,0,1,0,0,0,0,0,0,0,0],
    [0,0,.8,0,0,.8,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,.8,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,.4,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,.1,0,0,0,1,0,0,0,0],
    [0,0,.1,.2,.3,.4,0,0,1,0.5,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype = np.float32)# .repeat(2, 0).repeat(2, 1)

vs = np.zeros(weights.shape, dtype = np.float32)
vs.fill(0.04)

obstacles = np.stack((weights, vs), axis = -1)

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
], dtype = np.int32)#.repeat(2, 0).repeat(2, 1)

scene = vxl_scene(16, (128, 128), window = False)

scene.set_exterior(walls)
print(f'{scene.renderer.floor_height[None]=}')
scene.set_obstacles(obstacles)
scene.renderer.set_look_at(-0.3, -0.2, -0.016)
scene.renderer.set_camera_pos(1.78, -0.03, -0.015)
print('Configured scene')
raw_img = scene.render_scene()
print('Rendered scene')
# rimg = scene.random(0.1)
print('Converted to numpy array')
img = raw_img.to_numpy()
print(type(img))
print(img.shape)
scene.save_img(img, dirpath = "/spaths/tests/screenshots/")
print('Saved image')

sample = scene.random()
print('Sampled depth image')
print(sample.shape)
scene.save_img(sample, dirpath="/spaths/tests/screenshots/")

ll = scene.logpdf(sample)
print(f'log-likelihood = {ll}')
